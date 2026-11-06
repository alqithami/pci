"""Native RWARE 2.0.0 with an explicit, separate entry institution.

The real upstream reset/step govern physics, shelf pickup/delivery and task RNG.
Native actions remain NOOP/FORWARD/LEFT/RIGHT/TOGGLE_LOAD, not custom grid moves.
The institution is explicitly selected: legacy service-half entry quotas, or
geometry-only transport row bands. Both limit simultaneous entries and per-
agent window entries, not the custom warehouse's standing occupancy. The task
reward defaults to the exact upstream INDIVIDUAL native reward; optional
terminal-aware potential shaping is explicit and telescoping. Repeated pickups
or drops receive no nonpotential bonus.
Counterfactual movement is a batched transcription of the native functional
movement graph; it must be qualified against upstream step before experiments.
No RWARE action is claimed to be certified by the custom-warehouse circuit.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import copy
import numpy as np

from .environment import D, FEATURES, VIEW_INDICES, OVERLAPS

RWARE_SCHEMA = 'PCI-RWARE-LIGHT-ADMISSION-STATE-v1'
NATIVE_ACTIONS = ('NOOP', 'FORWARD', 'LEFT', 'RIGHT', 'TOGGLE_LOAD')
DIRECTION_MOVES = np.array(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=np.int64)
RWARE_FEATURES = FEATURES + ('announced_quota_0', 'announced_quota_1',
                             'announced_entry_capacity_0', 'announced_entry_capacity_1')


@dataclass(frozen=True)
class RwareConfig:
    layout: str = 'tiny'
    n_agents: int = 4
    horizon: int = 256
    quota_window: int = 64
    quota: int = 3
    capacity: int = 2
    rule_change: bool = False
    zone_mode: str = 'service_halves'
    reward_mode: str = 'native'
    shaping_gamma: float = .99
    shaping_scale: float = .05


def _inside(pos, grid_size, zone_mode='service_halves'):
    h, w = grid_size
    if zone_mode=='transport_bands':
        # Geometry-only one-row crossing bands; native floor/traffic dynamics
        # remain authoritative. Each band includes every column at its row.
        return np.stack((pos[...,1]==h//3,pos[...,1]==(2*h)//3),axis=-1)
    if zone_mode!='service_halves':
        raise ValueError('zone_mode must be service_halves or transport_bands')
    service = pos[..., 1] >= h - 3
    return np.stack((service & (pos[..., 0] < w // 2),
                     service & (pos[..., 0] >= w // 2)), axis=-1)


def _validate(s):
    if s.get('schema') != RWARE_SCHEMA or s.get('domain') != 'rware':
        raise ValueError('light native RWARE snapshot required')
    cfg = RwareConfig(**s['config']); n = cfg.n_agents
    if cfg.zone_mode not in ('service_halves','transport_bands'):
        raise ValueError('unknown announced RWARE institution zone mode')
    if cfg.reward_mode not in ('native','potential') or not np.isfinite(cfg.shaping_gamma) or not 0 < cfg.shaping_gamma <= 1 or not np.isfinite(cfg.shaping_scale) or cfg.shaping_scale < 0:
        raise ValueError('invalid announced RWARE reward mode/potential parameters')
    if not 1 <= n <= 32 or not 0 <= int(s['t']) < cfg.horizon:
        raise ValueError('nonterminal RWARE N1..32 required')
    old = np.asarray(s['positions'], dtype=np.int64)
    dirs = np.asarray(s['directions'], dtype=np.int64)
    carrying = np.asarray(s['carrying'], dtype=bool)
    counts = np.asarray(s['counts'], dtype=np.int64)
    quota = np.asarray(s['quotas'], dtype=np.int64)
    caps = np.asarray(s['capacities'], dtype=np.int64)
    h, w = map(int, s['grid_size'])
    shelves = np.asarray(s['shelf_occupied'], dtype=bool)
    if old.shape != (n, 2) or dirs.shape != (n,) or carrying.shape != (n,) or counts.shape != (n, 2) or quota.shape != (n, 2) or caps.shape != (2,) or shelves.shape != (h, w):
        raise ValueError('malformed RWARE state dimensions')
    if np.any(old < 0) or np.any(old[:, 0] >= w) or np.any(old[:, 1] >= h) or len(set(map(tuple, old))) != n or np.any((dirs < 0) | (dirs > 3)) or np.any(counts < 0) or np.any(counts > quota):
        raise ValueError('invalid RWARE positions/directions/counters')
    q = max(1, cfg.quota - int(cfg.rule_change and int(s['t']) >= cfg.horizon // 2))
    if np.any(quota != q) or np.any(caps != cfg.capacity) or np.any(caps < 1):
        raise ValueError('RWARE rules disagree with announced schedule')
    return cfg, old, dirs, carrying, counts, quota, caps, (h, w), shelves


def rware_conflict_possible(snapshot):
    """All-proposal entry bound; does not certify native physical feasibility."""
    cfg,old,dirs,carrying,counts,quota,caps,(h,w),shelves=_validate(snapshot)
    target=old+DIRECTION_MOVES[dirs]
    target=np.clip(target,(0,0),(w-1,h-1))
    lookup=np.full((h,w),-1,dtype=np.int64);lookup[old[:,1],old[:,0]]=np.arange(len(old))
    occupant=lookup[target[:,1],target[:,0]]
    loaded_target=(occupant>=0)&carrying[occupant.clip(0)]
    blocked=carrying&(target!=old).any(-1)&shelves[target[:,1],target[:,0]]&~loaded_target
    target=np.where(blocked[:,None],old,target)
    entry=_inside(target,(h,w),cfg.zone_mode)&~_inside(old,(h,w),cfg.zone_mode)
    return bool(np.any(counts+entry>quota) or np.any(entry.sum(0)>caps))


def rware_physical(snapshot, actions):
    """Batched native 2.0 movement graph, without native task/RNG mutation.

    Functional components either contain a cycle or end in an empty cell.
    Native commits cycle members (except two-cycles), or one longest incoming
    path to an empty cell. Ambiguous equal-length paths use the exact native
    NetworkX graph procedure: subgraph-view iteration can alter tie order, so
    agent-index tie breaking alone is not equivalent to upstream step.
    """
    cfg, old, dirs, carrying, _, _, _, grid_size, shelves = _validate(snapshot)
    a = np.asarray(actions)
    n = len(old)
    if a.ndim != 2 or a.shape[1] != n or a.dtype.kind not in 'iu' or np.any((a < 0) | (a > 4)):
        raise ValueError('integer native actions[Q,N] in0..4')
    a = a.astype(np.int64, copy=False); q = len(a); h, w = grid_size
    unclipped = old[None, :, :] + np.where((a == 1)[..., None], DIRECTION_MOVES[dirs][None, :, :], 0)
    wall = ((unclipped[..., 0] < 0) | (unclipped[..., 0] >= w) |
            (unclipped[..., 1] < 0) | (unclipped[..., 1] >= h))
    targets = unclipped.copy()
    targets[..., 0] = targets[..., 0].clip(0, w - 1)
    targets[..., 1] = targets[..., 1].clip(0, h - 1)
    lookup = np.full((h, w), -1, dtype=np.int64)
    lookup[old[:, 1], old[:, 0]] = np.arange(n)
    target_agent = lookup[targets[..., 1], targets[..., 0]]
    target_loaded = (target_agent >= 0) & carrying[target_agent.clip(0)]
    blocked_shelf = (carrying[None, :] & (targets != old).any(-1) &
                     shelves[targets[..., 1], targets[..., 0]] & ~target_loaded)
    targets = np.where(blocked_shelf[..., None], old, targets)
    edge = (targets[:, :, None, :] == old[None, None, :, :]).all(-1)
    next_idx = np.where(edge.any(-1), edge.argmax(-1), -1)
    base = np.broadcast_to(np.arange(n), (q, n))
    cur = base.copy(); cycle_length = np.zeros((q, n), dtype=np.int64)
    for k in range(1, n + 1):
        cur = np.where(cur >= 0, np.take_along_axis(next_idx, cur.clip(0), axis=1), -1)
        hit = (cur == base) & (cycle_length == 0)
        cycle_length[hit] = k
    committed = (cycle_length == 1) | (cycle_length >= 3)
    # Distances/pointers for acyclic components. Values on cycle components
    # are ignored; they cannot lead to an empty-cell endpoint.
    dist = np.zeros((q, n), dtype=np.int64)
    for _ in range(n):
        scores = np.where(edge, dist[:, :, None] + 1, -1000000)
        dist = np.maximum(scores.max(axis=1), 0)
    best_prev = scores.argmax(axis=1)
    has_prev = edge.any(axis=1)
    empty = next_idx < 0
    same_sink = (targets[:, :, None, :] == targets[:, None, :, :]).all(-1)
    idx = np.arange(n)
    better = ((dist[:, None, :] > dist[:, :, None]) |
              ((dist[:, None, :] == dist[:, :, None]) & (idx[None, None, :] < idx[None, :, None])))
    winner = empty & ~(same_sink & empty[:, None, :] & better).any(axis=-1)
    frontier = winner.copy(); committed |= frontier
    back = best_prev[:, :, None] == idx[None, None, :]
    back &= has_prev[:, :, None]
    for _ in range(n):
        frontier = (frontier[:, :, None] & back).any(axis=1)
        committed |= frontier
    # Native NetworkX subgraph copies can reorder predecessor nodes when a
    # component is small relative to the complete graph. Only ambiguous paths
    # need its graph procedure; all other rows remain fully batched. This is
    # exact graph selection, not a simulator/world deepcopy or task transition.
    max_scores = scores.max(axis=1)
    ambiguous_internal = (((scores == max_scores[:, None, :]).sum(axis=1) > 1) & has_prev).any(axis=1)
    sink_tie = (same_sink & empty[:, :, None] & empty[:, None, :] &
                (dist[:, :, None] == dist[:, None, :]))
    sink_tie[:, idx, idx] = False
    fallback = ambiguous_internal | sink_tie.any(axis=(1, 2))
    if fallback.any():
        import networkx as nx
        old_index = {tuple(p): i for i, p in enumerate(old)}
        for row in np.flatnonzero(fallback):
            graph = nx.DiGraph()
            for source, target in zip(old, targets[row]):
                graph.add_edge(tuple(source), tuple(target))
            chosen = np.zeros(n, dtype=bool)
            for component in (graph.subgraph(c).copy() for c in nx.weakly_connected_components(graph)):
                try:
                    cycle = nx.algorithms.find_cycle(component)
                    if len(cycle) == 2:
                        continue
                    nodes = [edge[0] for edge in cycle]
                except nx.NetworkXNoCycle:
                    nodes = nx.algorithms.dag_longest_path(component)
                for node in nodes:
                    if node in old_index:
                        chosen[old_index[node]] = True
            committed[row] = chosen
    positions = np.where((committed & (a == 1))[..., None], targets, old)
    effective = np.where((a == 1) & ~(positions != old).any(-1), 0, a)
    # Orientation/toggle commands keep their native meanings and are never
    # replaced merely because they leave Cartesian position unchanged.
    return {'positions': positions, 'effective_actions': effective,
            'wall_flags': wall | blocked_shelf,
            'collision_flags': (a == 1) & ~wall & ~blocked_shelf & ~committed,
            'native_graph_tie_fallback': fallback}


def evaluate_rware_proposals(snapshot, actions):
    cfg, old, _, _, counts, quota, caps, grid_size, _ = _validate(snapshot)
    a = np.asarray(actions)
    raw = rware_physical(snapshot, a)
    was = _inside(old, grid_size,cfg.zone_mode)

    def predicate(positions):
        entries = _inside(positions, grid_size,cfg.zone_mode) & ~was
        qbad = ((counts + entries) > quota).any(-1)
        excess = np.maximum(entries.sum(axis=1) - caps, 0)
        return entries, qbad, excess

    _, raw_qbad, raw_excess = predicate(raw['positions'])
    final = a.copy(); current = raw
    fallback_count = raw['native_graph_tie_fallback'].astype(np.int64)
    resolution_rows = np.ones(len(a), dtype=np.int64)
    for _ in range(2 * cfg.n_agents + 2):
        entries, qbad, excess = predicate(current['positions'])
        reject = qbad.copy()
        # Preserve prior external institution: higher-index entrants are
        # canceled first. This is distinct from the custom wait/served order.
        for r in range(2):
            ranks = np.cumsum(entries[:, ::-1, r], axis=1)[:, ::-1]
            reject |= entries[:, :, r] & (ranks <= excess[:, r, None])
        if not reject.any():
            if excess.any():
                raise ValueError('infeasible RWARE entry limit')
            break
        final[reject] = 0
        current = rware_physical(snapshot, final)
        fallback_count += current['native_graph_tie_fallback']
        resolution_rows += 1
    else:
        raise ValueError('RWARE institutional correction did not converge')
    pa, ia = raw['effective_actions'], current['effective_actions']
    pc = np.count_nonzero(a != pa, axis=1)
    ic = np.count_nonzero(pa != ia, axis=1)
    tc = np.count_nonzero(a != ia, axis=1)
    return {'actions': a.copy(), 'physical_actions': pa, 'admitted_actions': ia,
            'admitted_native_actions': final, 'physical_positions': raw['positions'],
            'admitted_positions': current['positions'], 'wall_flags': raw['wall_flags'],
            'collision_flags': raw['collision_flags'], 'quota_flags': raw_qbad,
            'capacity_excess': raw_excess, 'physical_replacements': pc,
            'institutional_replacements': ic, 'total_replacements': tc,
            'physical_cost': pc.astype(float) / cfg.n_agents,
            'institutional_cost': ic.astype(float) / cfg.n_agents,
            'total_cost': tc.astype(float) / cfg.n_agents,
            'institutional_bad_joint': raw_qbad.any(axis=1) | raw_excess.any(axis=1),
            'raw_admissible': (pc == 0) & (ic == 0),
            'native_graph_tie_fallback_evaluations': fallback_count,
            'physical_resolution_rows': resolution_rows}


class RwareInstitution:
    """Real native dynamics with public task/rule features common to all arms."""
    feature_spec = {'feature_dim': 27, 'action_count': 5,
                    'feature_names': RWARE_FEATURES,
                    'action_names': NATIVE_ACTIONS,
                    'view_indices': tuple(tuple(sorted(set(v) | {23, 24, 25, 26})) for v in VIEW_INDICES),
                    'overlaps': OVERLAPS,
                    'institution': 'explicit announced zone_mode entry quota and simultaneous-entry capacity',
                    'task_reward': 'announced reward_mode: native INDIVIDUAL or terminal-aware potential shaping; no pickup bonus'}

    def __init__(self, config, seed):
        from rware.warehouse import Warehouse as Native, RewardType
        from importlib.metadata import version
        if version('rware') != '2.0.0':
            raise RuntimeError('RWARE2.0.0 is the qualified native transition version')
        self.config = RwareConfig(**config) if isinstance(config, dict) else config
        if self.config.layout not in ('tiny', 'small') or not 1 <= self.config.n_agents <= 32:
            raise ValueError('native RWARE tiny/small and N1..32 required')
        if self.config.horizon < 2 or min(self.config.quota_window, self.config.quota, self.config.capacity) < 1:
            raise ValueError('positive RWARE horizon/rule parameters required')
        if self.config.zone_mode not in ('service_halves','transport_bands'):
            raise ValueError('unknown RWARE zone mode')
        if self.config.reward_mode not in ('native','potential') or not np.isfinite(self.config.shaping_gamma) or not 0 < self.config.shaping_gamma <= 1 or not np.isfinite(self.config.shaping_scale) or self.config.shaping_scale < 0:
            raise ValueError('invalid RWARE reward mode/potential parameters')
        self.rng = np.random.default_rng(seed); self.episode = -1
        self.native = Native(shelf_columns=3, column_height=8,
            shelf_rows=1 if self.config.layout == 'tiny' else 2, n_agents=self.n,
            msg_bits=0, sensor_range=1, request_queue_size=2*self.n,
            max_inactivity_steps=None, max_steps=self.config.horizon,
            reward_type=RewardType.INDIVIDUAL)
        self.reset()

    @property
    def n(self):
        return self.config.n_agents

    @property
    def pos(self):
        return np.array([(a.x, a.y) for a in self.native.agents], dtype=np.int64)

    def rules(self):
        q = max(1, self.config.quota - int(self.config.rule_change and self.t >= self.config.horizon // 2))
        return np.full((self.n, 2), q, dtype=np.int64), np.full(2, self.config.capacity, dtype=np.int64)

    def reset(self):
        self.episode += 1; self.t = 0
        self.native_seed = int(self.rng.integers(0, 2**31 - 1))
        self.native.reset(seed=self.native_seed)
        self.counts = np.zeros((self.n, 2), dtype=np.int64)
        self.delivered = np.zeros(self.n, dtype=np.int64)
        self.wait = np.zeros(self.n, dtype=np.int64); self.last = None
        return self.features()

    def oracle_snapshot(self):
        q, c = self.rules()
        return {'schema': RWARE_SCHEMA, 'domain': 'rware', 'config': asdict(self.config),
                't': self.t, 'positions': self.pos.tolist(),
                'directions': [a.dir.value for a in self.native.agents],
                'carrying': [a.carrying_shelf is not None for a in self.native.agents],
                'shelf_occupied': (self.native.grid[1] != 0).tolist(),
                'grid_size': list(self.native.grid_size), 'counts': self.counts.tolist(),
                'wait': self.wait.tolist(), 'delivered': self.delivered.tolist(),
                'quotas': q.tolist(), 'capacities': c.tolist()}

    def _target(self, agent):
        requested = {s.id for s in self.native.request_queue}
        if agent.carrying_shelf is not None:
            if agent.carrying_shelf.id in requested:
                candidates = [tuple(g) for g in self.native.goals]
            else:
                h, w = self.native.grid_size
                candidates = [(x, y) for y in range(h) for x in range(w)
                    if not self.native._is_highway(x, y) and
                    self.native.grid[1, y, x] in (0, agent.carrying_shelf.id)]
        else:
            candidates = [(s.x, s.y) for s in self.native.request_queue]
        if not candidates:
            raise RuntimeError('no native task/return destination')
        return min(candidates, key=lambda p: (abs(p[0]-agent.x)+abs(p[1]-agent.y), p[1], p[0]))

    @staticmethod
    def _potential(pos, direction, goal):
        delta = DIRECTION_MOVES[int(direction)]
        dx, dy = goal[0]-pos[0], goal[1]-pos[1]
        aligned = delta[0]*dx + delta[1]*dy > 0
        return abs(dx)+abs(dy)+(.25 if (dx or dy) and not aligned else 0.)

    def features(self):
        q, cap = self.rules(); old = self.pos; h, w = self.native.grid_size
        now = _inside(old, (h, w),self.config.zone_mode); occupancy = now.sum(0)
        f = np.zeros((self.n, 5, 27), dtype=np.float32)
        requested = {s.id for s in self.native.request_queue}
        # Pure kinematics are features, not a substitute transition simulator.
        turn_order = (0, 3, 1, 2)
        for i, agent in enumerate(self.native.agents):
            goal = self._target(agent); loaded = agent.carrying_shelf is not None
            d0 = self._potential(old[i], agent.dir.value, goal)
            for action in range(5):
                direction = agent.dir.value
                if action in (2, 3):
                    direction = turn_order[(turn_order.index(direction) + (-1 if action == 2 else 1)) % 4]
                nxt = old[i] + (DIRECTION_MOVES[agent.dir.value] if action == 1 else 0)
                bad = not (0 <= nxt[0] < w and 0 <= nxt[1] < h)
                nxt = np.clip(nxt, (0, 0), (w-1, h-1))
                if loaded and action == 1 and not np.array_equal(nxt, old[i]):
                    sid, aid = self.native.grid[1, nxt[1], nxt[0]], self.native.grid[0, nxt[1], nxt[0]]
                    bad |= bool(sid and not (aid and self.native.agents[aid-1].carrying_shelf is not None))
                if bad:
                    nxt = old[i]
                entry = _inside(nxt[None], (h, w),self.config.zone_mode)[0] & ~now[i]
                near = any(j != i and np.array_equal(p, nxt) for j, p in enumerate(old))
                useful_toggle = action == 4 and ((not loaded and self.native.grid[1, agent.y, agent.x] in requested) or
                    (loaded and agent.carrying_shelf.id not in requested and not self.native._is_highway(agent.x, agent.y)))
                progress = d0 - self._potential(nxt, direction, goal) + float(useful_toggle)
                deliver = loaded and agent.carrying_shelf.id in requested and tuple(nxt) in self.native.goals
                f[i, action] = [progress, float(bad), float(near), *(nxt-old[i]), *entry.astype(float),
                    *((self.counts[i]+entry > q[i]).astype(float)), float(deliver), float(loaded),
                    (self.t % self.config.quota_window)/self.config.quota_window,
                    *np.maximum(q[i]-self.counts[i], 0)/q[i], min(self.wait[i]/self.config.horizon, 1),
                    (self.delivered[i]+1)/(self.delivered.sum()+self.n),
                    agent.x/(w-1), agent.y/(h-1), goal[0]/(w-1), goal[1]/(h-1), d0/(h+w),
                    *occupancy/np.maximum(cap, 1), *q[i]/3., *cap/self.n]
        if not np.isfinite(f).all():
            raise ValueError('nonfinite native RWARE features')
        return f

    def potential(self):
        """Bounded full-state public-goal potential, not an extra task reward.

        Every native target is in the finite grid and the orientation term is
        at most .25, hence |Phi_i| <= H+W+.25. Native request/shelf/carry state
        determines the target even when a pickup/delivery changes that target.
        Those discontinuities do not break the telescoping identity.
        """
        return -np.array([self._potential((a.x,a.y),a.dir.value,self._target(a))
                          for a in self.native.agents],dtype=np.float64)

    def transition(self, actions, *, enforce=False, certifier=None):
        if certifier is not None:
            raise ValueError('no RWARE proof circuit is claimed')
        if self.t >= self.config.horizon:
            raise RuntimeError('reset required')
        a = np.asarray(actions)
        if a.shape != (self.n,) or a.dtype.kind not in 'iu' or np.any((a < 0) | (a > 4)):
            raise ValueError('integer native actions[N] in0..4')
        before = self.oracle_snapshot(); old = self.pos
        phi_before=self.potential() if self.config.reward_mode=='potential' else None
        native_rng_before = copy.deepcopy(self.native.np_random.bit_generator.state)
        requested_before = [s.id for s in self.native.request_queue]
        stage = evaluate_rware_proposals(before, a[None])
        commands = stage['admitted_native_actions'][0] if enforce else a
        carries = [x.carrying_shelf.id if x.carrying_shelf else None for x in self.native.agents]
        requested = {s.id for s in self.native.request_queue}
        result = self.native.step(commands.tolist())
        expected = stage['admitted_positions'][0] if enforce else stage['physical_positions'][0]
        if not np.array_equal(self.pos, expected):
            raise AssertionError('batched RWARE physical prediction disagrees with actual native step: '+str({
                'step': self.t, 'before': before, 'proposal': a.tolist(),
                'commands': commands.tolist(), 'expected': expected.tolist(),
                'actual': self.pos.tolist()}))
        nr = np.asarray(result[1], dtype=float)
        if np.any(nr < 0) or not np.allclose(nr, np.rint(nr)):
            raise AssertionError('unexpected native individual task reward')
        deliveries = np.rint(nr).astype(np.int64)
        after_carries = [x.carrying_shelf.id if x.carrying_shelf else None for x in self.native.agents]
        pickups = np.array([b is None and c in requested for b, c in zip(carries, after_carries)])
        drops = np.array([b is not None and c is None for b, c in zip(carries, after_carries)])
        terminal=self.t+1==self.config.horizon
        phi_after=(np.zeros(self.n,dtype=np.float64) if terminal else self.potential()) if phi_before is not None else None
        shaping=self.config.shaping_scale*(self.config.shaping_gamma*phi_after-phi_before) if phi_before is not None else np.zeros(self.n,dtype=np.float64)
        task = nr+shaping
        entries = _inside(self.pos,self.native.grid_size,self.config.zone_mode) & ~_inside(old,self.native.grid_size,self.config.zone_mode)
        quota, caps = self.rules()
        actual_ok = not np.any(self.counts + entries > quota) and not np.any(entries.sum(0) > caps)
        if enforce and not actual_ok:
            raise AssertionError('native applied RWARE action violates entry institution')
        self.counts += entries; self.delivered += deliveries; self.wait += 1; self.wait[deliveries > 0] = 0
        if len(set(map(tuple, self.pos))) != self.n:
            raise AssertionError('actual native physical collision')
        metrics = {'delivered': int(deliveries.sum()), 'pickups': int(pickups.sum()),
            'native_toggle_drops':int(drops.sum()),
            'task_reward': float(task.sum()), 'native_task_reward': float(nr.sum()),
            'potential_shaping_reward':float(shaping.sum()),
            'proposed_compliant': int(stage['raw_admissible'][0]), 'executed_compliant': int(actual_ok),
            'wall_attempts': int(stage['wall_flags'][0].sum()),
            'collision_attempts': int(stage['collision_flags'][0].sum()),
            'quota_violations': int(stage['quota_flags'][0].sum()),
            'capacity_excess': int(stage['capacity_excess'][0].sum()),
            'interventions': int(stage['total_replacements'][0]) if enforce else int(stage['physical_replacements'][0]),
            'physical_replacements': int(stage['physical_replacements'][0]),
            'institutional_replacements': int(stage['institutional_replacements'][0]) if enforce else 0,
            'total_replacements': int(stage['total_replacements'][0]) if enforce else int(stage['physical_replacements'][0]),
            'actual_vertex_collisions': 0, 'move_count': int((old != self.pos).any(-1).sum()),
            'max_wait': int(self.wait.max())}
        costs = stage['wall_flags'][0].astype(float) + stage['collision_flags'][0].astype(float) + stage['quota_flags'][0].astype(float) + stage['capacity_excess'][0].sum()/self.n
        self.last = {'before': before, 'episode': self.episode, 'step': self.t,
            'native_reset_seed': self.native_seed, 'native_rng_state_before': native_rng_before,
            'native_rng_state_after': copy.deepcopy(self.native.np_random.bit_generator.state),
            'requested_shelf_ids_before': requested_before,
            'requested_shelf_ids_after': [s.id for s in self.native.request_queue],
            'reward_mode':self.config.reward_mode,'shaping_gamma':self.config.shaping_gamma,
            'shaping_scale':self.config.shaping_scale,
            'potential_before':phi_before.tolist() if phi_before is not None else None,
            'potential_after_for_shaping':phi_after.tolist() if phi_after is not None else None,
            'potential_shaping_rewards':shaping.tolist(),
            'actions': a.tolist(),
            'admitted_actions': commands.tolist(), 'canonical_admitted_actions': stage['admitted_actions'][0].tolist(),
            'positions_after': self.pos.tolist(), 'native_rewards': nr.tolist(), 'metrics': metrics}
        self.t += 1
        if self.t % self.config.quota_window == 0:
            self.counts.fill(0)
        done = self.t == self.config.horizon
        if bool(result[2]) != done or bool(result[3]):
            raise AssertionError('native and wrapper horizon termination disagree')
        return self.features(), task.astype(np.float32), costs.astype(np.float32), done, metrics

    def state_dict(self):
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        self.__dict__.update(copy.deepcopy(state))


class RwareOracle:
    def __init__(self, envs_or_snapshots, *, _stats=None):
        self.snapshots = [x.oracle_snapshot() if hasattr(x, 'oracle_snapshot') else copy.deepcopy(x) for x in envs_or_snapshots]
        if not self.snapshots:
            raise ValueError('nonempty RWARE state batch required')
        ns = {_validate(s)[0].n_agents for s in self.snapshots}
        if len(ns) != 1:
            raise ValueError('same N in RWARE oracle batch')
        self.n_agents = ns.pop(); self.n_actions = 5; self.batch_size = len(self.snapshots)
        self._possible=[rware_conflict_possible(s) for s in self.snapshots]
        self.stats = _stats if _stats is not None else {'callback_calls': 0, 'requested_rows': 0, 'evaluated_rows': 0,
            'native_graph_tie_fallback_evaluations': 0, 'physical_resolution_rows': 0,
            'certified_zero_rows': 0}
        self.counters = self.stats

    @classmethod
    def from_snapshots(cls, snapshots):
        return cls(snapshots)

    def subset(self, indices):
        return type(self)([self.snapshots[int(i)] for i in indices], _stats=self.stats)

    def evaluate(self, actions):
        a = actions.detach().cpu().numpy() if hasattr(actions, 'detach') else np.asarray(actions)
        if a.ndim != 3 or a.shape[0] != self.batch_size or a.shape[2] != self.n_agents:
            raise ValueError('RWARE actions[B,Q,N] required')
        self.stats['callback_calls'] += 1
        self.stats['requested_rows'] += int(a.shape[0] * a.shape[1]); self.stats['evaluated_rows'] += int(a.shape[0] * a.shape[1])
        rows = [evaluate_rware_proposals(s, x) for s, x in zip(self.snapshots, a)]
        self.stats['native_graph_tie_fallback_evaluations'] += sum(int(r['native_graph_tie_fallback_evaluations'].sum()) for r in rows)
        self.stats['physical_resolution_rows'] += sum(int(r['physical_resolution_rows'].sum()) for r in rows)
        return {key: np.stack([r[key] for r in rows]) for key in rows[0]}

    def __call__(self, actions):
        a=actions.detach().cpu().numpy() if hasattr(actions,'detach') else np.asarray(actions)
        if a.ndim!=3 or a.shape[0]!=self.batch_size or a.shape[2]!=self.n_agents or a.dtype.kind not in 'iu' or np.any((a<0)|(a>4)):
            raise ValueError('integer RWARE actions[B,Q,N] in0..4 required')
        self.stats['callback_calls']+=1;self.stats['requested_rows']+=int(a.shape[0]*a.shape[1])
        result=np.zeros(a.shape[:2],dtype=np.float64)
        for b,(snapshot,possible) in enumerate(zip(self.snapshots,self._possible)):
            if possible:
                stage=evaluate_rware_proposals(snapshot,a[b]);result[b]=stage['institutional_cost']
                self.stats['evaluated_rows']+=len(a[b])
                self.stats['native_graph_tie_fallback_evaluations']+=int(stage['native_graph_tie_fallback_evaluations'].sum())
                self.stats['physical_resolution_rows']+=int(stage['physical_resolution_rows'].sum())
            else:self.stats['certified_zero_rows']+=len(a[b])
        if hasattr(actions, 'detach'):
            import torch
            return torch.as_tensor(result, device=actions.device, dtype=torch.float64)
        return result

    def admissible(self, actions, *, require_physical=True):
        r = self.evaluate(actions); mask = r['institutional_replacements'] == 0
        if require_physical:
            mask &= r['physical_replacements'] == 0
        if hasattr(actions, 'detach'):
            import torch
            return torch.as_tensor(mask, device=actions.device, dtype=torch.bool)
        return mask
