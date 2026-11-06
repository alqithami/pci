"""Nonmutating admission oracle for the existing native custom warehouse.

This module changes neither environment dynamics nor admission priority. It
batches a *supplied* set of proposals and supports the native 1..32 agents.
Costs are institutional-stage Hamming replacements divided by N exactly once.
Full task/RNG state is deliberately absent from the lightweight oracle record.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import copy
import numpy as np

from .environment import EnvConfig, MOVES, geometry, FEATURES, VIEW_INDICES, OVERLAPS

SCHEMA = 'PCI-NATIVE-LIGHT-ADMISSION-STATE-v1'


def snapshot_oracle(env):
    """Capture only state needed by the correction predicate and tie breaking."""
    if hasattr(env, 'oracle_snapshot'):
        return env.oracle_snapshot()
    q, c = env.rules()
    cfg = asdict(env.config) if is_dataclass(env.config) else dict(env.config)
    return {'schema': SCHEMA, 'domain': 'custom', 'config': cfg,
            'positions': np.asarray(env.pos, dtype=np.int64).tolist(),
            'counts': np.asarray(env.counts, dtype=np.int64).tolist(),
            'wait': np.asarray(env.wait, dtype=np.int64).tolist(),
            'delivered': np.asarray(env.delivered, dtype=np.int64).tolist(),
            't': int(env.t), 'quotas': np.asarray(q, dtype=np.int64).tolist(),
            'capacities': np.asarray(c, dtype=np.int64).tolist()}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(value, shape, name):
    a = np.asarray(value)
    _require(a.shape == shape and a.dtype.kind in 'iu', name)
    return a.astype(np.int64, copy=True)


def _inside(positions, regions):
    x, y = positions[..., 0, None], positions[..., 1, None]
    return ((x >= regions[:, 0]) & (x <= regions[:, 1]) &
            (y >= regions[:, 2]) & (y <= regions[:, 3]))


def _validated_snapshot(value):
    s = snapshot_oracle(value) if not isinstance(value, dict) else copy.deepcopy(value)
    _require(s.get('schema') == SCHEMA and s.get('domain') == 'custom',
             'native custom light snapshot required')
    cfg = EnvConfig(**s['config'])
    n = cfg.n_agents
    _require(1 <= n <= 32 and 0 <= s['t'] < cfg.horizon, 'valid nonterminal N=1..32')
    old = _integer(s['positions'], (n, 2), 'positions[N,2]')
    counts = _integer(s['counts'], (n, 2), 'counts[N,2]')
    wait = _integer(s['wait'], (n,), 'wait[N]')
    delivered = _integer(s['delivered'], (n,), 'delivered[N]')
    quota = _integer(s['quotas'], (n, 2), 'quotas[N,2]')
    cap = _integer(s['capacities'], (2,), 'capacities[2]')
    expected_q = max(1, cfg.quota - int(cfg.rule_change and s['t'] >= cfg.horizon // 2))
    _require(np.all(quota == expected_q) and np.all(cap == cfg.capacity),
             'snapshot rules disagree with native schedule')
    walls, regions, *_ = geometry(cfg.layout)
    regions = np.asarray(regions, dtype=np.int64)
    _require(np.all(old >= 0) and np.all(old[:, 0] < walls.shape[1]) and
             np.all(old[:, 1] < walls.shape[0]), 'positions in native grid')
    _require(not walls[old[:, 1], old[:, 0]].any() and
             len(set(map(tuple, old))) == n, 'distinct native floor positions')
    _require(np.all(counts >= 0) and np.all(counts <= quota) and
             np.all(wait >= 0) and np.all(delivered >= 0) and
             np.all(quota >= 1) and np.all(cap >= 1), 'valid counters/rules')
    was = _inside(old, regions)
    _require(np.all(was.sum(axis=0) <= cap), 'pre-existing capacity infeasibility')
    return (s, old, counts, wait, delivered, regions, quota, cap, walls, was)


def _physical(candidate, old):
    """Native simultaneous stop cascade, batched over Q proposals."""
    n = len(old)
    nxt = candidate.copy()
    rejected = np.zeros(nxt.shape[:2], dtype=bool)
    for _ in range(n + 1):
        moved = (nxt != old).any(axis=-1)
        same = (nxt[:, :, None, :] == nxt[:, None, :, :]).all(axis=-1)
        goes_old = (nxt[:, :, None, :] == old[None, None, :, :]).all(axis=-1)
        conflict = same | (goes_old & goes_old.transpose(0, 2, 1))
        conflict[:, np.arange(n), np.arange(n)] = False
        bad = conflict.any(axis=-1) & moved
        if not bad.any():
            break
        nxt = np.where(bad[..., None], old, nxt)
        rejected |= bad
    same = (nxt[:, :, None, :] == nxt[:, None, :, :]).all(axis=-1)
    same[:, np.arange(n), np.arange(n)] = False
    _require(not same.any(), 'native physical fixed point not unique')
    return nxt, rejected


def _position_actions(positions, old):
    delta = positions - old
    _require(np.all(np.abs(delta).sum(axis=-1) <= 1), 'non-native displacement')
    dx, dy = delta[..., 0], delta[..., 1]
    return np.where(dx == 1, 1, np.where(dx == -1, 2,
                    np.where(dy == -1, 3, np.where(dy == 1, 4, 0)))).astype(np.int64)


def evaluate_native_proposals(snapshot, actions, *, _state=None):
    """Return every stage for actions[Q,N]; never mutate world or consume RNG."""
    state = _validated_snapshot(snapshot) if _state is None else _state
    _, old, counts, wait, delivered, regions, quota, cap, walls, was = state
    n = len(old)
    a = np.asarray(actions)
    _require(a.ndim == 2 and a.shape[1] == n and a.dtype.kind in 'iu' and
             np.all((a >= 0) & (a < 5)), 'integer actions[Q,N] in 0..4')
    a = a.astype(np.int64, copy=False)
    candidate = old + MOVES[a]
    _require(np.all(candidate >= 0) and np.all(candidate[..., 0] < walls.shape[1]) and
             np.all(candidate[..., 1] < walls.shape[0]), 'one-step targets in grid')
    wall = walls[candidate[..., 1], candidate[..., 0]]
    candidate = np.where(wall[..., None], old, candidate)
    physical, collision = _physical(candidate, old)

    def metrics(nxt):
        now = _inside(nxt, regions)
        entry = now & ~was
        qbad = ((counts + entry) > quota).any(axis=-1)
        excess = np.maximum(now.sum(axis=1) - cap, 0)
        return entry, qbad, excess

    _, raw_quota, raw_excess = metrics(physical)
    admitted = physical.copy()
    order = np.lexsort((np.arange(n), -delivered, wait))
    for _ in range(2 * n + 2):
        entry, qbad, excess = metrics(admitted)
        reject = qbad.copy()
        for r in range(2):
            entrants = entry[:, order, r]
            rank = np.cumsum(entrants, axis=1)
            reject[:, order] |= entrants & (rank <= excess[:, r, None])
        if not reject.any():
            _require(not excess.any(), 'infeasible current native capacity')
            break
        admitted = np.where(reject[..., None], old, admitted)
        admitted, _ = _physical(admitted, old)
    else:
        raise ValueError('native institutional fixed point failed')
    pa = _position_actions(physical, old)
    ia = _position_actions(admitted, old)
    pc = np.count_nonzero(a != pa, axis=1)
    ic = np.count_nonzero(pa != ia, axis=1)
    tc = np.count_nonzero(a != ia, axis=1)
    return {'actions': a.copy(), 'physical_actions': pa, 'admitted_actions': ia,
            'physical_positions': physical, 'admitted_positions': admitted,
            'wall_flags': wall, 'collision_flags': collision,
            'quota_flags': raw_quota, 'capacity_excess': raw_excess,
            'physical_replacements': pc, 'institutional_replacements': ic,
            'total_replacements': tc, 'physical_cost': pc.astype(float) / n,
            'institutional_cost': ic.astype(float) / n,
            'total_cost': tc.astype(float) / n,
            'institutional_bad_joint': raw_quota.any(axis=1) | raw_excess.any(axis=1),
            'raw_admissible': (pc == 0) & (ic == 0)}


def _conflict_possible(state):
    """Conservative all-proposal certificate, independent of sampled actions.

    Native physical resolution can only retain a wall-corrected one-step target
    or the old cell. If that superset cannot violate any institution rule, every
    institutional cost is exactly zero. The converse is deliberately not used.
    This says nothing about physical correction or raw-action admissibility.
    """
    _,old,counts,_,_,regions,quota,cap,walls,was=state
    targets=old[:,None,:]+MOVES[None,:,:]
    blocked=walls[targets[...,1],targets[...,0]]
    targets=np.where(blocked[...,None],old[:,None,:],targets)
    possible=_inside(targets,regions)
    entry=possible&~was[:,None,:]
    return bool(np.any(counts[:,None,:]+entry>quota[:,None,:]) or
                np.any(possible.any(axis=1).sum(axis=0)>cap))


class NativeWarehouseOracle:
    """Fixed-state batch oracle; normalized costs for actions[B,Q,N].

    ``stats`` separates requested proposals, evaluated proposal rows and logical
    rows established zero by a conservative all-action certificate. These are
    not elapsed work or individual scalar-native function invocations. Exact
    reuse, when desired, is explicit in ExactTableOracleCache rather than here.
    """
    def __init__(self, envs_or_snapshots, *, _stats=None):
        values = list(envs_or_snapshots)
        _require(bool(values), 'at least one state required')
        self._states = [_validated_snapshot(s) for s in values]
        self.snapshots = [s[0] for s in self._states]
        self._possible = [_conflict_possible(s) for s in self._states]
        ns = {int(s['config']['n_agents']) for s in self.snapshots}
        _require(len(ns) == 1, 'same N within oracle batch')
        self.n_agents = ns.pop()
        self.batch_size = len(values)
        self.n_actions = 5
        self.stats = _stats if _stats is not None else {
            'callback_calls': 0, 'requested_rows': 0, 'evaluated_rows': 0,
            'certified_zero_rows': 0}
        self.counters = self.stats

    @classmethod
    def from_snapshots(cls, snapshots):
        return cls(snapshots)

    def subset(self, indices):
        return type(self)([self.snapshots[int(i)] for i in indices], _stats=self.stats)

    def evaluate(self, actions):
        is_torch = hasattr(actions, 'detach')
        array = actions.detach().cpu().numpy() if is_torch else np.asarray(actions)
        _require(array.ndim == 3 and array.shape[0] == self.batch_size and
                 array.shape[2] == self.n_agents, 'actions[B,Q,N]')
        self.stats['callback_calls'] += 1
        self.stats['requested_rows'] += int(array.shape[0] * array.shape[1])
        self.stats['evaluated_rows'] += int(array.shape[0] * array.shape[1])
        rows = [evaluate_native_proposals(s, a, _state=state)
                for s,a,state in zip(self.snapshots,array,self._states)]
        return {name: np.stack([r[name] for r in rows]) for name in rows[0]}

    def __call__(self, actions):
        array=actions.detach().cpu().numpy() if hasattr(actions,'detach') else np.asarray(actions)
        _require(array.ndim==3 and array.shape[0]==self.batch_size and array.shape[2]==self.n_agents and
                 array.dtype.kind in 'iu' and np.all((array>=0)&(array<5)), 'integer actions[B,Q,N]')
        self.stats['callback_calls']+=1
        self.stats['requested_rows']+=int(array.shape[0]*array.shape[1])
        costs=np.zeros(array.shape[:2],dtype=np.float64)
        for b,(snapshot,state,possible) in enumerate(zip(self.snapshots,self._states,self._possible)):
            if possible:
                costs[b]=evaluate_native_proposals(snapshot,array[b],_state=state)['institutional_cost']
                self.stats['evaluated_rows']+=len(array[b])
            else:
                self.stats['certified_zero_rows']+=len(array[b])
        if hasattr(actions, 'detach'):
            import torch
            return torch.as_tensor(costs, device=actions.device, dtype=torch.float64)
        return costs

    def admissible(self, actions, *, require_physical=True):
        r = self.evaluate(actions)
        mask = r['institutional_replacements'] == 0
        if require_physical:
            mask &= r['physical_replacements'] == 0
        if hasattr(actions, 'detach'):
            import torch
            return torch.as_tensor(mask, device=actions.device, dtype=torch.bool)
        return mask


CUSTOM_FEATURE_SPEC = {'feature_dim': len(FEATURES), 'action_count': 5,
    'feature_names': FEATURES, 'action_names': ('stay', 'east', 'west', 'north', 'south'),
    'view_indices': VIEW_INDICES, 'overlaps': OVERLAPS,
    'institution': 'regional standing-occupancy capacity and per-agent entry quota'}


def make_environment(domain, config, seed):
    """Construct the explicitly named domain without replacing its physics."""
    if domain in ('warehouse', 'custom'):
        from .environment import Warehouse
        cfg = EnvConfig(**config) if isinstance(config, dict) else config
        env = Warehouse(cfg, seed)
        env.feature_spec = copy.deepcopy(CUSTOM_FEATURE_SPEC)
        return env
    if domain == 'rware':
        from .rware_adapter import RwareInstitution
        return RwareInstitution(config, seed)
    raise ValueError('domain must be warehouse/custom or rware')


def make_oracle(envs_or_snapshots, domain=None):
    values = list(envs_or_snapshots)
    if not values:
        raise ValueError('nonempty fixed-state batch required')
    first = values[0]
    actual = first.get('domain') if isinstance(first, dict) else (
        'rware' if hasattr(first, 'native') and hasattr(first, 'oracle_snapshot') else 'custom')
    if domain is not None and domain not in ((actual,) if actual == 'rware' else ('custom', 'warehouse')):
        raise ValueError('declared domain disagrees with fixed-state snapshot')
    if actual == 'rware':
        from .rware_adapter import RwareOracle
        return RwareOracle(values)
    if actual == 'custom':
        return NativeWarehouseOracle(values)
    raise ValueError('unsupported oracle snapshot domain')


class ExactTableOracleCache:
    """N4 fixed-state cost/feasibility cache with explicit work accounting.

    Construct only for a protocol that actually requires the full exact table.
    It deliberately rejects N!=4 and never pretends sampled estimators require
    enumeration. Precomputation visits each requested state once in bounded
    chunks. Policies/gradients never enter a cache key: the contained snapshots
    are frozen, and only the state-dependent checker result is retained.
    ``requested_rows`` includes precomputation and later logical lookups;
    ``evaluated_rows`` excludes cache hits and certified-zero logical rows.
    ``precompute_rows`` counts initial logical table entries, including zeros
    established by a conservative certificate. Do not add it to evaluated work.
    """
    def __init__(self, oracle, *, chunk_states=32, include_admissibility=True):
        if oracle.n_agents != 4 or oracle.n_actions != 5:
            raise ValueError('exact table cache is deliberately restricted to N4,A5')
        if not isinstance(chunk_states, int) or chunk_states < 1:
            raise ValueError('positive precompute state chunk required')
        self.oracle = oracle
        self.n_agents, self.n_actions, self.batch_size = 4, 5, oracle.batch_size
        self.snapshots = copy.deepcopy(oracle.snapshots)
        self.counters = {'callback_calls': 0, 'requested_rows': 0, 'evaluated_rows': 0,
                         'cache_hit_rows': 0, 'precompute_rows': 0}
        self.stats = self.counters
        self.cost_table = np.empty((self.batch_size, 625), dtype=np.float64)
        self.valid_table = np.empty((self.batch_size, 625), dtype=bool) if include_admissibility else None
        grid = np.indices((5,)*4, dtype=np.int64).reshape(4, -1).T.copy()
        for first in range(0, self.batch_size, chunk_states):
            last = min(first+chunk_states, self.batch_size)
            selected = oracle.subset(range(first,last))
            proposals = np.broadcast_to(grid, (last-first,625,4))
            stage = self._base_call(selected, proposals, cost_only=not include_admissibility)
            self.cost_table[first:last] = stage['institutional_cost'] if include_admissibility else stage
            if include_admissibility:
                self.valid_table[first:last] = ((stage['physical_replacements']==0) &
                                                (stage['institutional_replacements']==0))
            self.counters['precompute_rows'] += (last-first)*625
        self.cost_table.setflags(write=False)
        if self.valid_table is not None:
            self.valid_table.setflags(write=False)

    def _base_call(self, oracle, actions, *, cost_only=False, admissible_only=False,
                   require_physical=True):
        before = dict(oracle.counters)
        result = (oracle(actions) if cost_only else oracle.admissible(actions,require_physical=require_physical)
                  if admissible_only else oracle.evaluate(actions))
        for key, value in oracle.counters.items():
            difference = value-before.get(key,0)
            self.counters[key] = self.counters.get(key,0)+difference
        return result

    def subset(self, indices):
        indices = np.asarray(list(indices), dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or np.any(indices<0) or np.any(indices>=self.batch_size):
            raise ValueError('nonempty valid cache subset indices')
        child = object.__new__(type(self))
        child.oracle = self.oracle.subset(indices)
        child.n_agents, child.n_actions, child.batch_size = 4, 5, len(indices)
        child.snapshots = [self.snapshots[int(i)] for i in indices]
        child.cost_table = self.cost_table[indices]
        child.valid_table = self.valid_table[indices] if self.valid_table is not None else None
        child.cost_table.setflags(write=False)
        if child.valid_table is not None:child.valid_table.setflags(write=False)
        child.counters = self.counters;child.stats = self.counters
        return child

    def _lookup(self, actions, table):
        a = actions.detach().cpu().numpy() if hasattr(actions,'detach') else np.asarray(actions)
        if a.ndim!=3 or a.shape[0]!=self.batch_size or a.shape[2]!=4 or a.dtype.kind not in 'iu' or np.any((a<0)|(a>4)):
            raise ValueError('integer cache actions[B,Q,4] in0..4')
        index = (a*np.array((125,25,5,1))).sum(-1)
        value = np.take_along_axis(table,index,axis=1)
        self.counters['callback_calls'] += 1
        self.counters['requested_rows'] += int(a.shape[0]*a.shape[1])
        self.counters['cache_hit_rows'] += int(a.shape[0]*a.shape[1])
        if hasattr(actions,'detach'):
            import torch
            return torch.as_tensor(value,device=actions.device,
                dtype=torch.bool if value.dtype==bool else torch.float64)
        return value

    def __call__(self, actions):
        return self._lookup(actions,self.cost_table)

    def admissible(self, actions, *, require_physical=True):
        if require_physical and self.valid_table is None:
            return self._base_call(self.oracle,actions,admissible_only=True,require_physical=True)
        table = self.valid_table if require_physical else self.cost_table==0
        return self._lookup(actions,table)

    def evaluate(self, actions):
        return self._base_call(self.oracle,actions)
