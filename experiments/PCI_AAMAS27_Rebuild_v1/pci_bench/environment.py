"""A documented, discrete multi-robot warehouse; not RWARE or physical robotics.

Actions: stay, east, west, north, south. Tasks require an actual pickup and delivery.
Synchronous movement prevents vertex collisions and swaps for ALL controllers.
Institutional capacity/quota admission is a separate optional layer. Both proposed
and executed violations are measured; blocked collisions are never called collisions.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
from functools import lru_cache
import copy
import numpy as np

MOVES = np.array([[0, 0], [1, 0], [-1, 0], [0, -1], [0, 1]], dtype=np.int64)
FEATURES = (
    'progress', 'wall', 'occupied_target', 'dx', 'dy', 'entry_0', 'entry_1',
    'quota_excess_0', 'quota_excess_1', 'delivery_if_reached', 'carrying',
    'window_phase', 'remaining_0', 'remaining_1', 'waiting_age', 'served_share',
    'x', 'y', 'goal_x', 'goal_y', 'distance', 'resource_occupancy_0', 'resource_occupancy_1',
)
D = len(FEATURES)
# View-specific observations, fixed before experiments. All include task direction.
VIEW_INDICES = (
    (0, 1, 2, 3, 4, 5, 6, 10, 11, 16, 17, 18, 19, 20, 21, 22),
    (0, 1, 2, 3, 4, 9, 10, 11, 16, 17, 18, 19, 20),
    (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
)
# Edge stalks contain expected, physically interpretable action effects.
OVERLAPS = ((0, 1, (3, 4, 2)), (0, 2, (5, 6)), (1, 2, (0, 9)))

@dataclass(frozen=True)
class EnvConfig:
    layout: str = 'twodoors'
    n_agents: int = 4
    horizon: int = 256
    quota_window: int = 64
    quota: int = 3
    capacity: int = 2
    rule_change: bool = False

@lru_cache(maxsize=32)
def geometry(layout: str):
    size = 13
    walls = np.zeros((size, size), dtype=bool)
    walls[[0, -1], :] = True
    walls[:, [0, -1]] = True
    if layout == 'open':
        for y in (4, 8):
            walls[y, 3:5] = True
            walls[y, 8:10] = True
        regions = ((5, 7, 3, 3), (5, 7, 9, 9))
    elif layout == 'twodoors':
        walls[1:-1, 6] = True
        walls[3, 6] = walls[9, 6] = False
        regions = ((5, 7, 3, 3), (5, 7, 9, 9))
    elif layout == 'staggered':
        walls[1:-1, 4] = True
        walls[3, 4] = walls[9, 4] = False
        walls[1:-1, 8] = True
        walls[5, 8] = walls[9, 8] = False
        regions = ((3, 5, 3, 3), (7, 9, 5, 5))
    else:
        raise ValueError(f'Unknown layout {layout!r}')
    pickups = ((2, 2), (2, 4), (2, 8), (2, 10))
    depots = ((10, 3), (10, 9))
    floor = tuple((x, y) for y in range(size) for x in range(size) if not walls[y, x])
    # All-pairs shortest paths on this immutable geometry, independent of traffic.
    distance = {}
    for goal in pickups + depots:
        arr = np.full_like(walls, 10000, dtype=np.int64)
        arr[goal[1], goal[0]] = 0
        q = deque([goal])
        while q:
            x, y = q.popleft()
            for dx, dy in MOVES[1:]:
                xx, yy = x + int(dx), y + int(dy)
                if 0 <= xx < size and 0 <= yy < size and not walls[yy, xx] and arr[yy, xx] == 10000:
                    arr[yy, xx] = arr[y, x] + 1
                    q.append((xx, yy))
        if any(arr[y, x] == 10000 for x, y in floor):
            raise AssertionError(f'Disconnected map: {layout}')
        distance[goal] = arr
    walls.setflags(write=False)
    return walls, regions, pickups, depots, floor, distance


def inside(positions, regions):
    pos = np.asarray(positions, dtype=np.int64)
    return np.stack([(pos[..., 0] >= lo_x) & (pos[..., 0] <= hi_x) &
                     (pos[..., 1] >= lo_y) & (pos[..., 1] <= hi_y)
                     for lo_x, hi_x, lo_y, hi_y in regions], axis=-1)


def monitor_record(record: dict) -> dict:
    """Independent scalar specification. No calls to Warehouse's transition code."""
    cfg = record['config']
    walls, regions, *_ = geometry(cfg['layout'])
    old = record['before']['positions']
    acts = record['actions']
    count = record['before']['counts']
    quota, cap = record['quotas'], record['capacities']
    n = len(old)
    if len(acts) != n or len(count) != n or len(quota) != n:
        raise ValueError('Malformed record dimensions')
    nxt, errors = [], []
    for i in range(n):
        a = acts[i]
        if not isinstance(a, int) or not 0 <= a < 5:
            raise ValueError(f'Invalid action {i}: {a}')
        x, y = old[i]
        dx, dy = map(int, MOVES[a])
        xx, yy = x + dx, y + dy
        if not (0 <= x < len(walls) and 0 <= y < len(walls)) or walls[y, x]:
            errors.append(f'old_floor:{i}')
        if not (0 <= xx < len(walls) and 0 <= yy < len(walls)) or walls[yy, xx]:
            errors.append(f'floor:{i}')
        nxt.append([xx, yy])
    for i in range(n):
        for j in range(i):
            if old[i] == old[j]: errors.append(f'old_vertex:{j}:{i}')
            if nxt[i] == nxt[j]: errors.append(f'vertex:{j}:{i}')
            if nxt[i] == old[j] and nxt[j] == old[i]: errors.append(f'swap:{j}:{i}')
    entries = [[0] * len(regions) for _ in old]
    for r, (lx, hx, ly, hy) in enumerate(regions):
        total = 0
        for i, ((x,y), (xx,yy)) in enumerate(zip(old, nxt)):
            was = lx <= x <= hx and ly <= y <= hy
            now = lx <= xx <= hx and ly <= yy <= hy
            total += int(now)
            entries[i][r] = int(now and not was)
            if not isinstance(count[i][r], int) or count[i][r] < 0:
                errors.append(f'negative_count:{i}:{r}')
            if count[i][r] + entries[i][r] > quota[i][r]:
                errors.append(f'quota:{i}:{r}')
        if total > cap[r]: errors.append(f'capacity:{r}')
    return {'compliant': not errors, 'errors': errors, 'next_positions': nxt, 'entries': entries}

class Warehouse:
    def __init__(self, config: EnvConfig, seed: int):
        if not 1 <= config.n_agents <= 32: raise ValueError('n_agents must be 1..32')
        if config.horizon < 2 or config.quota_window < 1 or config.quota < 1 or config.capacity < 1:
            raise ValueError('Invalid episode or institution parameters')
        self.config = config
        self.walls, self.regions, self.pickups, self.depots, self.floor, self.distances = geometry(config.layout)
        self.rng = np.random.default_rng(seed)
        self.episode = -1
        self.reset()

    @property
    def n(self): return self.config.n_agents

    def reset(self):
        self.episode += 1
        self.t = 0
        allowed = [p for p in self.floor if not inside(np.array([p]), self.regions).any()]
        selected = self.rng.choice(len(allowed), self.n, replace=False)
        self.pos = np.array([allowed[int(k)] for k in selected], dtype=np.int64)
        self.carrying = np.zeros(self.n, dtype=bool)
        self.delivered = np.zeros(self.n, dtype=np.int64)
        self.wait = np.zeros(self.n, dtype=np.int64)
        self.counts = np.zeros((self.n, 2), dtype=np.int64)
        # Per-agent exogenous task streams are independent of the policies' RNGs.
        self.tasks = self.rng.integers(0, len(self.pickups), size=(self.n, self.config.horizon + 2))
        self.destinations = self.rng.integers(0, len(self.depots), size=(self.n, self.config.horizon + 2))
        self.task_index = np.zeros(self.n, dtype=np.int64)
        self.goals = np.array([self.pickups[self.tasks[i, 0]] for i in range(self.n)], dtype=np.int64)
        self.last = None
        return self.features()

    def rules(self):
        # Rule shift is a quota reduction at an already scheduled quota reset.
        changed = self.config.rule_change and self.t >= self.config.horizon // 2
        quota = max(1, self.config.quota - int(changed))
        return np.full((self.n, 2), quota, dtype=np.int64), np.full(2, self.config.capacity, dtype=np.int64)

    def candidates(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.n,) or not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions > 4)):
            raise ValueError(f'actions must have shape ({self.n},) and integer codes 0..4')
        raw = self.pos + MOVES[actions]
        bad = self.walls[raw[:, 1], raw[:, 0]]  # valid old positions ensure candidates stay within array
        raw[bad] = self.pos[bad]
        return raw, bad

    def physical_resolve(self, candidate):
        nxt = np.array(candidate, copy=True)
        rejected = np.zeros(self.n, dtype=bool)
        for _ in range(self.n + 1):
            bad = np.zeros(self.n, dtype=bool)
            for i in range(self.n):
                for j in range(i):
                    same = np.array_equal(nxt[i], nxt[j])
                    swap = np.array_equal(nxt[i], self.pos[j]) and np.array_equal(nxt[j], self.pos[i])
                    if same or swap:
                        # If a stationary agent occupies a cell, only the mover can be stopped.
                        bad[i] |= not np.array_equal(nxt[i], self.pos[i])
                        bad[j] |= not np.array_equal(nxt[j], self.pos[j])
            if not bad.any(): break
            nxt[bad] = self.pos[bad]
            rejected |= bad
        if len({tuple(p) for p in nxt}) != self.n: raise AssertionError('Physical collision after resolution')
        return nxt, rejected

    def institution_metrics(self, nxt):
        was, now = inside(self.pos, self.regions), inside(nxt, self.regions)
        entry = now & ~was
        q, c = self.rules()
        per_agent = ((self.counts + entry) > q).any(axis=1)
        excess = np.maximum(now.sum(axis=0) - c, 0)
        return entry, per_agent, excess

    def shield(self, candidate):
        nxt, _ = self.physical_resolve(candidate)
        q, cap = self.rules()
        if np.any(self.counts > q):
            raise RuntimeError('Cannot certify an already over-budget state; start a fresh shielded episode')
        for _ in range(2 * self.n + 2):
            entry, qbad, excess = self.institution_metrics(nxt)
            reject = qbad.copy()
            now = inside(nxt, self.regions)
            for r in range(2):
                if excess[r] > 0:
                    entrants = np.flatnonzero(entry[:, r])
                    # Reject newer / more-served entrants first; deterministic index tie-break.
                    order = sorted(entrants, key=lambda i: (int(self.wait[i]), -int(self.delivered[i]), int(i)))
                    reject[order[:int(excess[r])]] = True
            if not reject.any():
                if excess.any(): raise RuntimeError('Infeasible capacity in current state')
                return nxt
            nxt[reject] = self.pos[reject]
            nxt, _ = self.physical_resolve(nxt)
        raise RuntimeError('Shield did not converge')

    def actions_for_positions(self, nxt):
        delta = np.asarray(nxt) - self.pos
        result = []
        for d in delta:
            matches = np.flatnonzero((MOVES == d).all(axis=1))
            if len(matches) != 1: raise AssertionError('Illegal displacement')
            result.append(int(matches[0]))
        return result

    def record_for_actions(self, actions):
        q, cap = self.rules()
        return {'config': asdict(self.config), 'episode': self.episode, 'step': self.t,
                'before': {'positions': self.pos.tolist(), 'counts': self.counts.tolist()},
                'actions': list(map(int, actions)), 'quotas': q.tolist(), 'capacities': cap.tolist()}

    def transition(self, actions, *, enforce=False, certifier=None):
        if self.t >= self.config.horizon: raise RuntimeError('reset required after episode termination')
        candidate, wall = self.candidates(actions)
        physical, collision = self.physical_resolve(candidate)
        _, raw_qbad, raw_cap = self.institution_metrics(physical)
        raw_compliant = not (wall.any() or collision.any() or raw_qbad.any() or raw_cap.any())
        nxt = self.shield(candidate) if enforce else physical
        applied_actions = self.actions_for_positions(nxt)
        cert_record = self.record_for_actions(applied_actions)
        independent = monitor_record(cert_record)
        if enforce and not independent['compliant']:
            raise AssertionError(f'Shield/monitor disagreement: {independent["errors"]}')
        if certifier is not None:
            if not enforce: raise ValueError('Cryptographic admission requires enforce=True')
            if certifier(copy.deepcopy(cert_record)) is not True:
                raise RuntimeError('Admission rejected: no transition was executed')
        old_dist = np.array([self.distances[tuple(g)][p[1], p[0]] for p,g in zip(self.pos, self.goals)])
        new_dist = np.array([self.distances[tuple(g)][p[1], p[0]] for p,g in zip(nxt, self.goals)])
        moved = (nxt != self.pos).any(axis=1)
        entry = inside(nxt, self.regions) & ~inside(self.pos, self.regions)
        self.counts += entry.astype(np.int64)
        self.pos = nxt
        reached = (self.pos == self.goals).all(axis=1)
        delivery = reached & self.carrying
        pickup = reached & ~self.carrying
        self.delivered += delivery.astype(np.int64)
        self.wait += 1
        self.wait[delivery] = 0
        for i in np.flatnonzero(reached):
            if delivery[i]:
                self.task_index[i] += 1
                self.carrying[i] = False
                self.goals[i] = self.pickups[self.tasks[i, self.task_index[i]]]
            else:
                self.carrying[i] = True
                self.goals[i] = self.depots[self.destinations[i, self.task_index[i]]]
        task_reward = delivery.astype(float) + 0.2*pickup + 0.05*(old_dist-new_dist) - 0.01*moved
        violation_cost = (wall.astype(float) + collision.astype(float) + raw_qbad.astype(float)
                          + float(raw_cap.sum())/self.n)
        metrics = {'delivered': int(delivery.sum()), 'pickups': int(pickup.sum()),
                   'task_reward': float(task_reward.sum()), 'proposed_compliant': int(raw_compliant),
                   'executed_compliant': int(independent['compliant']),
                   'wall_attempts': int(wall.sum()), 'collision_attempts': int(collision.sum()),
                   'quota_violations': int(raw_qbad.sum()), 'capacity_excess': int(raw_cap.sum()),
                   'interventions': int(np.sum(np.asarray(actions) != np.asarray(applied_actions))),
                   'actual_vertex_collisions': self.n-len({tuple(p) for p in self.pos}),
                   'move_count': int(moved.sum()), 'max_wait': int(self.wait.max())}
        cert_record['after_positions'] = self.pos.tolist()
        self.last = cert_record
        self.t += 1
        # Reset happens before the next action; publicly attested state contains reset counters.
        if self.t % self.config.quota_window == 0: self.counts.fill(0)
        return self.features(), task_reward.astype(np.float32), violation_cost.astype(np.float32), self.t == self.config.horizon, metrics

    def features(self):
        q, cap = self.rules()
        f = np.zeros((self.n, 5, D), dtype=np.float32)
        now = inside(self.pos, self.regions)
        region_occupancy = now.sum(axis=0)
        for i in range(self.n):
            x, y = self.pos[i]
            goal = tuple(self.goals[i])
            dist = self.distances[goal]
            d0 = dist[y, x]
            for a, (dx,dy) in enumerate(MOVES):
                xx, yy = x+dx, y+dy
                wall = bool(self.walls[yy,xx])
                if wall: xx,yy=x,y
                near = any(j!=i and (self.pos[j] == (xx,yy)).all() for j in range(self.n))
                entered = inside(np.array([[xx,yy]]),self.regions)[0] & ~now[i]
                f[i,a] = [float(d0-dist[yy,xx]), float(wall), float(near), float(dx), float(dy),
                          *entered.astype(float), *((self.counts[i]+entered > q[i]).astype(float)),
                          float(self.carrying[i] and (xx,yy)==goal), float(self.carrying[i]),
                          (self.t % self.config.quota_window)/self.config.quota_window,
                          *np.maximum(q[i]-self.counts[i],0)/np.maximum(q[i],1),
                          min(self.wait[i]/self.config.horizon,1),
                          (self.delivered[i]+1)/(self.delivered.sum()+self.n),
                          x/12,y/12,goal[0]/12,goal[1]/12,d0/30,
                          *region_occupancy/np.maximum(cap,1)]
        if not np.isfinite(f).all(): raise AssertionError('Nonfinite observation')
        return f

    def state_dict(self):
        return {k: copy.deepcopy(v) for k,v in self.__dict__.items() if k not in
                ('walls','regions','pickups','depots','floor','distances','rng') } | {'rng_state': copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self,state):
        state=copy.deepcopy(state)
        self.rng.bit_generator.state=state.pop('rng_state')
        self.__dict__.update(state)
