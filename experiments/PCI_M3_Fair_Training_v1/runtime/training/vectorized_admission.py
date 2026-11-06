"""Exact all-action admission costs for a valid four-agent warehouse state.

This is a vectorized transcription of the pinned native operational checker,
not a new admission policy. Only the 625 joint proposals are batched. Every
proposal retains the original iterative physical correction, quota/capacity
priority and post-institutional physical cascade. No task transition, policy
sample, world constructor, RNG access, or mutation of the supplied world occurs.

The caller must supply a nonterminal, admissible native Warehouse before its
next action, with any scheduled quota reset already reflected in counts. This
strict contract rejects pre-existing infeasibility instead of repairing it.
"""
from __future__ import annotations

import numpy as np

PINNED_NATIVE_SHA256 = "cc2ba5ad437b517f9aa3ab96119ce3fa3dba2c51098a7ff2d77ef6977f3e059c"
MOVES = np.asarray(((0, 0), (1, 0), (-1, 0), (0, -1), (0, 1)), dtype=np.int64)
ACTIONS = np.indices((5, 5, 5, 5), dtype=np.int64).reshape(4, -1).T.copy()
MOVES.setflags(write=False)
ACTIONS.setflags(write=False)


class AdmissionInputError(ValueError):
    """An input is outside the explicitly supported native state contract."""


def _require(condition, label):
    if not condition:
        raise AdmissionInputError(label)


def _integers(value, shape, label):
    result = np.asarray(value)
    _require(result.shape == shape and result.dtype.kind == "i", label)
    return result


def _inside(positions, regions):
    x, y = positions[..., 0, None], positions[..., 1, None]
    return ((x >= regions[:, 0]) & (x <= regions[:, 1]) &
            (y >= regions[:, 2]) & (y <= regions[:, 3]))


def _physical(candidate, old):
    """Batch the native n+1 fixed-point iterations across joint proposals."""
    nxt = candidate.copy()
    rejected = np.zeros(nxt.shape[:2], dtype=bool)
    for _ in range(5):
        moved = (nxt != old).any(axis=-1)
        same = (nxt[:, :, None, :] == nxt[:, None, :, :]).all(axis=-1)
        goes_to_old = (nxt[:, :, None, :] == old[None, None, :, :]).all(axis=-1)
        conflict = same | (goes_to_old & goes_to_old.transpose(0, 2, 1))
        conflict[:, np.arange(4), np.arange(4)] = False
        bad = conflict.any(axis=-1) & moved
        if not bad.any():
            break
        nxt = np.where(bad[..., None], old, nxt)
        rejected |= bad
    else:
        # Native code also checks uniqueness after its bounded loop. A further
        # pass here is validation only and does not change the supplied result.
        same = (nxt[:, :, None, :] == nxt[:, None, :, :]).all(axis=-1)
        same[:, np.arange(4), np.arange(4)] = False
        _require(not same.any(), "PHYSICAL_FIXED_POINT_INFEASIBLE")
    return nxt, rejected


def _position_actions(positions, old):
    delta = positions - old
    dx, dy = delta[..., 0], delta[..., 1]
    _require(np.all(np.abs(dx) + np.abs(dy) <= 1), "NON_NATIVE_DISPLACEMENT")
    return np.where(dx == 1, 1, np.where(dx == -1, 2,
                    np.where(dy == -1, 3, np.where(dy == 1, 4, 0)))).astype(np.int64)


def _validate_state(env):
    _require(env.config.n_agents == 4, "EXACT_FOUR_AGENT_INPUT")
    _require(0 <= env.t < env.config.horizon, "NONTERMINAL_PREACTION_INPUT")
    old = _integers(env.pos, (4, 2), "INTEGER_POSITIONS_4_BY_2")
    counts = _integers(env.counts, (4, 2), "INTEGER_COUNTS_4_BY_2")
    wait = _integers(env.wait, (4,), "INTEGER_WAIT_4")
    delivered = _integers(env.delivered, (4,), "INTEGER_DELIVERED_4")
    regions = _integers(env.regions, (2, 4), "TWO_RECTANGULAR_REGIONS")
    quota, capacity = env.rules()
    quota = _integers(quota, (4, 2), "INTEGER_QUOTAS_4_BY_2")
    capacity = _integers(capacity, (2,), "INTEGER_CAPACITIES_2")
    walls = np.asarray(env.walls)
    _require(walls.ndim == 2 and walls.dtype == bool, "BOOLEAN_WALL_GRID")
    _require(np.all(old >= 0) and np.all(old[:, 0] < walls.shape[1]) and
             np.all(old[:, 1] < walls.shape[0]), "POSITIONS_IN_GRID")
    _require(not walls[old[:, 1], old[:, 0]].any(), "POSITIONS_ON_FLOOR")
    _require(len(set(map(tuple, old.tolist()))) == 4, "UNIQUE_OLD_POSITIONS")
    _require(np.all(counts >= 0) and np.all(counts <= quota) and
             np.all(quota >= 1) and np.all(capacity >= 1) and
             np.all(wait >= 0) and np.all(delivered >= 0), "VALID_PREACTION_COUNTERS")
    was = _inside(old, regions)
    _require(np.all(was.sum(axis=0) <= capacity), "VALID_PREACTION_CAPACITY")
    return old, counts, wait, delivered, regions, quota, capacity, walls, was


def _institutional_conflict_possible(state):
    """Conservative possibility bound before physical resolution.

    Each agent's five wall-corrected targets include its old cell (stay).
    Physical resolution can only replace a target by that old cell, so every
    resolved position belongs to this superset. If no single target can exceed
    a quota and the number of agents able to occupy a region does not exceed
    its capacity, every joint proposal is institutionally valid after physical
    resolution. The converse is deliberately not claimed: collisions may make
    a seemingly possible institutional conflict impossible.
    """
    old, counts, _, _, regions, quota, capacity, walls, was = state
    targets = old[:, None, :] + MOVES[None, :, :]
    _require(np.all(targets >= 0) and np.all(targets[..., 0] < walls.shape[1]) and
             np.all(targets[..., 1] < walls.shape[0]), "ONE_STEP_TARGETS_IN_GRID")
    blocked = walls[targets[..., 1], targets[..., 0]]
    targets = np.where(blocked[..., None], old[:, None, :], targets)
    possible_inside = _inside(targets, regions)
    possible_entry = possible_inside & ~was[:, None, :]
    quota_possible = np.any((counts[:, None, :] + possible_entry) > quota[:, None, :])
    capacity_possible = np.any(possible_inside.any(axis=1).sum(axis=0) > capacity)
    return bool(quota_possible or capacity_possible)


def _evaluate(env, *, _validated_state=None):
    state = _validate_state(env) if _validated_state is None else _validated_state
    old, counts, wait, delivered, regions, quota, capacity, walls, was = state
    candidate = old + MOVES[ACTIONS]
    _require(np.all(candidate >= 0) and np.all(candidate[..., 0] < walls.shape[1]) and
             np.all(candidate[..., 1] < walls.shape[0]), "ONE_STEP_TARGETS_IN_GRID")
    wall = walls[candidate[..., 1], candidate[..., 0]]
    candidate = np.where(wall[..., None], old, candidate)
    physical, collision = _physical(candidate, old)

    def metrics(nxt):
        now = _inside(nxt, regions)
        entry = now & ~was
        qbad = ((counts + entry) > quota).any(axis=-1)
        excess = np.maximum(now.sum(axis=1) - capacity, 0)
        return entry, qbad, excess

    _, quota_flags, capacity_excess = metrics(physical)
    nxt = physical.copy()
    # Sorting the same four-agent priority once is equivalent to sorting each
    # row/region's entrant subset by the identical total order.
    order = np.lexsort((np.arange(4), -delivered, wait))
    for _ in range(10):
        entry, qbad, excess = metrics(nxt)
        reject = qbad.copy()
        for region in range(2):
            entrants = entry[:, order, region]
            ranks = np.cumsum(entrants, axis=1)
            reject[:, order] |= entrants & (ranks <= excess[:, region, None])
        if not reject.any():
            _require(not excess.any(), "INFEASIBLE_CURRENT_CAPACITY")
            return physical, nxt, wall, collision, quota_flags, capacity_excess
        # Quota and BOTH region-capacity reject sets are accumulated against
        # the same candidate, then applied once before physical resolution.
        nxt = np.where(reject[..., None], old, nxt)
        nxt, _ = _physical(nxt, old)
    raise AdmissionInputError("INSTITUTIONAL_FIXED_POINT_DID_NOT_CONVERGE")


def admission_table(env):
    """Return the exact 625-row table; replacement fields are RAW counts.

    The *_cost fields normalize those Hamming counts by four exactly once.
    The original native source identity is pinned in the external validation
    receipt, rather than reading and hashing a source file on each live step.
    Returned collision/quota/capacity flags refer to the PHYSICAL stage before
    institutional admission, as in the native checker and retained M1 arrays.
    """
    physical, admitted, wall, collision, quota, capacity = _evaluate(env)
    physical_actions = _position_actions(physical, env.pos)
    admitted_actions = _position_actions(admitted, env.pos)
    p = np.count_nonzero(ACTIONS != physical_actions, axis=1)
    i = np.count_nonzero(physical_actions != admitted_actions, axis=1)
    t = np.count_nonzero(ACTIONS != admitted_actions, axis=1)
    return {"actions": ACTIONS.copy(), "physical_actions": physical_actions,
            "admitted_actions": admitted_actions, "physical_positions": physical,
            "admitted_positions": admitted, "wall_flags": wall,
            "collision_flags": collision, "quota_flags": quota,
            "capacity_excess": capacity, "physical_replacements": p,
            "institutional_replacements": i, "total_replacements": t,
            "physical_cost": p.astype(np.float64) / 4.,
            "institutional_cost": i.astype(np.float64) / 4.,
            "total_cost": t.astype(np.float64) / 4.,
            "institutional_bad_joint": quota.any(axis=1) | capacity.any(axis=1)}


def institutional_replacement_counts(env):
    """Return 625 RAW institutional Hamming counts for an expected-risk loss.

    A conservative per-agent target-set certificate skips exhaustive physical
    enumeration only when all joint actions must have zero institutional cost.
    Otherwise the complete exact checker is evaluated. Physical-only correction
    costs are never inferred from this zero-institutional certificate.
    """
    state = _validate_state(env)
    if not _institutional_conflict_possible(state):
        return np.zeros(625, dtype=np.int64)
    physical, admitted, *_ = _evaluate(env, _validated_state=state)
    # Each corrected position uniquely determines its native action at fixed
    # old positions. Comparing positions therefore gives the same Hamming count.
    return np.count_nonzero((physical != admitted).any(axis=-1), axis=1)
