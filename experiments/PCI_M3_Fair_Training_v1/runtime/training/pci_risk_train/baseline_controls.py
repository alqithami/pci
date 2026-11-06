"""Explicit, independently testable controls for the fair-training extension.

This module does not claim a new constrained optimization algorithm. The cost
surrogate is the pessimistic clipped PPO surrogate; feasibility is assessed on
observed institutional replacement cost, not inferred from a fitted cost value.
The NumPy routines are calculation references, not substitutes for optimizer tests.
"""
from __future__ import annotations

import math
import numpy as np


def projected_dual_update(dual, observed_cost, target, learning_rate, maximum):
    """Projected ascent for a rollout-mean cost constraint, in cost/agent-step."""
    values = (dual, observed_cost, target, learning_rate, maximum)
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Nonfinite dual-update input')
    if not (0 <= observed_cost <= 1 and 0 <= target <= 1 and
            0 <= dual <= maximum and learning_rate > 0 and maximum > 0):
        raise ValueError('Invalid normalized institutional-cost dual update')
    return float(np.clip(dual + learning_rate * (observed_cost - target), 0, maximum))


def gae_numpy(rewards, values, dones, last_values, gamma=.99, lam=.95):
    """Terminal-masked GAE reference; rewards/values [T,E,N], dones [T,E]."""
    r, v, d, last = map(lambda x: np.asarray(x, dtype=np.float64),
                         (rewards, values, dones, last_values))
    if r.shape != v.shape or r.ndim != 3 or d.shape != r.shape[:2] or last.shape != r.shape[1:]:
        raise ValueError('GAE dimensions differ')
    if not all(np.isfinite(x).all() for x in (r, v, d, last)) or not np.isin(d, (0, 1)).all():
        raise ValueError('Invalid GAE input')
    advantage = np.zeros_like(r); carry = np.zeros_like(last)
    for t in range(len(r) - 1, -1, -1):
        nxt = last if t == len(r) - 1 else v[t + 1]
        live = (1 - d[t])[..., None]
        carry = r[t] + gamma * nxt * live - v[t] + gamma * lam * live * carry
        advantage[t] = carry
    return advantage, advantage + v


def clipped_policy_surrogates(logratio, reward_advantage, cost_advantage, clip_coef):
    """Return minimized reward loss and conservative positive cost surrogate.

    Inputs may be NumPy arrays or Torch tensors. In particular the cost branch
    is MAX, not MIN: minimizing it cannot exploit clipping to hide a cost rise.
    No advantage normalization occurs here.
    """
    if not 0 < clip_coef < 1:
        raise ValueError('Invalid PPO clip coefficient')
    if hasattr(logratio, 'detach'):
        import torch
        ratio = logratio.exp(); clipped = ratio.clamp(1 - clip_coef, 1 + clip_coef)
        return (torch.maximum(-ratio * reward_advantage, -clipped * reward_advantage).mean(),
                torch.maximum(ratio * cost_advantage, clipped * cost_advantage).mean())
    ratio = np.exp(np.asarray(logratio)); clipped = np.clip(ratio, 1 - clip_coef, 1 + clip_coef)
    return (float(np.maximum(-ratio * reward_advantage, -clipped * reward_advantage).mean()),
            float(np.maximum(ratio * cost_advantage, clipped * cost_advantage).mean()))


def normalize_reward_and_scale_cost(reward_advantage, cost_advantage, epsilon=1e-8):
    """Center reward advantages; scale BOTH signals by the same reward SD.

    The cost advantage has no independent standardization. Centering it is not
    needed; its fitted cost-value baseline already supplies a state baseline.
    This preserves a declared common reward/cost scale within each rollout.
    """
    if hasattr(reward_advantage, 'detach'):
        scale = reward_advantage.std(unbiased=False).clamp_min(epsilon)
        return ((reward_advantage - reward_advantage.mean()) / scale,
                cost_advantage / scale, scale)
    r = np.asarray(reward_advantage, dtype=np.float64)
    scale = max(float(r.std()), epsilon)
    return (r - r.mean()) / scale, np.asarray(cost_advantage) / scale, scale


def component_gradient_diagnostics(components, parameters):
    """Pre-clipping full-parameter component norms and selected cosines.

    ``components`` must contain the already weighted contributions to the actual
    minimized loss. autograd.grad does not populate parameter .grad fields.
    Retaining the graph permits the one subsequent actual loss.backward().
    """
    import torch
    parameters = tuple(p for p in parameters if p.requires_grad)
    vectors = {}
    for name, loss in components.items():
        if loss.requires_grad:
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True,
                                            allow_unused=True)
        else:
            gradients = (None,) * len(parameters)
        vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                                   for p, g in zip(parameters, gradients)])
    result = {name + '_grad_norm': float(torch.linalg.vector_norm(v).detach().cpu())
              for name, v in vectors.items()}
    ppo = vectors['reward_policy'] + vectors['cost_policy']
    ppo_norm = torch.linalg.vector_norm(ppo)
    aux = vectors['auxiliary']; aux_norm = torch.linalg.vector_norm(aux)
    result['ppo_policy_grad_norm'] = float(ppo_norm.detach().cpu())
    result['auxiliary_to_ppo_grad_norm'] = (float((aux_norm / ppo_norm).detach().cpu())
                                           if float(ppo_norm) > 0 else None)
    result['auxiliary_ppo_grad_cosine'] = (
        float((torch.dot(aux, ppo) / (aux_norm * ppo_norm)).detach().cpu())
        if float(aux_norm) > 0 and float(ppo_norm) > 0 else None)
    summed = sum(vectors.values())
    result['summed_component_grad_norm'] = float(torch.linalg.vector_norm(summed).detach().cpu())
    if not all(v is None or math.isfinite(v) for v in result.values()):
        raise RuntimeError('Nonfinite component gradient diagnostic')
    return result


def gradient_norm(parameters):
    import torch
    gradients = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    if not gradients:
        return 0.
    return float(torch.linalg.vector_norm(torch.cat(gradients)).cpu())
