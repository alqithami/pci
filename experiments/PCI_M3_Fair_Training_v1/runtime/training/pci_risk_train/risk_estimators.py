"""Fixed-state expected correction cost under a *product* proposal policy.

The checker callback accepts integer joint proposals ``[B,Q,N]`` and returns
already normalized institutional costs ``[B,Q]`` in [0,1].  For the warehouse
this is institutional-stage Hamming count / N, excluding physical corrections.
No normalization is applied again here.  Costs, states, and sampled actions are
fixed during differentiation.  Repeated PPO epochs sample from the CURRENT
probabilities supplied to each call, never silently from stored old actions.

Exact summation, score gradients, and conditional enumeration are conventional
finite-distribution estimators.  This module claims no new estimation theorem.
Its costs and gradients concern a fixed state, not the state-visitation law.
The callback is responsible for immutable snapshots and actual checker timing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    import torch
except ModuleNotFoundError:  # NumPy-only mathematics checks can import this file.
    torch = None


@dataclass(frozen=True)
class RiskEstimate:
    """Per-state auxiliary and detached risk estimate with requested work.

    ``loss.mean()`` is the training auxiliary.  A sampled loss is a
    value-corrected surrogate: its *value* estimates J, while its gradient is
    the chosen unbiased fixed-state gradient estimate.  ``queries`` counts
    requested joint-action rows, not native checker calls after zero shortcuts
    or caching. ``callback_calls`` counts actual Python callback invocations.
    """

    loss: Any
    value: Any
    queries: int
    callback_calls: int
    estimator: str
    samples: int
    baseline: str


def _probabilities(probabilities):
    if torch is None:
        raise ImportError('PyTorch is required for differentiable risk estimators')
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError('probabilities must be a Torch tensor [B,N,A]')
    if probabilities.ndim != 3 or probabilities.shape[1] < 1 or probabilities.shape[2] < 2:
        raise ValueError('Expected product-policy probabilities [B,N,A], N>=1, A>=2')
    if not probabilities.is_floating_point():
        raise TypeError('Policy probabilities must have a floating-point dtype')
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise ValueError('Policy probabilities must be finite and nonnegative')
    tolerance = max(1e-6, min(1e-3, 8 * torch.finfo(probabilities.dtype).eps))
    if not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]),
                          atol=tolerance, rtol=0):
        raise ValueError('Each agent distribution must sum to one')
    return probabilities.shape


def _positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(label + ' must be a positive integer')
    return value


def _query(callback: Callable, actions, probabilities, query_chunk_size):
    """Query operational costs without differentiating or silently clipping them."""
    parts = []
    calls = 0
    for start in range(0, actions.shape[1], query_chunk_size):
        block = actions[:, start:start + query_chunk_size]
        with torch.no_grad():
            result = callback(block)
        if isinstance(result, torch.Tensor) and result.requires_grad:
            raise ValueError('Operational checker costs must not require gradients')
        result = torch.as_tensor(result, device=probabilities.device,
                                 dtype=probabilities.dtype)
        if result.shape != block.shape[:2]:
            raise ValueError('Checker must return normalized costs [B,Q]')
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Checker returned nonfinite costs')
        if bool((result < 0).any()) or bool((result > 1).any()):
            raise ValueError('Checker costs must already be normalized to [0,1]')
        parts.append(result.detach())
        calls += 1
    return torch.cat(parts, dim=1), calls


def _empty(probabilities, estimator, samples, baseline):
    value = probabilities.sum((1, 2)) * 0
    return RiskEstimate(value, value.detach(), 0, 0, estimator, samples, baseline)


def exact_product_risk(probabilities, cost_callback: Callable, *,
                       max_joint_actions: int = 65536,
                       query_chunk_size: int = 1024) -> RiskEstimate:
    """Exact differentiable J for a bounded product-policy action space.

    The N=4,A=5 reference has 625 joint actions.  The explicit size bound
    prevents accidentally treating exponential enumeration as a scalable path.
    The same function supports smaller diagnostic distributions.
    """
    batch, agents, actions = _probabilities(probabilities)
    _positive_integer(max_joint_actions, 'max_joint_actions')
    _positive_integer(query_chunk_size, 'query_chunk_size')
    count = actions ** agents
    if count > max_joint_actions:
        raise ValueError('Exact enumeration exceeds max_joint_actions; use sampling')
    if batch == 0:
        return _empty(probabilities, 'exact', count, 'none')
    flat = torch.arange(count, device=probabilities.device, dtype=torch.long)
    tuples = torch.stack([(flat // actions ** (agents - i - 1)) % actions
                          for i in range(agents)], dim=-1)
    proposals = tuples.unsqueeze(0).expand(batch, -1, -1)
    costs, calls = _query(cost_callback, proposals, probabilities, query_chunk_size)
    product = probabilities.new_ones((batch, count))
    for i in range(agents):
        product = product * probabilities[:, i, tuples[:, i]]
    value = (product * costs).sum(-1)
    return RiskEstimate(value, value.detach(), batch * count, calls,
                        'exact', count, 'none')


def sampled_product_risk(probabilities, cost_callback: Callable, *, samples: int,
                         generator=None, estimator: str = 'conditional',
                         baseline: str = 'loo',
                         query_chunk_size: int = 1024) -> RiskEstimate:
    """Unbiased fixed-state gradient estimates under the current product policy.

    ``score`` requests B*K costs and uses joint score gradients.  Its optional
    leave-one-out baseline is independent of the action of the sample it
    centers.  An ordinary same-batch mean baseline is deliberately unsupported:
    it shrinks the expected gradient by (K-1)/K.

    ``conditional`` samples other agents' actions, then enumerates each
    agent's A alternatives. It requests B*K*N*A costs, with O(B*K*A*N) proposal
    memory by querying one agent at a time. The sum includes shared-parameter
    contributions through every agent. No factorization of the cost predicate
    is required, but the proposal distribution itself MUST factorize.

    The baseline argument applies only to ``score``. Conditional enumeration
    does not claim a guaranteed variance reduction for the full summed gradient;
    cross-agent covariance and checker costs must be evaluated empirically.
    """
    batch, agents, actions = _probabilities(probabilities)
    _positive_integer(samples, 'samples')
    _positive_integer(query_chunk_size, 'query_chunk_size')
    if estimator not in ('score', 'conditional'):
        raise ValueError("estimator must be 'score' or 'conditional'")
    if baseline not in ('none', 'loo'):
        raise ValueError("baseline must be 'none' or action-independent 'loo'")
    if estimator == 'score' and baseline == 'loo' and samples < 2:
        raise ValueError('Leave-one-out baseline requires samples >= 2')
    used_baseline = baseline if estimator == 'score' else 'none'
    if batch == 0:
        return _empty(probabilities, estimator, samples, used_baseline)
    with torch.no_grad():
        draws = torch.multinomial(probabilities.detach().reshape(batch * agents, actions),
                                  samples, replacement=True, generator=generator)
        draws = draws.reshape(batch, agents, samples).permute(0, 2, 1).contiguous()
    calls = 0
    if estimator == 'score':
        costs, calls = _query(cost_callback, draws, probabilities, query_chunk_size)
        chosen = probabilities.unsqueeze(1).expand(-1, samples, -1, -1)
        chosen = chosen.gather(-1, draws.unsqueeze(-1)).squeeze(-1)
        if bool((chosen <= 0).any()):
            raise RuntimeError('Categorical sampling selected a zero-probability action')
        log_joint = chosen.log().sum(-1)
        if baseline == 'loo':
            center = (costs.sum(-1, keepdim=True) - costs) / (samples - 1)
        else:
            center = torch.zeros_like(costs)
        surrogate = ((costs - center) * log_joint).mean(-1)
        value = costs.mean(-1)
        queries = batch * samples
    else:
        surrogate = probabilities.new_zeros(batch)
        value = None
        alternatives = torch.arange(actions, device=probabilities.device)
        for i in range(agents):
            proposals = draws.unsqueeze(2).expand(-1, -1, actions, -1).clone()
            proposals[:, :, :, i] = alternatives.view(1, 1, -1)
            costs, count_calls = _query(cost_callback,
                                        proposals.reshape(batch, samples * actions, agents),
                                        probabilities, query_chunk_size)
            calls += count_calls
            costs = costs.reshape(batch, samples, actions)
            surrogate = surrogate + (probabilities[:, i, :].unsqueeze(1) * costs).sum(-1).mean(-1)
            if i == 0:
                # Every queried alternative set contains the original joint
                # draw. Reuse it, rather than asking for K additional costs.
                value = costs.gather(-1, draws[:, :, 0].unsqueeze(-1)).squeeze(-1).mean(-1)
        queries = batch * samples * agents * actions
    # A gradient surrogate need not have the numerical value of J.  Correct
    # its value without changing its gradient, so logging means what it says.
    loss = surrogate - surrogate.detach() + value
    return RiskEstimate(loss, value.detach(), queries, calls, estimator,
                        samples, used_baseline)


def estimate_product_risk(probabilities, cost_callback: Callable, *,
                          estimator: str = 'conditional', samples: int = 16,
                          generator=None, baseline: str = 'loo',
                          max_joint_actions: int = 65536,
                          query_chunk_size: int = 1024) -> RiskEstimate:
    """Unified entry point used by the complete-revision trainer."""
    if estimator == 'exact':
        return exact_product_risk(probabilities, cost_callback,
                                  max_joint_actions=max_joint_actions,
                                  query_chunk_size=query_chunk_size)
    return sampled_product_risk(probabilities, cost_callback, samples=samples,
                                generator=generator, estimator=estimator,
                                baseline=baseline, query_chunk_size=query_chunk_size)


def estimator_loss(probabilities, cost_callback: Callable, **kwargs) -> RiskEstimate:
    """Compatibility alias; call ``result.loss.mean()`` for scalar training."""
    return estimate_product_risk(probabilities, cost_callback, **kwargs)
