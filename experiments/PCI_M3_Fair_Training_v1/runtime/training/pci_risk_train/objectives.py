"""Bound-normalized original energies and exact fixed-state institutional risk.

The direct auxiliary is an existing expected-cost objective on a finite product
policy. It does not differentiate native state transitions or state occupancy.
The costs are RAW Hamming counts, normalized by four exactly once here.
"""
from __future__ import annotations
import torch
from .model import compatibility_energy

METHODS = ('task', 'fixed', 'adaptive', 'semantic', 'consensus', 'directrisk')
MODEL_METHOD = {'task':'ppo_task', 'fixed':'ppo_penalty',
                'adaptive':'ppo_penalty', 'semantic':'pci',
                'consensus':'consensus', 'directrisk':'ppo_penalty'}
SEMANTIC_BOUND_SCALE = 6. / 13.
CONSENSUS_BOUND_SCALE = 5. / 2.
# Lexicographic product order, identical to the native 625-row table.
ACTION_TUPLES = tuple((a,b,c,d) for a in range(5) for b in range(5)
                      for c in range(5) for d in range(5))


def institutional_risk(probabilities, raw_costs):
    """Per-state expected RAW/4 cost, retaining all states including zero rows."""
    if probabilities.ndim != 3 or probabilities.shape[1:] != (4,5):
        raise ValueError('Expected policy probabilities [B,4,5]')
    if raw_costs.shape != (len(probabilities),625):
        raise ValueError('Expected RAW integer cost table [B,625]')
    if raw_costs.requires_grad:
        raise ValueError('Operational cost tables must be fixed, not trainable')
    if not torch.isfinite(raw_costs).all() or bool((raw_costs < 0).any()) or bool((raw_costs > 4).any()):
        raise ValueError('RAW institutional Hamming costs must lie in 0..4')
    if raw_costs.is_floating_point() and not torch.equal(raw_costs, raw_costs.round()):
        raise ValueError('RAW institutional Hamming costs must be integers')
    actions = torch.tensor(ACTION_TUPLES, dtype=torch.long, device=probabilities.device)
    product = torch.ones((len(probabilities),625), dtype=probabilities.dtype,
                         device=probabilities.device)
    for i in range(4):
        product = product * probabilities[:,i,actions[:,i]]
    return (product * raw_costs.to(dtype=probabilities.dtype,
                                  device=probabilities.device) / 4.).sum(-1)


def auxiliary(method, dist, head_probabilities, features, raw_costs):
    """Scalar batch mean; all 1024/128 joint-state groups remain intact."""
    if features.shape[0] == 0:
        return dist.probs.sum()*0.
    if method == 'semantic':
        return SEMANTIC_BOUND_SCALE * compatibility_energy(
            head_probabilities,features,mode='pci').mean()
    if method == 'consensus':
        return CONSENSUS_BOUND_SCALE * compatibility_energy(
            head_probabilities,features,mode='consensus').mean()
    if method == 'directrisk':
        return institutional_risk(dist.probs,raw_costs).mean()
    if method not in METHODS:
        raise ValueError('Unknown method')
    return features.new_zeros(())
