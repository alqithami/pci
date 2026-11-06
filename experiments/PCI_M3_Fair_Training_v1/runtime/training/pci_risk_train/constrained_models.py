"""Matched actor with separate reward and institutional-cost value critics.

All comparison arms instantiate and fit both critics. Only the constrained arm
uses the cost advantage in its policy objective. The actor's context masks and
zero pooled-information slot are retained for the original warehouse features.
Other environments must provide explicit view masks; none are guessed.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical
from .environment import D, VIEW_INDICES
from .model import mlp


class ConstrainedContextPolicy(nn.Module):
    def __init__(self, hidden=64, *, feature_dim=D, action_count=5, view_indices=None):
        super().__init__()
        if feature_dim < 1 or action_count < 2 or hidden < 1:
            raise ValueError('Invalid policy dimensions')
        if view_indices is None:
            if feature_dim != D or action_count != 5:
                raise ValueError('Non-warehouse observations require explicit context view masks')
            view_indices = VIEW_INDICES
        if len(view_indices) != 3:
            raise ValueError('Exactly three declared context views required')
        masks = torch.zeros(3, feature_dim)
        for i, indices in enumerate(view_indices):
            if not indices or len(set(indices)) != len(indices) or any(not 0 <= j < feature_dim for j in indices):
                raise ValueError('Invalid context view indices')
            masks[i, list(indices)] = 1
        self.register_buffer('masks', masks)
        self.feature_dim = feature_dim; self.action_count = action_count; self.hidden = hidden
        self.view_indices = tuple(tuple(x) for x in view_indices)
        # Construct actor first, then reward critic, preserving original RNG order.
        self.heads = nn.ModuleList([mlp(2 * feature_dim, 1, hidden) for _ in range(3)])
        self.reward_critic = mlp(2 * action_count * feature_dim, 1, hidden)
        self.cost_critic = mlp(2 * action_count * feature_dim, 1, hidden)

    def forward(self, features):
        if features.ndim != 4 or features.shape[-2:] != (self.action_count, self.feature_dim):
            raise ValueError('Features must be [B,N,A,D] with the declared dimensions')
        pooled = features.mean(dim=1, keepdim=True).expand_as(features)
        zero_global = torch.zeros_like(pooled)
        logits = torch.stack([
            head(torch.cat((features * self.masks[i], zero_global), dim=-1)).squeeze(-1)
            for i, head in enumerate(self.heads)
        ], dim=-2)
        dist = Categorical(logits=logits.mean(dim=-2))
        critic_input = torch.cat((features.flatten(start_dim=-2),
                                  pooled.flatten(start_dim=-2)), dim=-1)
        return (dist, self.reward_critic(critic_input).squeeze(-1),
                self.cost_critic(critic_input).squeeze(-1), logits.softmax(dim=-1))

    def actor_parameters(self):
        return self.heads.parameters()

    def specification(self):
        return {'model': 'ConstrainedContextPolicy-v1', 'hidden': self.hidden,
                'feature_dim': self.feature_dim, 'action_count': self.action_count,
                'context_view_indices': [list(x) for x in self.view_indices],
                'pooled_actor_information': False,
                'actor_parameters': sum(p.numel() for p in self.heads.parameters()),
                'reward_critic_parameters': sum(p.numel() for p in self.reward_critic.parameters()),
                'cost_critic_parameters': sum(p.numel() for p in self.cost_critic.parameters())}
