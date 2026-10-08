"""Parameter-matched context policies and a differentiable overlap operator.

The consistency energy is a graph/cellular-sheaf compatibility energy on the
1-skeleton. It is NOT a first-cohomology obstruction or a hard safety certificate.
Every local policy head affects the executed policy via averaged logits.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .environment import D, VIEW_INDICES, OVERLAPS

METHODS=('ppo_task','ppo_penalty','consensus','pci','pci_shuffled','global_ppo')


def mlp(n_in,n_out,hidden):
    net=nn.Sequential(nn.Linear(n_in,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,n_out))
    for m in net.modules():
        if isinstance(m,nn.Linear):nn.init.orthogonal_(m.weight,np.sqrt(2));nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight,0.01 if n_out==1 else 1.0)
    return net

class ContextPolicy(nn.Module):
    def __init__(self,method='pci',hidden=64):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method
        masks=torch.zeros(3,D)
        for i,idx in enumerate(VIEW_INDICES):masks[i,list(idx)]=1
        self.register_buffer('masks',masks)
        # The pooled feature slot is zero for decentralized actors; architecture is matched.
        self.heads=nn.ModuleList([mlp(2*D,1,hidden) for _ in range(3)])
        self.critic=mlp(10*D,1,hidden)

    def forward(self,features):
        # features: [batch, agent, action, feature]
        pooled=features.mean(dim=1,keepdim=True).expand_as(features)
        global_slot=pooled if self.method=='global_ppo' else torch.zeros_like(pooled)
        logits=torch.stack([head(torch.cat((features*self.masks[i],global_slot),dim=-1)).squeeze(-1)
                            for i,head in enumerate(self.heads)],dim=-2)
        policy=Categorical(logits=logits.mean(dim=-2))
        own=features.flatten(start_dim=-2)
        global_obs=pooled.flatten(start_dim=-2)
        value=self.critic(torch.cat((own,global_obs),dim=-1)).squeeze(-1)
        return policy,value,logits.softmax(dim=-1)


def compatibility_energy(probs, features, mode='pci'):
    """Per [batch, agent] energy. No NumPy detach, random signals, or trainable maps."""
    if mode=='consensus':
        return sum((probs[...,u,:]-probs[...,v,:]).square().mean(-1) for u,v,_ in OVERLAPS)/len(OVERLAPS)
    result=torch.zeros_like(probs[...,0,0])
    for u,v,idx in OVERLAPS:
        effect=features[...,list(idx)]
        left=torch.einsum('...a,...ad->...d',probs[...,u,:],effect)
        right_p=probs[...,v,:]
        if mode=='pci_shuffled':right_p=right_p[..., [2,4,1,0,3]]
        right=torch.einsum('...a,...ad->...d',right_p,effect)
        result=result+(left-right).square().mean(-1)
    return result/len(OVERLAPS)


def gae(rewards,values,dones,last_value,gamma=0.99,lam=0.95):
    """Finite-horizon episodes. dones[T,E] genuinely terminate the task horizon."""
    adv=torch.zeros_like(rewards); carry=torch.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        nxt=last_value if t==rewards.shape[0]-1 else values[t+1]
        live=1-dones[t].unsqueeze(-1)
        delta=rewards[t]+gamma*nxt*live-values[t]
        carry=delta+gamma*lam*live*carry
        adv[t]=carry
    return adv,adv+values
