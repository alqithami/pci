import numpy as np
import torch
import pytest
from pci_bench.environment import EnvConfig,Warehouse,D
from pci_bench.model import ContextPolicy,compatibility_energy,gae,METHODS

def test_matched_parameter_counts():
    counts=[sum(p.numel() for p in ContextPolicy(m,16).parameters()) for m in METHODS]
    assert len(set(counts))==1

def test_consistency_zero_for_compatible_heads():
    f=torch.tensor(Warehouse(EnvConfig(),0).features()[None])
    p=torch.ones((1,4,3,5))/5
    assert torch.equal(compatibility_energy(p,f),torch.zeros(1,4))

def test_inconsistent_agents_cannot_cancel():
    f=torch.tensor(Warehouse(EnvConfig(n_agents=2),2).features()[None])
    p=torch.ones((1,2,3,5))/5
    p[0,0,0,:]=torch.tensor([0.,1.,0.,0.,0.]);p[0,1,0,:]=torch.tensor([0.,0.,1.,0.,0.])
    energy=compatibility_energy(p,f)
    assert torch.all(energy>0)

def test_differentiable_consistency_heads():
    torch.manual_seed(42);torch.set_num_threads(1)
    f=torch.tensor(Warehouse(EnvConfig(),0).features()[None])
    model=ContextPolicy('pci',16)
    _,_,p=model(f);loss=compatibility_energy(p,f).mean();loss.backward()
    for head in model.heads:
        assert sum(float(x.grad.abs().sum()) for x in head.parameters() if x.grad is not None)>0

def test_each_head_controls_action_distribution():
    torch.manual_seed(13);torch.set_num_threads(1)
    f=torch.tensor(Warehouse(EnvConfig(),0).features()[None]);m=ContextPolicy('pci',16)
    loss=-m(f)[0].log_prob(torch.tensor([[1,2,3,4]])).mean();loss.backward()
    for head in m.heads:
        assert sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None)>0

def test_residual_gradient_finite_difference():
    f=torch.tensor(Warehouse(EnvConfig(n_agents=2),0).features()[None],dtype=torch.float64)
    z=torch.randn((1,2,3,5),dtype=torch.float64,requires_grad=True)
    value=compatibility_energy(z.softmax(-1),f).sum();grad=torch.autograd.grad(value,z)[0]
    eps=1e-6;index=(0,0,1,3)
    plus=z.detach().clone();minus=z.detach().clone();plus[index]+=eps;minus[index]-=eps
    numerical=(compatibility_energy(plus.softmax(-1),f).sum()-compatibility_energy(minus.softmax(-1),f).sum())/(2*eps)
    assert abs(float(numerical-grad[index]))<1e-7

def test_terminal_gae_never_bootstraps_next_episode():
    rewards=torch.tensor([[[1.]],[[2.]]]);values=torch.zeros_like(rewards)
    dones=torch.ones(2,1);last=torch.tensor([[100.]])
    a,r=gae(rewards,values,dones,last)
    assert torch.equal(a,rewards)

def test_nonterminal_gae_bootstraps():
    a,_=gae(torch.tensor([[[1.]]]),torch.tensor([[[2.]]]),torch.zeros(1,1),torch.tensor([[3.]]),gamma=0.5,lam=1)
    assert a.item()==0.5


def test_actual_heldout_episode_probability_normalization():
    from pci_bench.evaluate import one_episode
    torch.manual_seed(7);torch.set_num_threads(1)
    e=Warehouse(EnvConfig(horizon=32,n_agents=4),100000)
    m=ContextPolicy('pci',16)
    result=one_episode(e,model=m,seed=100000,enforce=True)
    assert result['episode_len']==32 and result['executed_compliance']==1


def test_typed_overlap_preserves_equivalence_not_full_consensus():
    f=Warehouse(EnvConfig(n_agents=2),0).features()[0]
    # Safety-efficiency edge has three effect coordinates for five actions.
    restriction=f[:,[3,4,2]].T.astype(float)
    constraint=np.vstack((restriction,np.ones(5)))
    _,_,vh=np.linalg.svd(constraint,full_matrices=True);h=vh[-1]
    h=h/max(abs(h))
    p=np.ones(5)/5+.1*h;q=np.ones(5)/5-.1*h
    assert min(p)>0 and min(q)>0 and np.isclose(p.sum(),1) and np.isclose(q.sum(),1)
    assert np.linalg.norm(restriction@(p-q))<1e-7
    assert np.linalg.norm(p-q)>.1

def test_zero_compatibility_is_not_operational_compliance():
    from pci_bench.environment import monitor_record
    e=Warehouse(EnvConfig(n_agents=2),19);e.counts[0,0]=4
    f=torch.tensor(e.features()[None]);p=torch.zeros((1,2,3,5));p[:,:,:,0]=1
    assert compatibility_energy(p,f).sum().item()==0
    assert not monitor_record(e.record_for_actions([0,0]))['compliant']
