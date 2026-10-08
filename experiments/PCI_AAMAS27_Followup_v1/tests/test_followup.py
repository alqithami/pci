import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from pci_followup.model import Policy,energy,Q_ARRAY,centered_maps,dual_update,METHODS
from pci_followup.envs import RwareInstitution,ExternalConfig,make_env
from pci_followup.suite import plan
from pci_followup.train import train
from pci_followup.evaluate import episode
from pci_bench.environment import Warehouse,EnvConfig

@pytest.mark.parametrize('mode',METHODS)
def test_matched_policy_updates(mode):
    torch.set_num_threads(1);torch.manual_seed(991)
    e=Warehouse(EnvConfig(),42);f=torch.tensor(e.features()[None]);m=Policy(mode)
    dist,v,cv,p=m(f);reg=energy(p,f,mode,m.random_action_maps)
    assert dist.probs.shape==(1,4,5) and v.shape==cv.shape==(1,4)
    (dist.log_prob(torch.ones((1,4),dtype=torch.long)).mean()+cv.square().mean()+reg.mean()).backward()
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in h.parameters()) for h in m.heads)

def test_parameter_counts_matched():
    assert len({sum(p.numel() for p in Policy(m).parameters()) for m in METHODS})==1

@pytest.mark.parametrize('mode',('consensus_norm','pci_norm','random_norm'))
def test_identical_contexts_zero(mode):
    f=torch.randn(4,2,5,23);q=torch.randn(4,2,1,5).softmax(-1).expand(-1,-1,3,-1)
    assert torch.max(torch.abs(energy(q,f,mode)))<1e-12

def test_random_rotation_preserves_constant_and_spectrum():
    for Q in Q_ARRAY:
        np.testing.assert_allclose(Q.T@Q,np.eye(5),atol=1e-12)
        np.testing.assert_allclose(Q@np.ones(5),np.ones(5),atol=1e-12)
        F=np.random.default_rng(10).normal(size=(5,3));F-=F.mean(0)
        np.testing.assert_allclose(np.linalg.svd(Q@F,compute_uv=False),np.linalg.svd(F,compute_uv=False),atol=1e-12)
        assert not np.allclose(Q@F,F)

def test_scale_invariance():
    torch.manual_seed(1);f=torch.randn(2,4,5,23);p=torch.randn(2,4,3,5).softmax(-1)
    torch.testing.assert_close(energy(p,f,'pci_norm'),energy(p,f*7,'pci_norm'))
    torch.testing.assert_close(energy(p,f,'random_norm'),energy(p,f*7,'random_norm'))

def test_zero_maps_safe():
    p=torch.randn(1,4,3,5).softmax(-1)
    assert torch.equal(energy(p,torch.zeros(1,4,5,23),'pci_norm'),torch.zeros(1,4))

@pytest.mark.parametrize('cost,expected',[(.2,.34),(0.,.24)])
def test_dual_direction(cost,expected):
    assert dual_update(.25,cost,.02,.5,10)==pytest.approx(expected)

def test_dual_projected_and_finite():
    assert dual_update(0.,0.,1.,.5,10)==0
    assert dual_update(10.,2.,0.,.5,10)==10
    with pytest.raises(ValueError):dual_update(1,np.nan,.02,.5,10)

def rw(layout='tiny'):
    return RwareInstitution(ExternalConfig(layout=layout),413)

@pytest.mark.parametrize('layout',('tiny','small'))
def test_rware_native_parity_no_admission(layout):
    e=rw(layout);native=copy.deepcopy(e.native);rng=np.random.default_rng(33)
    for _ in range(16):
        actions=rng.integers(0,5,4)
        original=native.step(actions.tolist())
        _,_,_,_,m=e.transition(actions,enforce=False)
        assert [(a.x,a.y,a.dir.value) for a in e.native.agents]==[(a.x,a.y,a.dir.value) for a in native.agents]
        assert [s.id for s in e.native.request_queue]==[s.id for s in native.request_queue]
        assert e.last['native_rewards']==original[1]
        assert m['delivered']==sum(original[1])

def test_rware_actual_delivery_event():
    e=rw();s=e.native.request_queue[0];a=e.native.agents[0];gx,gy=e.native.goals[0]
    # Deliberately constructed native positive fixture, not a reported episode.
    for i,agent in enumerate(e.native.agents):agent.x,agent.y=i*3,0
    a.x,a.y=gx,gy;a.carrying_shelf=s;s.x,s.y=gx,gy;e.native._recalc_grid()
    _,_,_,_,m=e.transition(np.zeros(4,dtype=int))
    assert m['delivered']>=1 and e.delivered[0]>=1
    assert sum(e.last['native_rewards'])==m['delivered']

def setup_entry(e,agents=1):
    from rware.warehouse import Direction
    for i,a in enumerate(e.native.agents):a.x,a.y=i*3,0;a.dir=Direction.DOWN
    for i in range(agents):e.native.agents[i].x,e.native.agents[i].y=[0,1,3][i],e.native.grid_size[0]-4
    e.native._recalc_grid()

def test_rware_quota_rejection_and_cost():
    e=rw();setup_entry(e);e.counts[0,0]=3;actions=np.array([1,0,0,0])
    old=e.pos.copy();_,_,cost,_,m=e.transition(actions,enforce=True)
    assert m['proposed_compliant']==0 and m['executed_compliant']==1 and cost[0]>0
    np.testing.assert_array_equal(e.pos[0],old[0]);assert e.last['admitted_actions'][0]==0

def test_rware_entry_capacity_rejection():
    e=rw();setup_entry(e,3);_,_,_,_,m=e.transition(np.array([1,1,1,0]),enforce=True)
    assert m['capacity_excess']==1 and m['executed_compliant']==1
    assert e.counts[:,0].sum()==2

@pytest.mark.parametrize('layout',('tiny','small'))
def test_rware_reset_and_replay(layout):
    e=rw(layout);rng=np.random.default_rng(1)
    for _ in range(12):e.transition(rng.integers(0,5,4))
    s=e.state_dict();actions=np.array([0,1,2,3]);x=e.transition(actions)
    e.load_state_dict(s);y=e.transition(actions)
    np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1]);assert x[4]==y[4]
    e.t=63;e.native._cur_steps=63;e.counts[:]=2;e.transition(np.zeros(4,dtype=int))
    assert e.t==64 and not e.counts.any()

def test_full_plan_no_original_seeds_and_fixed_size():
    p=plan(False);assert len(p['runs'])==125
    assert sum(c['total_steps'] for c in p['runs'])==32768000
    assert {c['seed'] for c in p['runs']}=={101,102,103,104,105}
    assert {c['method'] for c in p['runs']}==set(METHODS)
    assert len({c['run_id'] for c in p['runs']})==125
    assert sum(c['domain']=='rware' for c in p['runs'])==50

@pytest.mark.parametrize('domain',('custom','rware'))
@pytest.mark.parametrize('method',('pci_norm','random_norm','ppo_lagrangian'))
def test_real_small_training_checkpoint(domain,method,tmp_path):
    c=next(x for x in plan(True)['runs'] if x['domain']==domain and x['method']==method)
    c=copy.deepcopy(c);c.update(total_steps=128,n_envs=2,rollout_steps=32,minibatch_groups=32)
    d=train(c,tmp_path);assert d['env_steps']==128 and d['updates']==2 and d['parameter_change_l2']>0
    assert train(c,tmp_path)==d
    m=Policy(method);m.load_state_dict(torch.load(tmp_path/'checkpoint.pt',weights_only=False)['model']);m.eval()
    r=episode(c,m,413,True);assert r['executed_compliance']==1 and r['actual_vertex_collisions']==0
