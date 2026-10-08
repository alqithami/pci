import copy
import numpy as np
import pytest
from pci_bench.environment import EnvConfig,Warehouse,geometry,monitor_record
from pci_bench.crypto_check import samples

@pytest.mark.parametrize('layout',['open','twodoors','staggered'])
def test_maps_connected_and_reproducible(layout):
    a=Warehouse(EnvConfig(layout=layout),21);b=Warehouse(EnvConfig(layout=layout),21)
    assert np.array_equal(a.features(),b.features())
    for t in range(24):
        actions=np.array([(t+i)%5 for i in range(a.n)])
        aa=a.transition(actions,enforce=True);bb=b.transition(actions,enforce=True)
        assert np.array_equal(aa[0],bb[0]);assert aa[-1]==bb[-1]

@pytest.mark.parametrize('layout',['open','twodoors','staggered'])
@pytest.mark.parametrize('n',[2,4,8,12])
def test_shield_scalar_monitor_agreement(layout,n):
    e=Warehouse(EnvConfig(layout=layout,n_agents=n,horizon=128,rule_change=True),7)
    rng=np.random.default_rng(99)
    for _ in range(128):
        _,_,_,_,m=e.transition(rng.integers(0,5,n),enforce=True)
        assert m['executed_compliant']==1
        assert m['actual_vertex_collisions']==0
        assert monitor_record(e.last)['next_positions']==e.pos.tolist()

@pytest.mark.parametrize('name',['collision','swap','wall','capacity','quota'])
def test_negative_transition_fixtures(name):
    good,bad=samples()
    assert monitor_record(good)['compliant']
    assert not monitor_record(bad[name])['compliant']

def test_aggregate_capacity_is_not_pairwise_approximation():
    e=Warehouse(EnvConfig(n_agents=3),0)
    e.pos[:]=[[4,3],[6,3],[8,3]]
    r=e.record_for_actions([1,0,2]);r['capacities']=[2,2]
    # Each of the three pairs occupies 2 cells, but all three exceed cap 2.
    assert 'capacity:0' in monitor_record(r)['errors']

def test_pickup_and_delivery_are_real_events():
    e=Warehouse(EnvConfig(layout='open',n_agents=2),0)
    e.pos[:]=[[2,2],[1,1]];e.goals[0]=[2,2]
    _,_,_,_,m=e.transition(np.array([0,0]),enforce=True)
    assert m['pickups']==1 and m['delivered']==0 and e.carrying[0]
    # Start a legal state at the actual destination and execute stay.
    e.pos[0]=e.goals[0]
    _,_,_,_,m=e.transition(np.array([0,0]),enforce=True)
    assert m['delivered']==1 and not e.carrying[0] and e.delivered[0]==1

def test_rejected_certificate_does_not_execute():
    e=Warehouse(EnvConfig(n_agents=2),12)
    before=e.pos.copy();counts=e.counts.copy();time=e.t
    with pytest.raises(RuntimeError,match='no transition'):
        e.transition(np.array([0,0]),enforce=True,certifier=lambda r:False)
    assert np.array_equal(before,e.pos) and np.array_equal(counts,e.counts) and time==e.t

def test_certificate_is_checked_before_mutation():
    e=Warehouse(EnvConfig(n_agents=2),12)
    before=e.pos.copy()
    def accept(record):
        assert np.array_equal(before,e.pos)
        assert record['before']['positions']==before.tolist()
        return True
    e.transition(np.array([1,0]),enforce=True,certifier=accept)

def test_checkpoint_preserves_environment_stream():
    e=Warehouse(EnvConfig(),0)
    for _ in range(3):e.transition(np.array([0,1,2,3]),enforce=True)
    f=Warehouse(EnvConfig(),123);f.load_state_dict(e.state_dict())
    assert np.array_equal(e.features(),f.features())
    assert np.array_equal(e.reset(),f.reset())

def test_waiting_has_no_fake_deliveries():
    e=Warehouse(EnvConfig(n_agents=2),19)
    e.pos[:]=[[1,1],[11,11]]
    for _ in range(20):
        m=e.transition(np.array([0,0]))[-1]
        assert m['delivered']==0
