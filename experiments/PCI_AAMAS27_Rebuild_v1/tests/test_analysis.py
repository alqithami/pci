import copy,json
from pathlib import Path
import numpy as np
import pandas as pd
from pci_bench.analysis import bootstrap_mean,nondominated,summarize,MEASURES,write_tex
from pci_bench.suite import create_plan

def test_single_seed_has_no_invented_interval():
    assert bootstrap_mean([3.])==(3.,None,None)

def test_bootstrap_seed_means_not_row_counts():
    rows=[]
    # One seed has 100 repeated evaluations, one has 1: equal seed weights, not 100:1.
    for seed,repeats,value in [(0,100,0.),(1,1,1.)]:
        for ep in range(repeats):
            rows.append({'method':'pci','consistency_coef':.2,'layout':'open','scenario':'nominal','shield':0,'n_agents':4,
                         'run_id':str(seed),'train_seed':seed,'eval_episode':ep,**{k:value for k in MEASURES}})
    means,table=summarize(pd.DataFrame(rows));r=table[table.measure=='proposed_compliance'].iloc[0]
    assert r['mean']==.5 and r.n_units==2 and r.uncertainty_unit=='training_seed'

def test_nondominated_is_empirical():
    assert nondominated([[1,1],[0,0],[2,0],[0,2]]).tolist()==[True,False,True,True]

def test_plan_counts_no_phantom_runs():
    p=create_plan('deadline')
    assert len(p['runs'])==110 and len(p['controllers'])==12
    assert len({r['run_id'] for r in p['runs']})==110
    assert p['total_joint_training_steps']==110*262144
    assert p['crypto_anchor'] in {r['run_id'] for r in p['runs']}

def test_latex_rows_have_two_backslashes(tmp_path):
    data=[]
    for metric in ['throughput_per_1000_steps','interventions']:
        data.append({'scenario':'nominal','shield':1,'consistency_coef':.2,'method':'pci','layout':'open',
                     'measure':metric,'mean':1.,'ci_low':None,'ci_high':None})
    write_tex(pd.DataFrame(data),tmp_path)
    line=[x for x in (tmp_path/'nominal_table.tex').read_text().splitlines() if x.startswith('open')][0]
    assert line.endswith('\\\\')
