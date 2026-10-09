#!/usr/bin/env python3
"""Integrity checks for the post-review analysis, not new experimental trials."""
import csv,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reanalyze_nominal import ROOT,EXPECTED,holm
checks=[]
def ok(name,test):
    if not test:raise AssertionError(name)
    checks.append(name)
counts=json.loads((ROOT/'data/reanalysis/nominal_counts.json').read_text())
ok('nominal_integer_digest',hashlib.sha256(json.dumps(counts,sort_keys=True,separators=(',',':')).encode()).hexdigest()==EXPECTED)
sens=list(csv.DictReader((ROOT/'data/reanalysis/nominal_sensitivity.csv').open()))
ok('complete_40_family',len(sens)==40)
original={(r['domain'],r['layout'],r['baseline'],'compliance' if r['metric']=='proposed_compliance' else r['metric']):r for r in csv.DictReader((ROOT/'data/followup/primary_paired_effects.csv').open())}
maxdiff=0.
for r in sens:
    old=original[(r['domain'],r['layout'],r['baseline'],r['metric'])]
    for newcol,oldcol in [('difference','difference'),('bootstrap_ci_low','ci_low'),('bootstrap_ci_high','ci_high')]:maxdiff=max(maxdiff,abs(float(r[newcol])-float(old[oldcol])))
ok('original_presentation_bootstrap_precision',maxdiff<5e-10)
ok('holm_known_values',np.allclose(holm([.01,.03,.04]),[.03,.06,.06]))
ok('holm_permutation_equivariance',np.allclose(holm([.04,.01,.03]),[.06,.03,.06]))
ok('no_nominal_holm_t_below_005',all(float(r['paired_t_p_holm40'])>=.05 for r in sens))
ok('no_nominal_holm_flip_below_005',all(float(r['exact_mean_signflip_p_holm40'])>=.05 for r in sens))
for base in ('penalty','consensus_norm','random_norm','ppo_lagrangian'):
    r=next(r for r in sens if (r['domain'],r['layout'],r['baseline'],r['metric'])==('rware','tiny',base,'throughput'))
    a=np.array(counts['rware/tiny/pci_norm'])[:,1];b=np.array(counts['rware/tiny/'+base])[:,1]
    delta=(a-b)*1000/6144
    test=stats.permutation_test((delta,),np.mean,permutation_type='samples',alternative='two-sided',n_resamples=np.inf)
    ok('scipy_exact_flip_'+base,np.isclose(test.pvalue,float(r['exact_mean_signflip_p'])))
inter=list(csv.DictReader((ROOT/'data/reanalysis/intervention_units.csv').open()))
ok('all_12_intervention_cases',len(inter)==12)
ok('separate_valid_denominators',all(int(r['changed_joint_steps'])<=int(r['replaced_agent_actions'])<=int(r['agent_decisions']) and int(r['changed_joint_steps'])<=int(r['joint_steps']) and int(r['agent_decisions'])==int(r['agents'])*int(r['joint_steps']) for r in inter))
r8=[r for r in inter if r['agents']=='8']
ok('n8_changed_steps_74_88',min(int(r['changed_joint_steps']) for r in r8)==74 and max(int(r['changed_joint_steps']) for r in r8)==88)
report={'status':'PASS','checks':checks,'count':len(checks),'scope':'post-review calculation/identity checks, not independent training or cryptographic validation','max_original_presentation_difference':maxdiff}
(ROOT/'checks/SENSITIVITY_CHECK_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
