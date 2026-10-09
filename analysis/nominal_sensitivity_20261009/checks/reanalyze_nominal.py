#!/usr/bin/env python3
"""Post-review sensitivity analysis of the complete 40-contrast nominal family.

Input totals reproduce the original 24 x 256-step episodes per training seed.
They were checked against all 125 original per-run evaluation files. This analysis
preserves the original pointwise bootstrap; it does not retroactively preregister
Holm adjustment or change the evaluation outcome/selection rule.
"""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='91c68f761cdc483beddf1622b36e70e91fca91314557126af0a25baf77531524'
BASELINES=('penalty','consensus_norm','random_norm','ppo_lagrangian')

def holm(values):
    values=np.asarray(values,float);order=np.argsort(values,kind='stable');out=np.empty(len(values));running=0.
    for rank,i in enumerate(order):
        running=max(running,(len(values)-rank)*values[i]);out[i]=min(1.,running)
    return out

def run(out:Path):
    g=json.loads((ROOT/'data/reanalysis/nominal_counts.json').read_text())
    raw=json.dumps(g,sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('Input identity mismatch')
    out.mkdir(parents=True,exist_ok=True);rows=[];paired=[];conditions=[]
    for key,v in sorted(g.items()):
        assert [r[0] for r in v]==list(range(101,106))
        assert all(isinstance(x,int) for r in v for x in r)
        assert all(0<=r[2]<=6144 for r in v)
        domain,layout,method=key.split('/')
        for seed,t,c in v:
            conditions.append(dict(domain=domain,layout=layout,method=method,seed=seed,
                deliveries=t,compliant_steps=c,throughput=1000*t/6144,compliance=c/6144))
    configs=sorted(set('/'.join(k.split('/')[:2]) for k in g))
    signs=np.asarray(list(itertools.product((-1,1),repeat=5)),dtype=np.int64)
    bootstrap_idx=np.random.default_rng(92170).integers(0,5,(5000,5))
    for config in configs:
        a=np.asarray(g[config+'/pci_norm'],dtype=np.int64)
        for base in BASELINES:
            b=np.asarray(g[config+'/'+base],dtype=np.int64)
            assert np.array_equal(a[:,0],b[:,0])
            for j,metric,scale in [(1,'throughput',1000/6144),(2,'compliance',1/6144)]:
                counts=a[:,j]-b[:,j];delta=counts*scale;mean=float(delta.mean())
                # Integer arithmetic prevents floating-tolerance changes to exact p-values.
                flip_p=float(np.mean(np.abs(signs@counts)>=abs(counts.sum())))
                if np.ptp(delta)==0:
                    tlo=thi=mean;tp=1. if mean==0 else 0.
                else:
                    result=stats.ttest_1samp(delta,0);ci=result.confidence_interval(.95)
                    tlo,thi,tp=float(ci.low),float(ci.high),float(result.pvalue)
                boot=delta[bootstrap_idx].mean(1);blo,bhi=map(float,np.quantile(boot,[.025,.975]))
                nz=counts[counts!=0];sign_p=float(stats.binomtest(int(np.sum(nz>0)),len(nz),.5).pvalue) if len(nz) else 1.
                domain,layout=config.split('/')
                row=dict(domain=domain,layout=layout,baseline=base,metric=metric,n_pairs=5,
                    difference=mean,bootstrap_ci_low=blo,bootstrap_ci_high=bhi,
                    paired_t_ci_low=tlo,paired_t_ci_high=thi,paired_t_p=tp,
                    exact_mean_signflip_p=flip_p,exact_sign_p=sign_p,
                    positive_pairs=int(np.sum(counts>0)),zero_pairs=int(np.sum(counts==0)))
                rows.append(row)
                for i,seed in enumerate(a[:,0]):paired.append(dict(domain=domain,layout=layout,baseline=base,metric=metric,seed=int(seed),difference=float(delta[i])))
    assert len(rows)==40 and len(paired)==200 and len(conditions)==125
    for name in ['paired_t_p','exact_mean_signflip_p']:
        adjusted=holm([r[name] for r in rows])
        for r,value in zip(rows,adjusted):r[name+'_holm40']=float(value)
    for name,data in [('nominal_sensitivity.csv',rows),('nominal_paired_seeds.csv',paired),('nominal_seed_means.csv',conditions)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    report={'status':'COMPLETE','analysis_type':'post-review sensitivity; not a new confirmatory experiment',
        'input_count_digest':EXPECTED,'input_policies':125,'nominal_episodes_represented':3000,
        'contrasts':40,'paired_seed_values':200,'bootstrap_draws':5000,'bootstrap_seed':92170,
        'family':'five configurations x four comparators x two nominal unshielded endpoints',
        'holm_t_below_0_05':sum(r['paired_t_p_holm40']<.05 for r in rows),
        'holm_signflip_below_0_05':sum(r['exact_mean_signflip_p_holm40']<.05 for r in rows),
        'assumptions':{'paired_t':'independent normal seed differences for exact finite-sample coverage',
            'exact_signflip':'sign-symmetry / within-pair label-exchangeability null; common seeds alone do not establish this',
            'bootstrap':'original pointwise percentile interval from five empirical seed values; finite support and no multiplicity guarantee'},
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('nominal_*.csv'))}}
    (out/'STATISTICAL_SENSITIVITY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    for r in rows:
        if r['domain']=='rware' and r['layout']=='tiny' and r['metric']=='throughput':print(r)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'data/reanalysis');run(p.parse_args().out)
