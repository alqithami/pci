"""Complete-data-only analysis. No favorable-result selection or pooled domains."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import read_csv_strict,atomic_json,atomic_csv,sha256

METRICS=('throughput_per_1000_steps','proposed_compliance','executed_compliance','cost_rate','interventions','max_wait_final','mean_semantic_energy')

def interval(v):
    x=np.asarray(v,dtype=float)
    if not np.isfinite(x).all() or len(x)<1:raise ValueError('Invalid estimates')
    if len(x)==1:return float(x[0]),None,None
    rng=np.random.default_rng(92170);draws=x[rng.integers(0,len(x),(5000,len(x)))].mean(1)
    return float(x.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))

def analyse(root):
    root=Path(root);p=json.loads((root/'plan.json').read_text());out=root/'analysis';out.mkdir(exist_ok=True)
    allrows=[];uprows=[]
    for c in p['runs']:
        d=root/'runs'/c['run_id'];t=json.loads((d/'TRAIN_COMPLETE.json').read_text());e=json.loads((d/'EVAL_COMPLETE.json').read_text())
        if t['env_steps']!=c['total_steps'] or t['checkpoint_sha256']!=sha256(d/'checkpoint.pt'):raise AssertionError('Training integrity failure')
        if e['csv_sha256']!=sha256(d/'evaluation.csv'):raise AssertionError('Evaluation integrity failure')
        rows=read_csv_strict(d/'evaluation.csv');expected=p['eval_episodes']*2*(1 if p['profile']=='smoke' else 6)
        if len(rows)!=expected:raise AssertionError('Evaluation coverage incomplete')
        unique=set()
        for r in rows:
            key=(r['scenario'],r['shield'],r['eval_episode'])
            if key in unique:raise AssertionError('Duplicate evaluation');
            unique.add(key)
            r['train_seed']=int(r['train_seed']);r['shield']=int(r['shield']);r['eval_episode']=int(r['eval_episode'])
            for m in METRICS:r[m]=float(r[m])
            if abs(r['throughput_per_1000_steps']-1000*float(r['deliveries'])/float(r['episode_len']))>1e-10:raise AssertionError('Throughput mismatch')
            if sum(json.loads(r['per_agent_deliveries']))!=float(r['deliveries']):raise AssertionError('Delivery ledger mismatch')
            if r['shield'] and r['executed_compliance']!=1:raise AssertionError('Admitted episode violated operational rules')
            allrows.append(r)
        ups=read_csv_strict(d/'updates.csv')
        if any(float(u['actor_change_l2'])<=0 for u in ups):raise AssertionError('Non-learning policy')
        uprows+=ups
    M=pd.DataFrame(allrows);M.to_csv(out/'evaluation_all.csv',index=False)
    U=pd.DataFrame(uprows);U.to_csv(out/'updates_all.csv',index=False)
    keys=['domain','layout','method','train_seed','scenario','shield']
    run=M.groupby(keys,dropna=False)[list(METRICS)].mean().reset_index();run.to_csv(out/'run_means.csv',index=False)
    rows=[]
    for label,g in run.groupby(['domain','layout','method','scenario','shield']):
        for metric in METRICS:
            mean,lo,hi=interval(g[metric]);rows.append(dict(zip(['domain','layout','method','scenario','shield'],label))|
                {'metric':metric,'mean':mean,'ci_low':lo,'ci_high':hi,'n_training_seeds':len(g)})
    atomic_csv(out/'condition_summary.csv',rows)
    paired=[]
    for label,g in run.groupby(['domain','layout','scenario','shield']):
        for other in ('penalty','consensus_norm','random_norm','ppo_lagrangian'):
            a=g[g.method=='pci_norm'].set_index('train_seed');b=g[g.method==other].set_index('train_seed')
            if set(a.index)!=set(b.index):raise AssertionError('Unpaired seeds')
            for metric in METRICS:
                mean,lo,hi=interval(a[metric].sort_index()-b[metric].sort_index())
                paired.append(dict(zip(['domain','layout','scenario','shield'],label))|{'baseline':other,'metric':metric,
                    'difference':mean,'ci_low':lo,'ci_high':hi,'n_pairs':len(a)})
    atomic_csv(out/'paired_effects.csv',paired)
    crypto=json.loads((root/'certified/CERTIFIED_EPISODES_COMPLETE.json').read_text())
    wanted=4 if p['profile']=='smoke' else p['protocol']['crypto_expected_transitions']
    if crypto['certified_transitions']!=wanted:raise AssertionError('Incomplete certified trajectories')
    measurements=[];ep=[]
    for d in sorted((root/'certified').glob('n*_s*_*')):
        if not d.is_dir():continue
        mark=json.loads((d/'COMPLETE.json').read_text())
        if mark['csv_sha256']!=sha256(d/'measurements.csv'):raise AssertionError('Changed certificate timings')
        measurements+=read_csv_strict(d/'measurements.csv');ep.append({'case':d.name,**json.loads((d/'episode.json').read_text())})
    C=pd.DataFrame(measurements)
    for k in ('n_agents','witness_and_prove_ms','verify_ms','wall_total_ms','paired'):C[k]=pd.to_numeric(C[k])
    C.to_csv(out/'certificate_measurements.csv',index=False);atomic_csv(out/'certified_episode_outcomes.csv',ep)
    if p['profile']=='full' and len(C)!=p['protocol']['crypto_expected_measurements']:raise AssertionError('Incomplete measurements')
    C.groupby(['n_agents','protocol'])[['witness_and_prove_ms','verify_ms','wall_total_ms']].median().to_csv(out/'certificate_medians.csv')
    # Thin-line, B&W plus one accent figures; each is a separate axis.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    names={'penalty':'Fixed penalty','consensus_norm':'Normalized consensus','pci_norm':'Normalized PCI','random_norm':'Spectrum-matched rotation','ppo_lagrangian':'Adaptive PPO-Lagrangian'}
    primary=run[(run.scenario=='nominal')&(run.shield==0)]
    for (domain,layout),g in primary.groupby(['domain','layout']):
        fig,ax=plt.subplots(figsize=(5.5,3.6))
        for i,(method,h) in enumerate(g.groupby('method')):
            x,xlo,xhi=interval(h.proposed_compliance*100);y,ylo,yhi=interval(h.throughput_per_1000_steps)
            color='#b22222' if method=='pci_norm' else str(.1+i*.12)
            ax.errorbar(x,y,xerr=None if xlo is None else [[x-xlo],[xhi-x]],yerr=None if ylo is None else [[y-ylo],[yhi-y]],
                fmt=['o','s','^','D','v'][i],mfc='white',color=color,ms=4,elinewidth=.8,capsize=2,label=names[method])
        ax.set_xlabel('Proposed compliance (%)');ax.set_ylabel('Deliveries per 1,000 joint steps');ax.set_title(f'{domain} / {layout}')
        ax.legend(frameon=False,fontsize=7,loc='best');fig.tight_layout();fig.savefig(out/f'tradeoff_{domain}_{layout}.pdf');plt.close(fig)
    report={'status':'COMPLETE','study':'post-audit follow-up; not pooled with original study','learned_runs':len(p['runs']),
        'evaluation_rows':len(M),'certified_transitions':crypto['certified_transitions'],'certificate_measurements':len(C),
        'claim':'Effectiveness must be evaluated from measured effects; no automatic superiority conclusion.'}
    atomic_json(out/'ANALYSIS_REPORT.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);analyse(p.parse_args().results)
