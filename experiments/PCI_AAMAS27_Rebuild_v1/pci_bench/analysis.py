"""Strict run-level analysis. One figure per scientific question, never fake bands.

Independent training seeds, not correlated transitions, define uncertainty for RL.
Heuristics are a separate repeated-evaluation population. Proof latency is descriptive.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from .io import read_csv_strict,atomic_csv,atomic_json,sha256
from .evaluate import HEURISTICS

LABEL={'ppo_task':'PPO: task only','ppo_penalty':'PPO: scalar penalties','consensus':'PPO: full-action consensus',
       'pci':'PCI: typed overlaps','pci_shuffled':'PCI: mismatched maps','global_ppo':'PPO: pooled-state actor',
       'rule':'Greedy rules','market':'Reservation auction','distributed_consensus':'Two-round consensus',
       'centralized_priority':'Centralized priority planner'}
MEASURES=('throughput_per_1000_steps','proposed_compliance','executed_compliance','interventions',
          'collision_attempts','quota_violations','capacity_excess','task_return','jain_delivery','mean_consistency_energy')
KEYS=['method','consistency_coef','layout','scenario','shield','n_agents']

def bootstrap_mean(values,repeats=5000,seed=9217):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if len(a)==0:return None,None,None
    mean=float(a.mean())
    if len(a)<2:return mean,None,None
    rng=np.random.default_rng(seed)
    sampled=a[rng.integers(0,len(a),size=(repeats,len(a)))].mean(axis=1)
    lo,hi=np.quantile(sampled,[.025,.975])
    return mean,float(lo),float(hi)

def nondominated(points):
    p=np.asarray(points,dtype=float);keep=[]
    for i,v in enumerate(p):
        keep.append(not np.any(np.all(p>=v,axis=1)&np.any(p>v,axis=1)))
    return np.asarray(keep,dtype=bool)

def load_verified(directory):
    d=Path(directory);stamp=d/'EVAL_COMPLETE.json'
    if not stamp.exists():raise ValueError(f'Incomplete evaluation: {d}')
    saved=json.loads(stamp.read_text())
    if sha256(d/'evaluation.csv')!=saved['csv_sha256']:raise ValueError(f'Altered evaluation: {d}')
    raw=read_csv_strict(d/'evaluation.csv')
    if not raw:raise ValueError(f'No evaluation rows: {d}')
    frame=pd.DataFrame(raw)
    required=set(KEYS+['run_id','train_seed','eval_episode','eval_seed','deliveries','episode_len',*MEASURES])
    if not required<=set(frame):raise ValueError(f'Missing columns: {required-set(frame)}')
    nums=[c for c in frame if c not in ('method','layout','scenario','run_id','per_agent_deliveries')]
    for c in nums:frame[c]=pd.to_numeric(frame[c].replace('',np.nan),errors='raise')
    for c in ('proposed_compliance','executed_compliance'):
        if not frame[c].between(0,1).all():raise ValueError(f'Invalid probability {c}')
    if frame[['deliveries','episode_len','throughput_per_1000_steps']].isna().any().any():raise ValueError('Missing task counts')
    expected=1000*frame.deliveries/frame.episode_len
    if not np.allclose(expected,frame.throughput_per_1000_steps):raise ValueError('Throughput not computed from deliveries')
    if frame.duplicated(['run_id','scenario','shield','eval_episode']).any():raise ValueError('Duplicate evaluation episodes')
    if not (frame.loc[frame.shield==1,'executed_compliance']==1).all():raise ValueError('Shield violation: inspect before interpreting results')
    return frame

def summarize(frame):
    # Two levels, keeping each trained seed equally weighted despite step counts.
    runmeans=frame.groupby(KEYS+['run_id','train_seed'],dropna=False)[list(MEASURES)].mean().reset_index()
    rows=[]
    for key,g in runmeans.groupby(KEYS,dropna=False):
        base=dict(zip(KEYS,key));method=base['method']
        # Controllers have no trained seeds; episode-level intervals explicitly identified.
        if method in HEURISTICS:
            selector=np.ones(len(frame),dtype=bool)
            for k,v in base.items():selector &= frame[k].to_numpy()==v
            samples=frame.loc[selector];unit='evaluation_episode'
        else:samples=g;unit='training_seed'
        for measure in MEASURES:
            valid=samples[measure].dropna().to_numpy(float)
            mean,lo,hi=bootstrap_mean(valid)
            rows.append({**base,'measure':measure,'mean':mean,'ci_low':lo,'ci_high':hi,
                         'n_units':len(valid),'uncertainty_unit':unit,'interval':'seed/episode percentile bootstrap' if len(valid)>1 else 'not_estimable'})
    return runmeans,pd.DataFrame(rows)

def contrasts(runmeans):
    rows=[]
    core=runmeans[np.isclose(runmeans.consistency_coef,.2)]
    for (layout,scenario,shield,n),g in core.groupby(['layout','scenario','shield','n_agents']):
        for baseline in ('ppo_penalty','consensus','pci_shuffled','global_ppo'):
            left=g[g.method=='pci'].set_index('train_seed');right=g[g.method==baseline].set_index('train_seed')
            common=left.index.intersection(right.index)
            for metric in ('throughput_per_1000_steps','proposed_compliance','interventions'):
                if len(common)==0:continue
                diff=left.loc[common,metric].to_numpy()-right.loc[common,metric].to_numpy()
                m,lo,hi=bootstrap_mean(diff)
                rows.append({'layout':layout,'scenario':scenario,'shield':shield,'n_agents':n,'contrast':f'pci - {baseline}',
                    'measure':metric,'difference':m,'ci_low':lo,'ci_high':hi,'paired_training_seeds':len(common),
                    'interpretation':'exploratory pointwise interval; not multiplicity-adjusted'})
    return pd.DataFrame(rows)

def write_tex(table,out):
    d=table[(table.scenario=='nominal')&(table.shield==1)&(np.isclose(table.consistency_coef,.2)|table.method.isin(HEURISTICS))]
    lines=[r'\begin{tabular}{llrr}',r'\toprule',r'Layout & Method & Deliveries/1k steps & Interventions/episode \\',r'\midrule']
    def fmt(r):
        if r.empty:return '--'
        r=r.iloc[0]
        if pd.isna(r['mean']):return '--'
        if pd.isna(r['ci_low']):return f"{r['mean']:.2f}"
        return f"{r['mean']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
    for (layout,method),g in d.groupby(['layout','method']):
        label=LABEL[method].replace('&',r'\&')
        lines.append(f"{layout} & {label} & {fmt(g[g.measure=='throughput_per_1000_steps'])} & {fmt(g[g.measure=='interventions'])} " + r"\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'% Intervals for learned methods resample training seeds; controller intervals resample evaluation episodes.',r'% These uncertainty units differ. No interval is estimated from one training seed.']
    (out/'nominal_table.tex').write_text('\n'.join(lines)+'\n')

def make_figures(frame,means,table,learning,crypto,diag,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'lines.linewidth':1.2,
        'xtick.major.width':.65,'ytick.major.width':.65,'savefig.facecolor':'white'})
    palette={'pci':'#b22222','ppo_task':'#777777','ppo_penalty':'#111111','consensus':'#444444',
             'pci_shuffled':'#aaaaaa','global_ppo':'#666666'}
    marker={'pci':'o','ppo_penalty':'s','consensus':'^','ppo_task':'v','pci_shuffled':'X','global_ppo':'D'}
    made=[]
    def finish(fig,name):
        fig.tight_layout(pad=.8)
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight')
        plt.close(fig);made.append(name)
    # Actual unsupervised-policy tradeoffs, not an all-ones shield compliance axis.
    for layout in sorted(means.layout.unique()):
        group=means[(means.layout==layout)&(means.scenario=='nominal')&(means.shield==0)&(~means.method.isin(HEURISTICS))]
        conditions=list(group.groupby(['method','consistency_coef']))
        if len(conditions)<2:continue
        fig=plt.figure(figsize=(6.5,4.0));ax=fig.add_subplot(111);points=[]
        for (method,lam),g in conditions:
            x,xl,xh=bootstrap_mean(g.proposed_compliance);y,yl,yh=bootstrap_mean(g.throughput_per_1000_steps)
            color=palette[method];label=LABEL[method]+(rf' ($\lambda$={lam:g})' if method in ('pci','consensus') else '')
            ax.scatter(g.proposed_compliance*100,g.throughput_per_1000_steps,s=12,color=color,alpha=.35,marker=marker[method])
            ax.errorbar(x*100,y,xerr=None if xl is None else [[100*(x-xl)],[100*(xh-x)]],
                yerr=None if yl is None else [[y-yl],[yh-y]],fmt=marker[method],ms=5,mfc='white',mec=color,
                ecolor=color,color=color,capsize=2.5,label=label)
            points.append((x*100,y))
        p=np.asarray(points);front=p[nondominated(p)];front=front[np.argsort(front[:,0])]
        if len(front)>1:ax.plot(front[:,0],front[:,1],':',color='#999999',lw=.8,label='Empirical nondominated means')
        ax.set(xlabel='Proposed joint-action compliance (%)',ylabel='Actual deliveries per 1,000 steps',title=f'Unshielded policy tradeoff: {layout}')
        ax.grid(axis='y',color='.9',lw=.5);ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,fontsize=7)
        fig.text(.02,-.035,'Small markers: training seeds. Bars: seed-bootstrap 95% intervals, only when n ≥ 2.',fontsize=7)
        finish(fig,f'tradeoff_{layout}')
    # Delivered task performance with vs without identical admission for all methods.
    core=table[np.isclose(table.consistency_coef,.2)&(table.scenario=='nominal')&(table.measure=='throughput_per_1000_steps')]
    for layout in sorted(core.layout.unique()):
        g=core[core.layout==layout];methods=[m for m in LABEL if m in set(g.method)]
        if len(methods)<2:continue
        fig=plt.figure(figsize=(6.5,3.8));ax=fig.add_subplot(111)
        for shield,offset,style,color in [(0,-.12,'o','#555555'),(1,.12,'s','#b22222')]:
            for j,method in enumerate(methods):
                r=g[(g.method==method)&(g.shield==shield)]
                if r.empty:continue
                r=r.iloc[0];x=float(r['mean']);lo=r.ci_low;hi=r.ci_high
                ax.errorbar(x,j+offset,xerr=None if pd.isna(lo) else [[x-lo],[hi-x]],fmt=style,color=color,mfc='white',capsize=2,
                    label=('Unshielded' if shield==0 else 'Same admission monitor') if j==0 else None)
        ax.set_yticks(range(len(methods)),[LABEL[m] for m in methods]);ax.set(xlabel='Actual deliveries per 1,000 steps',title=f'Operational cost of admission: {layout}')
        ax.grid(axis='x',color='.9',lw=.5);ax.legend(frameon=False,fontsize=8);finish(fig,f'admission_cost_{layout}')
    # Held-out learning curves: separate seed means per checkpoint.
    if not learning.empty:
        for layout in sorted(learning.layout.unique()):
            g=learning[(learning.layout==layout)&np.isclose(learning.consistency_coef,.2)]
            if g.method.nunique()<2:continue
            fig=plt.figure(figsize=(6.5,3.8));ax=fig.add_subplot(111)
            for method,z in g.groupby('method'):
                values=[]
                for step,t in z.groupby('env_steps'):
                    seedmeans=t.groupby('train_seed').throughput_per_1000_steps.mean()
                    m,lo,hi=bootstrap_mean(seedmeans);values.append((step,m,lo,hi))
                a=np.asarray(values,dtype=float);color=palette[method]
                ax.plot(a[:,0],a[:,1],marker=marker[method],markersize=3,color=color,label=LABEL[method])
                good=np.isfinite(a[:,2])&np.isfinite(a[:,3])
                if good.any():ax.fill_between(a[good,0],a[good,2],a[good,3],color=color,alpha=.10,linewidth=0)
            ax.set(xlabel='Joint environment training steps',ylabel='Held-out deliveries per 1,000 steps',title=f'Policy learning: {layout}')
            ax.legend(frameon=False,fontsize=7);ax.grid(axis='y',color='.9',lw=.5);finish(fig,f'learning_{layout}')
    # Scalability across different populations at test time, no retraining.
    scaled=table[(table.scenario.isin(['scale_2','nominal','scale_8','scale_12']))&(table.shield==1)&(table.layout=='twodoors')&
                 (table.measure=='throughput_per_1000_steps')&np.isclose(table.consistency_coef,.2)]
    if scaled.n_agents.nunique()>1 and scaled.method.nunique()>1:
        fig=plt.figure(figsize=(6.3,3.8));ax=fig.add_subplot(111)
        for method,g in scaled.groupby('method'):
            g=g.sort_values('n_agents');ax.plot(g.n_agents,g['mean'],marker=marker[method],color=palette[method],label=LABEL[method])
            good=g.ci_low.notna()&g.ci_high.notna()
            if good.any():ax.fill_between(g.loc[good,'n_agents'],g.loc[good,'ci_low'],g.loc[good,'ci_high'],color=palette[method],alpha=.1,linewidth=0)
        ax.set(xlabel='Agents at evaluation (trained with 4)',ylabel='Deliveries per 1,000 steps',title='Population transfer with identical admission')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(frameon=False,fontsize=7);finish(fig,'population_transfer')
    if not crypto.empty:
        for n,g in crypto.groupby('n_agents'):
            fig=plt.figure(figsize=(6.0,3.8));ax=fig.add_subplot(111)
            for protocol,color,style in [('groth16','#111111','-'),('plonk','#b22222','--'),('ed25519_disclosure','#777777',':')]:
                v=np.sort(g.loc[g.protocol==protocol,'verify_ms'].to_numpy(float))
                if len(v):ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=protocol,color=color,ls=style)
            ax.set_xscale('log');ax.set(xlabel='Measured verification time (ms, log scale)',ylabel='Empirical cumulative fraction',title=f'Paired transition certificates: {int(n)} agents')
            ax.legend(frameon=False,fontsize=8);finish(fig,f'crypto_verify_n{int(n)}')
    affine=diag/'affine_contradiction.csv'
    if affine.exists():
        g=pd.DataFrame(read_csv_strict(affine)).apply(pd.to_numeric)
        fig=plt.figure(figsize=(5.8,3.7));ax=fig.add_subplot(111)
        for n,color,style in [(3,'#b22222','o'),(12,'#333333','s'),(48,'#888888','^')]:
            z=g[g.contexts==n].sort_values('injected_cycle_gap')
            ax.plot(z.injected_cycle_gap,z.analytic_minimum,color=color,lw=.8)
            ax.scatter(z.injected_cycle_gap,z.minimum_energy,facecolors='white',edgecolors=color,marker=style,s=25,label=f'{n} contexts')
        ax.set(xlabel='Specified affine cycle inconsistency',ylabel='Minimum compatibility energy',title='Contradictory rules: solver vs. analytical oracle')
        ax.legend(frameon=False,fontsize=8);fig.text(.02,-.025,'Markers: numerical solver. Lines: analytic minimum gap² / contexts.',fontsize=7);finish(fig,'affine_contradiction')
    return made

def analyze(root,allow_partial=False):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());out=root/'analysis';out.mkdir(exist_ok=True)
    directories=[root/'runs'/r['run_id'] for r in plan['runs']]+[root/'controllers'/r['run_id'] for r in plan['controllers']]
    missing=[str(d.relative_to(root)) for d in directories if not (d/'EVAL_COMPLETE.json').exists()]
    if missing and not allow_partial:raise RuntimeError(f'{len(missing)} evaluations incomplete. Run status; no automatic partial claims.')
    frames=[];learning=[]
    for d in directories:
        if not (d/'EVAL_COMPLETE.json').exists():continue
        frames.append(load_verified(d))
        p=d/'learning.csv'
        if p.exists():
            f=pd.DataFrame(read_csv_strict(p))
            for c in ('consistency_coef','env_steps','eval_episode','train_seed','throughput_per_1000_steps'):f[c]=pd.to_numeric(f[c],errors='raise')
            learning.append(f)
    if not frames:raise RuntimeError('No completed, valid evaluations to analyze')
    frame=pd.concat(frames,ignore_index=True);L=pd.concat(learning,ignore_index=True) if learning else pd.DataFrame()
    means,summary=summarize(frame);C=contrasts(means)
    frame.to_csv(out/'evaluation_all.csv',index=False);means.to_csv(out/'run_means.csv',index=False);summary.to_csv(out/'condition_summary.csv',index=False)
    if not C.empty:C.to_csv(out/'paired_contrasts.csv',index=False)
    if not L.empty:L.to_csv(out/'learning_all.csv',index=False)
    write_tex(summary,out)
    crypto=pd.DataFrame();p=root/'crypto_benchmark/proof_measurements.csv';stamp=root/'crypto_benchmark/CRYPTO_BENCH_COMPLETE.json'
    if stamp.exists():
        if sha256(p)!=json.loads(stamp.read_text())['measurement_csv_sha256']:raise ValueError('Proof measurement file changed')
        crypto=pd.DataFrame(read_csv_strict(p))
        for c in ('n_agents','witness_and_prove_ms','verify_ms','proof_json_bytes','public_json_bytes','wall_total_ms'):crypto[c]=pd.to_numeric(crypto[c],errors='raise')
        rows=[]
        for (protocol,n),g in crypto.groupby(['protocol','n_agents']):
            for metric in ('witness_and_prove_ms','verify_ms','wall_total_ms','proof_json_bytes','public_json_bytes'):
                a=g[metric].to_numpy(float)
                rows.append({'protocol':protocol,'n_agents':int(n),'metric':metric,'n_measurements':len(a),'median':float(np.median(a)),
                    'p95':float(np.quantile(a,.95)),'mean':float(a.mean()),'scope':'descriptive repeated measurements; not seed-level confidence intervals'})
        atomic_csv(out/'crypto_summary.csv',rows)
    made=make_figures(frame,means,summary,L,crypto,root/'diagnostics',out)
    report={'status':'INCOMPLETE' if missing or not stamp.exists() else 'COMPLETE','missing_evaluations':missing,
            'learned_runs_complete':int(means[~means.method.isin(HEURISTICS)].run_id.nunique()),
            'learned_runs_expected':len(plan['runs']),'figures':made,'crypto_measured':bool(stamp.exists()),
            'uncertainty':'training-seed bootstrap for learning; separately labelled evaluation-episode bootstrap for controllers',
            'warning':'No outcome, Pareto advantage, safety theorem, or acceptance is guaranteed.'}
    atomic_json(out/'ANALYSIS_REPORT.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();print(json.dumps(analyze(a.results,a.allow_partial),indent=2))
