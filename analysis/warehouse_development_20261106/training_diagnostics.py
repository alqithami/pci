#!/usr/bin/env python3
"""Closed development training diagnostics, independently checked; no efficacy inference."""
import argparse,ast,csv,io,json,math
from pathlib import Path
import bound_inputs as b

TRAIN_SOURCES=('__init__.py','baseline_controls.py','constrained_models.py','environment.py','environment_adapters.py','fair_trainer.py','io.py','model.py','publication_io.py','risk_estimators.py','rware_adapter.py','shield_policy.py')
GRADIENTS=('reward_policy','cost_policy','auxiliary','reward_value','cost_value','entropy','ppo_policy','preclip','clipped')
ROW_INTS=('update','joint_steps','agent_decisions','optimizer_steps','optimization_requested_rows','optimization_evaluated_rows','optimization_cache_hit_rows','optimization_precompute_rows','optimization_certified_zero_rows')
ROW_FLOATS=('mean_task_reward','mean_institutional_cost','mean_product_probe_cost','sampled_penalty_used','dual_used','dual_next','actual_auxiliary_coefficient','reward_advantage_sd','mean_approx_kl','actor_change_l2','parameter_change_l2','elapsed_s')
ROW_NULLABLE=('auxiliary_reference_n4','auxiliary_reference_ratio','mean_auxiliary_to_ppo_grad_norm','mean_auxiliary_ppo_grad_cosine')
COUNTERS=('optimizer_steps','proposed_agent_actions','native_oracle_matches','risk_requested_joint_queries','risk_callback_calls','deployment_shield_uses','collection_oracle_requested_rows','collection_oracle_callback_calls','optimization_oracle_requested_rows','optimization_oracle_callback_calls','collection_oracle_evaluated_rows','optimization_oracle_evaluated_rows','optimization_cache_hit_rows','optimization_precompute_rows','collection_certified_zero_rows','optimization_certified_zero_rows')
NEED_FILES=('TRAIN_COMPLETE.json','CLOSED_FILES.json','config.json','provenance.json','updates.csv')
GRAD_KEYS=tuple('mean_'+x+'_grad_norm' for x in GRADIENTS)
FIELDS=set(ROW_INTS+ROW_FLOATS+ROW_NULLABLE+GRAD_KEYS+('method',))
need=b.need

def near(a,c,label):need(math.isclose(a,c,rel_tol=1e-10,abs_tol=1e-12),label)
def bounded_float(x,label):
    value=float(x);need(math.isfinite(value),'NONFINITE:'+label);return value

def csv_rows(path):
    reader=csv.DictReader(io.StringIO(b.read_bytes(path).decode('utf-8')))
    need(reader.fieldnames is not None and len(set(reader.fieldnames))==len(reader.fieldnames) and set(reader.fieldnames)==FIELDS,'UPDATE_CSV_FIELDS')
    rows=[]
    for raw in reader:
        need(None not in raw and all(v is not None for v in raw.values()),'MALFORMED_CSV_ROW')
        r={'method':raw['method']}
        for k in ROW_INTS:
            need(raw[k].isdigit(),'NONINTEGER_UPDATE_FIELD:'+k);r[k]=int(raw[k])
        for k in ROW_FLOATS+GRAD_KEYS:r[k]=bounded_float(raw[k],k)
        for k in ROW_NULLABLE:r[k]=None if raw[k]=='' else bounded_float(raw[k],k)
        rows.append(r)
    return rows

def runtime_defaults(source):
    tree=ast.parse(b.read_bytes(Path(source)/'runtime/training/pci_risk_train/fair_trainer.py'))
    value=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULTS' for t in n.targets))
    need(isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='dict' and not value.args,'DEFAULTS_SOURCE_SHAPE')
    return {kw.arg:ast.literal_eval(kw.value) for kw in value.keywords}

def expanded_config(raw,defaults):
    c=defaults|raw;c['n_envs']=len(c['env_profile_ids'])
    if c['method'] not in ('direct','compatibility','consensus'):c['auxiliary_coef']=0.
    if c['method'] in ('lagrangian','shieldaware'):c['penalty_coef']=0.
    return c

def check_authority(bundle,source):
    c=bundle/'cohort';i=bundle/'inputs';sources,study,_=b.load_protocol(source)
    schemas={c/'COHORT_STATE.json':'PCI-FAIR-COHORT-STATE-v1',i/'CONTROLLER_CONFIG.json':'PCI-FAIR-COHORT-CONTROLLER-CONFIG-v1',i/'cases/CASE_MANIFEST.json':'PCI-COMPLETE-CASE-MANIFEST-v1',i/'SOURCE_DESIGN_LOCK.json':'PCI-COMPLETE-SOURCE-DESIGN-LOCK-v1',i/'LAUNCH_LOCK.json':'PCI-FAIR-SCIENTIFIC-LAUNCH-v1',i/'CUDA_READINESS.json':'PCI-FAIR-CUDA-READINESS-v1'}
    for p,schema in schemas.items():need(b.load(p).get('schema')==schema,'PRODUCER_SCHEMA:'+p.name)
    ex=b.load(c/'EXECUTION_LOCK.json');cfg=b.load(i/'CONTROLLER_CONFIG.json');ledger=b.load(i/'cases/CASE_MANIFEST.json');protocol=b.load(i/'cases/PROTOCOL_DRAFT.json');state=b.load(c/'COHORT_STATE.json')
    need(ex['source_sha256']==sources,'EXECUTION_SOURCE_BINDING')
    need(ex['controller_configuration']==cfg and ex['controller_configuration_file_sha256']==b.sha(i/'CONTROLLER_CONFIG.json'),'CONTROLLER_CONFIG_BINDING')
    need(cfg['role']=='scientific' and cfg['stage']=='development' and cfg['study_domain']=='warehouse' and cfg['device']=='cuda','CONTROLLER_SCOPE')
    need(state['run_id']==cfg['run_id'] and state['deadline_unix']==cfg['deadline_unix'],'STATE_IDENTITY')
    need(protocol==study.protocol(),'PROTOCOL_BODY')
    need(ex['case_manifest_sha256']==b.sha(i/'cases/CASE_MANIFEST.json'),'LEDGER_HASH')
    need(ledger['stage']=='development' and ledger['case_count']==144 and ledger['protocol_canonical_sha256']==b.digest(protocol),'LEDGER_SCOPE')
    design=b.load(i/'SOURCE_DESIGN_LOCK.json');launch=b.load(i/'LAUNCH_LOCK.json');ready=b.load(i/'CUDA_READINESS.json')
    need(design['source_sha256']==sources and design['source_map_sha256']==b.digest(sources) and design['protocol_canonical_sha256']==b.digest(protocol),'DESIGN_BINDING')
    need(design['scientific_launch_authorized'] is True and design['engineering_complete'] is True,'DESIGN_AUTHORIZATION')
    need(launch['scientific_launch_authorized'] is True and ready['status']=='PASS' and ready['all_required_methods_pass'] is True,'READINESS_AUTHORIZATION')
    expected={'controller_config_sha256':b.digest(cfg),'case_manifest_sha256':b.sha(i/'cases/CASE_MANIFEST.json'),'source_design_lock_sha256':b.sha(i/'SOURCE_DESIGN_LOCK.json'),'readiness_receipt_sha256':b.sha(i/'CUDA_READINESS.json'),'source_map_sha256':b.digest(sources),'out':cfg['out'],'deadline_unix':cfg['deadline_unix']}
    need(all(launch.get(k)==v and ex['authority'].get(k)==v for k,v in expected.items()),'LAUNCH_EXECUTION_BINDINGS')
    need(ready['source_map_sha256']==b.digest(sources) and launch['gpu_uuid']==ready['gpu_uuid'] and state['host_boot_id']==launch['host_boot_id']==ready['host_boot_id'],'HOST_SOURCE_BINDING')
    mapping={cfg['source_design_lock']:i/'SOURCE_DESIGN_LOCK.json',cfg['launch_lock']:i/'LAUNCH_LOCK.json',cfg['readiness_receipt']:i/'CUDA_READINESS.json',cfg['protocol_file']:i/'cases/PROTOCOL_DRAFT.json'}
    need(ex['authority']['external_record_sha256']=={k:b.sha(v) for k,v in mapping.items()},'EXTERNAL_AUTHORITY_BINDINGS')
    candidates=study.candidates();idx={x['candidate_id']:x for x in candidates};keys={(x['candidate_id'],s) for x in candidates for s in study.DEVELOPMENT_SEEDS}
    scheduled=b.keys_exact(ledger['cases'],keys,('candidate_id','training_seed'),'SCHEDULED_CASES');configs={}
    for (candidate,seed),row in scheduled.items():
        path=Path(row['path']);need(not path.is_absolute() and '..' not in path.parts,'UNSAFE_CONFIG_PATH');p=i/'cases'/path;raw=b.load(p)
        want=study.train_config(idx[candidate],seed,study.DEVELOPMENT_SEEDS.index(seed),role='development',steps=262144,lock_hash=b.digest(protocol),domain='warehouse')
        need(raw==want and row['case_id']==want['run_id'] and b.sha(p)==row['sha256'],'EXPECTED_CASE_CONFIGURATION')
        need(row['family']==candidate.split('_')[0] and row['domain']=='warehouse' and row['n_agents']==4 and row['run_role']=='development' and row['total_steps']==262144,'CASE_DESCRIPTOR')
        configs[raw['run_id']]=(candidate,raw)
    need(set(state['completed_training'])==set(configs),'ALL_144_TRAINING_CASES_CLOSED')
    return configs,state,sources,candidates

def validate_updates(rows,cfg,done):
    block=cfg['n_envs']*cfg['rollout_steps'];updates=cfg['total_steps']//block;method=cfg['method'];active=method=='shieldaware' or (method=='direct' and cfg['auxiliary_coef']>0)
    need(len(rows)==updates==256,'UPDATE_COUNT');dual=cfg['dual_initial'];last_elapsed=0.
    for index,r in enumerate(rows,1):
        need(r['update']==index and r['joint_steps']==index*block and r['agent_decisions']==index*block*4,'UPDATE_SEQUENCE_OR_BUDGET')
        need(r['method']==method,'UPDATE_METHOD')
        need(1<=r['optimizer_steps']<=cfg['update_epochs']*(block//cfg['minibatch_groups']),'OPTIMIZER_STEPS_BOUND')
        need(0<=r['mean_institutional_cost']<=1 and 0<=r['mean_product_probe_cost']<=1,'NORMALIZED_COST_BOUND')
        near(r['sampled_penalty_used'],cfg['penalty_coef'],'PENALTY_COEFFICIENT')
        near(r['actual_auxiliary_coefficient'],cfg['auxiliary_coef'],'AUXILIARY_COEFFICIENT')
        need(r['auxiliary_reference_n4']==cfg['auxiliary_reference_n4'] and r['auxiliary_reference_ratio']==(1. if cfg['auxiliary_reference_n4'] is not None else None),'AUXILIARY_REFERENCE')
        near(r['dual_used'],dual if method=='lagrangian' else 0.,'DUAL_USED_CONTINUITY')
        if method=='lagrangian':dual=min(cfg['dual_max'],max(0.,dual+cfg['dual_lr']*(r['mean_institutional_cost']-cfg['dual_target'])))
        near(r['dual_next'],dual,'PROJECTED_DUAL_RECURRENCE')
        need(all(r[k]>=0 for k in GRAD_KEYS),'NEGATIVE_GRADIENT_NORM')
        need(r['mean_clipped_grad_norm']<=cfg['max_grad_norm']*1.0001,'CLIPPING_BOUND')
        need(r['mean_clipped_grad_norm']<=r['mean_preclip_grad_norm']+2e-6,'CLIPPING_INCREASE')
        ppo=r['mean_ppo_policy_grad_norm'];aux=r['mean_auxiliary_grad_norm'];ratio=r['mean_auxiliary_to_ppo_grad_norm'];cos=r['mean_auxiliary_ppo_grad_cosine']
        need((ratio is None)==(ppo==0),'RATIO_MISSINGNESS')
        need(ratio is None or ratio>=0,'NEGATIVE_GRADIENT_RATIO');need(cos is None or -1.00001<=cos<=1.00001,'GRADIENT_COSINE_BOUND')
        if ppo==0:need(cos is None,'ZERO_PPO_COSINE_UNDEFINED')
        if aux==0:need(cos is None and (ratio is None or ratio==0),'ZERO_AUXILIARY_DIAGNOSTICS')
        if method!='lagrangian':
            need(r['mean_cost_policy_grad_norm']==0,'INACTIVE_COST_POLICY_GRADIENT');near(ppo,r['mean_reward_policy_grad_norm'],'PPO_REWARD_NORM_IDENTITY')
        policy_sum=r['mean_reward_policy_grad_norm']+r['mean_cost_policy_grad_norm'];norm_tol=2e-6+2e-5*max(policy_sum,ppo)
        need(abs(r['mean_reward_policy_grad_norm']-r['mean_cost_policy_grad_norm'])<=ppo+norm_tol and ppo<=policy_sum+norm_tol,'PPO_NORM_TRIANGLE')
        component_sum=sum(r['mean_'+k+'_grad_norm'] for k in ('reward_policy','cost_policy','auxiliary','entropy','reward_value','cost_value'))
        need(r['mean_preclip_grad_norm']<=component_sum+2e-6+2e-5*component_sum,'FULL_GRADIENT_NORM_BOUND')
        if cfg['auxiliary_coef']==0:need(aux==0,'INACTIVE_AUXILIARY_GRADIENT')
        if cfg['entropy_coef']==0:need(r['mean_entropy_grad_norm']==0,'INACTIVE_ENTROPY_GRADIENT')
        if method=='shieldaware':need(r['mean_institutional_cost']==0,'SHIELD_TRAINING_COST_ZERO')
        need(r['elapsed_s']>=last_elapsed and r['reward_advantage_sd']>0 and r['actor_change_l2']>=0 and r['parameter_change_l2']>=0,'UPDATE_NONNEGATIVE_DIAGNOSTICS');last_elapsed=r['elapsed_s']
        pre=block*625 if active else 0;hit=r['optimizer_steps']*cfg['minibatch_groups']*625 if active else 0
        need(r['optimization_precompute_rows']==pre and r['optimization_cache_hit_rows']==hit,'OPTIMIZATION_PRECOMPUTE_CACHE_WORK')
        need(r['optimization_requested_rows']==pre+hit and r['optimization_evaluated_rows']+r['optimization_certified_zero_rows']==pre,'OPTIMIZATION_WORK_CONSERVATION')
        need(all(r[k]>=0 for k in ROW_INTS),'NEGATIVE_INTEGER_DIAGNOSTIC')
        if method=='shieldaware':need(r['optimization_certified_zero_rows']==0,'SHIELD_PRECOMPUTE_NOT_ZERO_CERTIFIED')
    near(done['final_dual'],dual,'FINAL_DUAL')
    counters=done['counters'];need(set(counters)==set(COUNTERS) and all(type(v) is int and v>=0 for v in counters.values()),'COUNTER_SCHEMA')
    need(counters['optimizer_steps']==sum(r['optimizer_steps'] for r in rows),'OPTIMIZER_STEP_TOTAL')
    need(counters['proposed_agent_actions']==4*cfg['total_steps'] and counters['native_oracle_matches']==cfg['total_steps'],'COLLECTION_EXPOSURE')
    need(counters['deployment_shield_uses']==(cfg['total_steps'] if method=='shieldaware' else 0),'SHIELD_USE_COUNT')
    need(counters['collection_oracle_requested_rows']==cfg['total_steps']*(627 if method=='shieldaware' else 2),'COLLECTION_REQUESTED_ROWS')
    need(counters['collection_oracle_callback_calls']==cfg['total_steps']//cfg['n_envs']*(3 if method=='shieldaware' else 2),'COLLECTION_CALLBACKS')
    need(counters['collection_oracle_requested_rows']==counters['collection_oracle_evaluated_rows']+counters['collection_certified_zero_rows'],'COLLECTION_WORK_CONSERVATION')
    need(counters['collection_certified_zero_rows']<=cfg['total_steps'],'COLLECTION_CERTIFIED_ZERO_BOUND')
    for key,target in [('requested_rows','oracle_requested_rows'),('evaluated_rows','oracle_evaluated_rows'),('cache_hit_rows','cache_hit_rows'),('precompute_rows','precompute_rows'),('certified_zero_rows','certified_zero_rows')]:
        need(sum(r['optimization_'+key] for r in rows)==counters['optimization_'+target],'OPTIMIZATION_COUNTER_TOTAL:'+key)
    need(counters['optimization_oracle_callback_calls']==(updates*(block//32)+counters['optimizer_steps'] if active else 0),'OPTIMIZATION_CALLBACK_TOTAL')
    need(counters['risk_requested_joint_queries']==(counters['optimization_cache_hit_rows'] if method=='direct' and active else 0),'RISK_REQUESTED_TOTAL')
    need(counters['risk_callback_calls']==(counters['optimizer_steps'] if method=='direct' and active else 0),'RISK_CALLBACK_TOTAL')

def validate(bundle,source,*,_synthetic=False):
    bundle=Path(bundle).absolute();snapshot=b.InputSnapshot();token=b._SNAPSHOT.set(snapshot)
    try:
        synthetic=(bundle/'SYNTHETIC_TEST_ONLY.json').exists();need(not synthetic or _synthetic,'SYNTHETIC_INPUT_REFUSED')
        configs,state,sources,candidates=check_authority(bundle,source);defaults=runtime_defaults(source)
        ts={'pci_risk_train/'+name:sources['runtime/training/pci_risk_train/'+name] for name in TRAIN_SOURCES};trajectories=[];runs=[]
        for run_id,(candidate,raw) in configs.items():
            root=bundle/'cohort/training'/run_id;cfg=b.load(root/'config.json');done=b.load(root/'TRAIN_COMPLETE.json');closed=b.load(root/'CLOSED_FILES.json');prov=b.load(root/'provenance.json')
            need(cfg==expanded_config(raw,defaults),'EXPANDED_CONFIG_MISMATCH')
            need(done.get('schema')=='PCI-FAIR-TRAIN-COMPLETE-v1' and done.get('status')=='COMPLETE','TRAINING_NOT_COMPLETE')
            need(closed.get('schema')=='PCI-FAIR-TRAIN-CLOSED-FILES-v1' and closed.get('signature')==done['signature'],'CLOSED_MANIFEST_SCHEMA')
            controller=state['completed_training'][run_id]
            need(controller['train_complete_sha256']==b.sha(root/'TRAIN_COMPLETE.json') and controller['closed_manifest_sha256']==done['closed_manifest_sha256']==b.sha(root/'CLOSED_FILES.json') and controller['signature']==done['signature'] and controller['joint_steps']==262144,'CONTROLLER_TRAINING_BINDINGS')
            need(all(done[k]==cfg[k] for k in ('run_id','method','domain','seed','run_role','protocol_sha256')),'TRAIN_COMPLETE_IDENTITY')
            need(done['sources']==ts and prov['sources']==ts and prov['version']==done['version']=='PCI-FAIR-INSTITUTIONAL-PPO-ENGINEERING-v1','TRAINING_SOURCE_VERSION')
            signature=b.digest({'configuration':cfg,'sources':ts,'versions':prov['system']['versions'],'dependency_source_sha256':prov['system'].get('dependency_source_sha256',{})})
            need(signature==done['signature']==prov['signature'],'TRAINING_SIGNATURE')
            need(done['joint_steps']==262144 and done['agent_decisions']==1048576 and done['updates']==256 and done['complete_episodes']==1024,'COMPLETE_TRAINING_BUDGET')
            need('optional_publication_aliases' in closed and isinstance(closed['optional_publication_aliases'],dict),'MANIFEST_ALIAS_SCHEMA')
            for name,item in closed['files'].items():
                p=Path(name);need(not p.is_absolute() and '..' not in p.parts and type(item['bytes']) is int and item['bytes']>=0 and isinstance(item['sha256'],str) and len(item['sha256'])==64,'MANIFEST_FILE_DESCRIPTOR')
            for name in ('config.json','provenance.json','updates.csv'):
                item=closed['files'][name];body=b.read_bytes(root/name);need(len(body)==item['bytes'] and b.sha(root/name)==item['sha256'],'DIAGNOSTIC_BODY_HASH:'+name)
            for folder,suffix in [('rollouts','.npz'),('optimization','.json')]:
                need({n for n in closed['files'] if n.startswith(folder+'/')}=={f'{folder}/update_{u:06d}{suffix}' for u in range(1,257)},'CLOSED_UPDATE_MEMBERSHIP:'+folder)
            need(closed['files']['final_export.npz']['sha256']==done['portable_export_sha256'],'CLOSED_EXPORT_BINDING')
            need(done['checkpoint']=='checkpoints/update_000256.pt' and closed['files'][done['checkpoint']]['sha256']==done['checkpoint_sha256'],'FINAL_CHECKPOINT_BINDING')
            expected_phases={p:[1024]*64 for p in cfg['profiles']};need(done['phase_counts']==expected_phases,'BALANCED_PHASE_EXPOSURE')
            rows=csv_rows(root/'updates.csv');validate_updates(rows,cfg,done)
            identity={'run_id':run_id,'candidate_id':candidate,'family':cfg['method'],'training_seed':cfg['seed'],'beta':cfg['penalty_coef'],'lambda':cfg['auxiliary_coef'],'dual_target':cfg['dual_target'],'dual_learning_rate':cfg['dual_lr']}
            for row in rows:trajectories.append(identity|row)
            runs.append(identity|{'training_joint_steps':done['joint_steps'],'final_dual_state':done['final_dual'],'dual_active':cfg['method']=='lagrangian','last_recorded_mean_cost':rows[-1]['mean_institutional_cost'],'last_recorded_mean_aux_to_ppo_ratio':rows[-1]['mean_auxiliary_to_ppo_grad_norm'],'ratio_rows_unavailable':sum(r['mean_auxiliary_to_ppo_grad_norm'] is None for r in rows),'cosine_rows_unavailable':sum(r['mean_auxiliary_ppo_grad_cosine'] is None for r in rows),**done['counters']})
        snapshot.check()
        return {'status':'PASS_ALL_144_CLOSED_TRAINING_DIAGNOSTIC_BINDINGS','synthetic_only':synthetic,'source_map_sha256':b.digest(sources),'cohort_status_at_export':state['status'],'training_cases':144,'update_rows':len(trajectories),'independent_training_clusters':3,'evaluation_completion_asserted':False,'trajectories':trajectories,'runs':runs,'candidates':candidates,'input_sha256':{p.relative_to(bundle).as_posix():v[2] for p,v in snapshot.files.items() if p.is_relative_to(bundle)}}
    finally:b._SNAPSHOT.reset(token)

def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def report(data,out):
    out=Path(out);need(not out.exists(),'OUTPUT_ALREADY_EXISTS');out.mkdir(parents=True)
    write_csv(out/'training_update_diagnostics.csv',data['trajectories']);write_csv(out/'training_case_summary.csv',data['runs'])
    unavailable=[{'diagnostic':'Valid minibatch count for mean ratio/cosine','status':'UNAVAILABLE_IN_MINIMAL_EXPORT','reason':'updates.csv omits denominator counts; optimization/update_*.json is required.'},{'diagnostic':'Per-minibatch gradient distribution and recomputed means','status':'UNAVAILABLE_IN_MINIMAL_EXPORT','reason':'Original optimization JSON is not included; exported CSV means are hash-bound recorded summaries.'},{'diagnostic':'Collection oracle work per update','status':'UNAVAILABLE_IN_MINIMAL_EXPORT','reason':'Only run-total collection counters are exported.'},{'diagnostic':'Oracle CPU/GPU seconds and proof/deployment latency','status':'UNAVAILABLE','reason':'Logical work counters are not calibrated latency measures.'}]
    write_csv(out/'diagnostic_availability.csv',unavailable);render(data,out)
    (out/'interpretation.txt').write_text(INTERPRETATION)
    summary={k:v for k,v in data.items() if k not in ('trajectories','runs','candidates')};summary.update(implementation_sha256={name:b.sha(Path(__file__).resolve().parent/name) for name in ('bound_inputs.py','training_diagnostics.py')},report_scope='DESCRIPTIVE_DEVELOPMENT_TRAINING_DIAGNOSTICS',p_values_computed=False,causal_or_confirmatory_claim=False,raw_model_or_rollout_replay=False,diagnostic_file_body_hashes_verified=True)
    (out/'TRAINING_DIAGNOSTICS_VALIDATION.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');return summary

INTERPRETATION='''These are descriptive diagnostics from all 144 development runs and three independent training-seed clusters. They do not establish efficacy, convergence, generalization, or confirmation. The four warehouse profiles were all trained.

Gradient norms are pre-clipping full-parameter norms of already coefficient-weighted components. The PPO policy vector is the reward-policy plus cost-policy vector, excluding entropy and both critic losses. Each recorded mean_auxiliary_to_ppo_grad_norm is a mean of defined minibatch ratios, not a ratio of mean norms. Null values remain unavailable; valid denominator counts and minibatch distributions require retained optimization JSON. A mean cosine is likewise averaged only where both vectors are nonzero. No missing value is imputed as zero.

The sampled penalty coefficient is beta for Fixed/Direct and zero for Lagrangian/Shield. Only Lagrangian uses dual_used; its next dual is projected using the rollout's institutional cost. Other methods retain dormant dual_next state, which is not an active penalty. Dual trajectories alone do not prove constraint satisfaction.

Oracle requested rows count logical requests. Evaluated rows, certified-zero rows and cached rows partition requested work; precompute rows overlap evaluated/certified-zero rows and must not be added again. Collection includes the common monitor/native-match check and the independent product-policy probe, plus shielding when active. Optimization precomputation and cache-hit work are shown separately. Callback calls batch multiple rows. These counters do not measure GPU/CPU time or proof/execution latency.

The reporter verifies original diagnostic bytes against closed manifests, controller completion bindings, pinned source, launch/configuration signatures, update budgets, dual recurrence, and work accounting. It does not reopen model or rollout bodies, replay the training run, authenticate the publisher, or independently recompute gradient tensors. The controller-state snapshot may predate evaluation completion; no evaluation result is inferred from it.
'''

def render(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    title='SYNTHETIC SOFTWARE CONTROL — NOT RESEARCH DATA' if data['synthetic_only'] else 'Development training diagnostics · 3 training seeds'
    by={r['run_id']:[] for r in data['runs']}
    for row in data['trajectories']:by[row['run_id']].append(row)
    def save(fig,name):
        fig.suptitle(title)
        for suffix in ('pdf','svg'):fig.savefig(out/(name+'.'+suffix),bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(12,8),layout='constrained',sharex=True,sharey=True)
    lag=[c for c in data['candidates'] if c['family']=='lagrangian']
    for ax,c in zip(axes.flat,lag):
        for run in [r for r in data['runs'] if r['candidate_id']==c['candidate_id']]:
            rr=by[run['run_id']];ax.plot([r['joint_steps']/1000 for r in rr],[r['dual_used'] for r in rr],color='#21618c',alpha=.6,lw=.7)
        ax.set_title(f"target={c['dual_target']}; dual lr={c['dual_lr']}");ax.grid(alpha=.15)
    fig.supxlabel('Training joint steps (thousands)');fig.supylabel('Active dual multiplier')
    save(fig,'lagrangian_dual_trajectories')
    fig,axes=plt.subplots(3,4,figsize=(12,8),layout='constrained',sharex=True,sharey=True)
    for ax,c in zip(axes.flat,lag):
        for run in [r for r in data['runs'] if r['candidate_id']==c['candidate_id']]:
            rr=by[run['run_id']];ax.plot([r['joint_steps']/1000 for r in rr],[1000*r['mean_institutional_cost'] for r in rr],color='.35',alpha=.5,lw=.6)
        ax.axhline(1000*c['dual_target'],color='#21618c',ls='--',lw=.8);ax.set(title=f"target={c['dual_target']}; dual lr={c['dual_lr']}");ax.grid(alpha=.15)
    fig.supxlabel('Training joint steps (thousands)');fig.supylabel('Institutional replacements / 1,000 agent decisions')
    save(fig,'lagrangian_cost_and_setpoint')
    direct=[c for c in data['candidates'] if c['family']=='direct'];fig,axes=plt.subplots(3,4,figsize=(12,8),layout='constrained',sharex=True,sharey=True)
    for ax,c in zip(axes.flat,direct):
        for run in [r for r in data['runs'] if r['candidate_id']==c['candidate_id']]:
            rr=by[run['run_id']];ax.plot([r['joint_steps']/1000 for r in rr],[np.nan if r['mean_auxiliary_to_ppo_grad_norm'] is None else r['mean_auxiliary_to_ppo_grad_norm'] for r in rr],color='#21618c',alpha=.55,lw=.7)
        ax.set(title=f"beta={c['penalty_coef']}; lambda={c['auxiliary_coef']}");ax.grid(alpha=.15)
    fig.supxlabel('Training joint steps (thousands)');fig.supylabel('Auxiliary / PPO gradient norm (mean of defined minibatch ratios)')
    save(fig,'direct_gradient_ratio_trajectories')
    # Twelve candidates per family page; all four pages share global scales.
    from matplotlib.backends.backend_pdf import PdfPages
    global_max=max(r[k] for r in data['trajectories'] for k in ('mean_ppo_policy_grad_norm','mean_auxiliary_grad_norm'))
    with PdfPages(out/'all_candidate_component_gradient_norms.pdf') as pages:
        for family in ('fixed','direct','lagrangian','shieldaware'):
            fig,axes=plt.subplots(3,4,figsize=(11,7.5),layout='constrained',sharex=True,sharey=True)
            for ax,c in zip(axes.flat,[c for c in data['candidates'] if c['family']==family]):
                for run in [r for r in data['runs'] if r['candidate_id']==c['candidate_id']]:
                    rr=by[run['run_id']];x=[r['joint_steps']/1000 for r in rr]
                    ax.plot(x,[r['mean_ppo_policy_grad_norm'] for r in rr],color='.35',alpha=.4,lw=.6)
                    ax.plot(x,[r['mean_auxiliary_grad_norm'] for r in rr],color='#21618c',alpha=.55,lw=.6)
                ax.set_title(c['candidate_id']);ax.set_yscale('symlog',linthresh=1e-5)
                ax.set_ylim(-1e-6,max(global_max*1.1,1e-5));ax.set_xlim(0,262.144);ax.grid(alpha=.1)
            fig.supxlabel('Training joint steps (thousands)');fig.supylabel('Mean preclip norm: PPO policy (gray), weighted auxiliary (blue); symlog scale')
            fig.suptitle(title+' | '+family)
            pages.savefig(fig,bbox_inches='tight');fig.savefig(out/('all_candidate_component_gradient_norms_'+family+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    fields=[('collection_oracle_evaluated_rows','Collection evaluated rows'),('optimization_oracle_evaluated_rows','Optimization evaluated rows'),('optimization_cache_hit_rows','Optimization cached rows'),('optimization_oracle_callback_calls','Optimization callback calls')]
    for ax,(key,label) in zip(axes.flat,fields):
        for idx,c in enumerate(data['candidates']):
            rr=[r for r in data['runs'] if r['candidate_id']==c['candidate_id']];ax.scatter([idx]*3,[r[key]/262144 for r in rr],s=9,color='#21618c' if c['family']=='direct' else '.35',alpha=.7)
        ax.set(xticks=[5.5,17.5,29.5,41.5],xticklabels=['Fixed','Direct','Lagrangian','Shield'],ylabel=label+' / training joint step');ax.grid(alpha=.15)
    save(fig,'training_oracle_work')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained',sharex=True)
    for family,ax in zip(('fixed','direct'),axes):
        for run in [r for r in data['runs'] if r['family']==family]:
            rr=by[run['run_id']];ax.plot([r['joint_steps']/1000 for r in rr],[r['sampled_penalty_used'] for r in rr],color='#21618c' if family=='direct' else '.35',alpha=.25,lw=.7)
        ax.set(title=family,xlabel='Training joint steps (thousands)',ylabel='Sampled-reward penalty beta');ax.grid(alpha=.15)
    save(fig,'fixed_and_direct_penalty_coefficients')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    d=validate(a.bundle,a.source_root);result=report(d,a.out);print(json.dumps({k:result[k] for k in ('status','training_cases','update_rows','cohort_status_at_export')}))
if __name__=='__main__':main()
