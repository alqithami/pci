#!/usr/bin/env python3
"""Descriptive reporting after COMPLETE_AUDITED_COHORT; never selects partial data."""
import argparse,csv,hashlib,importlib.util,json,math,sys,os,stat
from contextvars import ContextVar
from collections import defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
PINNED_SOURCE='e7fbf20f649ce40a21d42d644756cf519dd6fd2717564c5bbcf0eb0793473ed7'
PROFILES=('capacity_tight','nominal','quota_tight','rule_shift')
COUNTS=('physical_replacements','institutional_replacements','total_replacements','deliveries','pickups','joint_steps')

def need(x,message):
    if not x:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
_SNAPSHOT=ContextVar('input_snapshot',default=None)
class InputSnapshot:
    def __init__(self):self.files={}
    @staticmethod
    def stamp(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    def fresh(self,path):
        p=Path(path).absolute()
        need(p.resolve()==p,'SYMLINK_INPUT_PATH:'+str(p))
        before=p.lstat();need(stat.S_ISREG(before.st_mode),'NONREGULAR_INPUT:'+str(p))
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            first=os.fstat(fd);need(self.stamp(first)==self.stamp(before),'INPUT_CHANGED_BEFORE_READ:'+str(p))
            with os.fdopen(fd,'rb',closefd=False) as f:body=f.read()
            last=os.fstat(fd);after=p.lstat()
            need(self.stamp(first)==self.stamp(last)==self.stamp(after) and len(body)==first.st_size,'INPUT_CHANGED_DURING_READ:'+str(p))
        finally:os.close(fd)
        return body,self.stamp(last)
    def read(self,path):
        p=Path(path).absolute()
        if p not in self.files:
            body,stamp=self.fresh(p);self.files[p]=(body,stamp,hashlib.sha256(body).hexdigest())
        return self.files[p][0]
    def check(self):
        for p,(body,stamp,hashed) in self.files.items():
            current,now=self.fresh(p)
            need(now==stamp and current==body,'INPUT_CHANGED_AFTER_READ:'+str(p))
def read_bytes(p):
    snapshot=_SNAPSHOT.get()
    return snapshot.read(p) if snapshot is not None else Path(p).read_bytes()
def sha(p):return hashlib.sha256(read_bytes(p)).hexdigest()
def unique_pairs(pairs):
    result={}
    for k,v in pairs:
        need(k not in result,'DUPLICATE_JSON_KEY:'+k);result[k]=v
    return result

def load(p):
    p=Path(p);need(p.is_file() and not p.is_symlink(),'MISSING_REGULAR_INPUT:'+str(p))
    return json.loads(read_bytes(p),object_pairs_hook=unique_pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON:'+x)))

def load_protocol(source):
    source=Path(source)
    paths={p.relative_to(source).as_posix():sha(p) for folder in ('runtime','protocol') for p in sorted((source/folder).rglob('*.py')) if '__pycache__' not in p.parts}
    need(digest(paths)==PINNED_SOURCE,'UNQUALIFIED_SOURCE_MAP')
    sys.path.insert(0,str(source/'protocol'))
    for name in ('bridge_design','build_study','frontier'):
        spec=importlib.util.spec_from_file_location(name,source/'protocol'/f'{name}.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return paths,sys.modules['build_study'],sys.modules['frontier']

def keys_exact(rows,keys,fields,label):
    found=[tuple(row[k] for k in fields) for row in rows]
    need(len(found)==len(set(found)) and set(found)==set(keys),'INCOMPLETE_OR_DUPLICATE_'+label)
    return {k:r for k,r in zip(found,rows)}

def validate_bundle(bundle,source,*,_test_only_synthetic=False):
    snapshot=InputSnapshot();token=_SNAPSHOT.set(snapshot)
    try:
        result=_validate_bundle(bundle,source,_test_only_synthetic=_test_only_synthetic)
        snapshot.check()
        result['input_sha256']={p.relative_to(Path(bundle).absolute()).as_posix():value[2] for p,value in snapshot.files.items() if p.is_relative_to(Path(bundle).absolute())}
        result['input_snapshot_stability']='PASS_SINGLE_BODY_HASH_PARSE_AND_FINAL_RECHECK'
        return result
    finally:_SNAPSHOT.reset(token)

def _validate_bundle(bundle,source,*,_test_only_synthetic=False):
    """Verify receipt bindings and independently aggregate all case summaries.

    Raw trace/model bodies are intentionally not required. This authenticates
    the exported audited record internally, not its publisher or external world.
    """
    bundle=Path(bundle);source=Path(source)
    synthetic=(bundle/'SYNTHETIC_TEST_ONLY.json').exists()
    need(not synthetic or _test_only_synthetic,'SYNTHETIC_INPUT_REFUSED_FOR_RESEARCH_REPORT')
    sources,study,frontier=load_protocol(source)
    e=bundle/'cohort/evaluation';c=bundle/'cohort';i=bundle/'inputs'
    schemas={
        c/'STUDY_COMPLETE.json':'PCI-FAIR-COHORT-COMPLETE-v1',
        i/'CONTROLLER_CONFIG.json':'PCI-FAIR-COHORT-CONTROLLER-CONFIG-v1',
        i/'cases/CASE_MANIFEST.json':'PCI-COMPLETE-CASE-MANIFEST-v1',
        i/'SOURCE_DESIGN_LOCK.json':'PCI-COMPLETE-SOURCE-DESIGN-LOCK-v1',
        i/'LAUNCH_LOCK.json':'PCI-FAIR-SCIENTIFIC-LAUNCH-v1',
        i/'CUDA_READINESS.json':'PCI-FAIR-CUDA-READINESS-v1',
        c/'EVALUATION_MANIFEST.json':'PCI-FAIR-EVALUATION-MANIFEST-v1',
        e/'MANIFEST.json':'PCI-FAIR-EVALUATION-MANIFEST-v1',
        e/'ANALYSIS.json':'PCI-FAIR-EVALUATION-AGGREGATE-v1',
        e/'EVALUATION_COMPLETE.json':'PCI-FAIR-EVALUATION-COMPLETE-v1',
        e/'RETAINED_RECORD_AUDIT.json':'PCI-FAIR-RETAINED-RECORD-AUDIT-v1',
    }
    for path,expected_schema in schemas.items():need(load(path).get('schema')==expected_schema,'PRODUCER_SCHEMA:'+path.name)
    final=load(c/'STUDY_COMPLETE.json')
    need(final.get('status')=='COMPLETE_AUDITED_COHORT','INCOMPLETE_STUDY')
    need(final.get('stage')=='development' and final.get('study_domain')=='warehouse','WRONG_STAGE_OR_DOMAIN')
    need(final.get('all_owned_worker_processes_closed') is True,'OWNED_WORKERS_NOT_CLOSED')
    need(final.get('trained_policies')==144 and final.get('training_joint_steps')==37748736,'TRAINING_BUDGET')
    ex=load(c/'EXECUTION_LOCK.json');cfg=load(i/'CONTROLLER_CONFIG.json')
    need(final['execution_lock_sha256']==sha(c/'EXECUTION_LOCK.json'),'EXECUTION_LOCK_HASH')
    need(ex['source_sha256']==sources and final['source_map_sha256']==digest(sources),'SOURCE_BINDING')
    need(ex['controller_configuration']==cfg and ex['controller_configuration_file_sha256']==sha(i/'CONTROLLER_CONFIG.json'),'CONTROLLER_CONFIGURATION_BINDING')
    need(cfg['role']=='scientific' and cfg['stage']=='development' and cfg['study_domain']=='warehouse' and cfg['device']=='cuda','CONFIGURATION_SCOPE')
    need(cfg['worlds_per_profile']==10,'WORLD_COUNT')
    need(final['absolute_deadline_unix']==cfg['deadline_unix'] and final['completed_unix']<=cfg['deadline_unix'],'DEADLINE_BINDING')
    ledger=load(i/'cases/CASE_MANIFEST.json');protocol=load(i/'cases/PROTOCOL_DRAFT.json')
    need(protocol==study.protocol(),'PROTOCOL_BODY_CHANGED')
    need(final['case_manifest_sha256']==ex['case_manifest_sha256']==sha(i/'cases/CASE_MANIFEST.json'),'CASE_MANIFEST_RAW_HASH')
    need(ledger['stage']=='development' and ledger['case_count']==144 and ledger['protocol_canonical_sha256']==digest(protocol),'CASE_MANIFEST_SCOPE')
    design=load(i/'SOURCE_DESIGN_LOCK.json');launch=load(i/'LAUNCH_LOCK.json');ready=load(i/'CUDA_READINESS.json')
    need(design['source_sha256']==sources and design['source_map_sha256']==digest(sources) and design['protocol_canonical_sha256']==digest(protocol),'DESIGN_LOCK_BINDING')
    need(design['scientific_launch_authorized'] is True and design['engineering_complete'] is True,'DESIGN_LOCK_AUTHORITY')
    need(launch['scientific_launch_authorized'] is True and ready['status']=='PASS' and ready['all_required_methods_pass'] is True,'LAUNCH_READINESS')
    expected={'controller_config_sha256':digest(cfg),'case_manifest_sha256':sha(i/'cases/CASE_MANIFEST.json'),'source_design_lock_sha256':sha(i/'SOURCE_DESIGN_LOCK.json'),'readiness_receipt_sha256':sha(i/'CUDA_READINESS.json'),'source_map_sha256':digest(sources),'out':cfg['out'],'deadline_unix':cfg['deadline_unix']}
    need(all(launch.get(k)==v for k,v in expected.items()),'LAUNCH_BINDINGS')
    need(ready['source_map_sha256']==digest(sources) and launch['gpu_uuid']==ready['gpu_uuid'] and launch['host_boot_id']==ready['host_boot_id'],'READY_HOST_SOURCE_BINDING')
    need(all(ex['authority'].get(k)==v for k,v in expected.items()),'EXECUTION_AUTHORITY_BINDING')
    path_map={cfg['source_design_lock']:i/'SOURCE_DESIGN_LOCK.json',cfg['launch_lock']:i/'LAUNCH_LOCK.json',cfg['readiness_receipt']:i/'CUDA_READINESS.json',cfg['protocol_file']:i/'cases/PROTOCOL_DRAFT.json'}
    need(ex['authority']['external_record_sha256']=={k:sha(v) for k,v in path_map.items()},'AUTHORITY_EXTERNAL_RECORDS')
    candidates=study.candidates();candidate_index={x['candidate_id']:x for x in candidates}
    coordinates={(x['candidate_id'],s) for x in candidates for s in study.DEVELOPMENT_SEEDS}
    scheduled=keys_exact(ledger['cases'],coordinates,('candidate_id','training_seed'),'TRAINING_CASES')
    configs={}
    for (candidate,seed),row in scheduled.items():
        path=Path(row['path']);need(not path.is_absolute() and '..' not in path.parts,'UNSAFE_CASE_PATH')
        p=i/'cases'/path;v=load(p);need(sha(p)==row['sha256'],'TRAINING_CONFIG_HASH')
        expected_cfg=study.train_config(candidate_index[candidate],seed,study.DEVELOPMENT_SEEDS.index(seed),role='development',steps=262144,lock_hash=digest(protocol),domain='warehouse')
        need(v==expected_cfg and row['case_id']==v['run_id'],'TRAINING_CONFIG_BODY')
        configs[row['case_id']]=v
    manifest=load(e/'MANIFEST.json')
    need(manifest==load(c/'EVALUATION_MANIFEST.json'),'EVALUATION_MANIFEST_COPIES')
    need(manifest['training_case_manifest']==ledger and manifest['training_case_manifest_sha256']==digest(ledger),'EVALUATION_TRAINING_LEDGER')
    need(manifest['stage']=='development' and manifest['run_role']=='development' and manifest['protocol_sha256']==digest(protocol),'EVALUATION_STAGE')
    need(manifest['worlds_per_profile']==10 and manifest['training_seed_clusters']==study.DEVELOPMENT_SEEDS,'EVALUATION_CLUSTER_DESIGN')
    eval_sources={n.removeprefix('runtime/'):v for n,v in sources.items() if n.startswith(('runtime/training/','runtime/evaluation_fair/'))}
    need(manifest['source_sha256']==eval_sources,'EVALUATION_SOURCE_MAP')
    policies=keys_exact(manifest['policies'],coordinates,('candidate_id','training_seed'),'EVALUATION_POLICIES')
    need(manifest['complete_policy_count']==144 and manifest['complete_episode_count']==5760,'EVALUATION_COUNTS')
    expected_cases={}
    for (candidate,seed),policy in policies.items():
        run=scheduled[candidate,seed]['case_id'];v=configs[run]
        need(policy['directory']==policy['run_id']==run and policy['family']==v['method'] and policy['domain']=='warehouse' and policy['n_agents']==4,'POLICY_DESCRIPTOR')
        cluster=study.DEVELOPMENT_SEEDS.index(seed)
        for profile in sorted(v['profiles']):
            for world in range(10):
                key=f'{run}__{profile}__w{world:02d}';worldseed=3200000000+4000000+1000*cluster+world
                expected_cases[key]={'case_id':key,'policy_directory':run,'candidate_id':candidate,'family':v['method'],'method':v['method'],'training_seed':seed,'domain':'warehouse','n_agents':4,'profile_id':profile,'environment_config':v['profiles'][profile],'world_index':world,'world_seed':worldseed,'policy_seed':worldseed+50000000}
    cases=keys_exact(manifest['cases'],[(k,) for k in expected_cases],('case_id',),'EVALUATION_CASES')
    need(all(row==expected_cases[key[0]] for key,row in cases.items()),'CASE_DESCRIPTOR_CHANGED')
    manifest_hash=digest(manifest)
    complete=load(e/'EVALUATION_COMPLETE.json');audit=load(e/'RETAINED_RECORD_AUDIT.json');analysis=load(e/'ANALYSIS.json');rows=load(e/'development_selection_rows.json');selection=load(c/'DEVELOPMENT_SELECTION.json')
    need(complete['status']=='COMPLETE' and audit['status']=='PASS','INCOMPLETE_EVALUATION_OR_AUDIT')
    need(complete['manifest_sha256']==audit['manifest_sha256']==manifest_hash,'EVALUATION_MANIFEST_BINDING')
    need(final['analysis_sha256']==complete['analysis_sha256']==sha(e/'ANALYSIS.json'),'ANALYSIS_HASH')
    need(final['retained_record_audit_sha256']==sha(e/'RETAINED_RECORD_AUDIT.json'),'AUDIT_HASH')
    need(complete['all_policy_count']==audit['policy_count']==144 and complete['all_episode_count']==audit['episode_count']==final['evaluated_episodes']==5760,'FINAL_EVALUATION_COUNTS')
    need(complete['joint_steps']==audit['joint_steps']==final['evaluation_joint_steps']==1474560 and complete['agent_decisions']==audit['agent_decisions']==final['evaluation_agent_decisions']==5898240,'FINAL_EVALUATION_EXPOSURE')
    for flag in ('independent_numpy_actor','independent_scalar_stage_checks','recorded_action_native_replay'):need(audit[flag] is True,'AUDIT_METHOD_FLAG')
    need(audit['replacement_policy_rollouts']==0,'REPLACEMENT_ROLLOUTS')
    case_reports=keys_exact(audit['case_reports'],[(k,) for k in expected_cases],('case_id',),'AUDIT_CASES')
    need({p.name for p in (e/'cases').iterdir()}=={k+'.json' for k in expected_cases},'CASE_FILE_MEMBERSHIP')
    need(set(complete['case_receipts'])==set(expected_cases),'CASE_RECEIPT_MEMBERSHIP')
    need(set(complete['trace_intent_receipts'])==set(expected_cases),'TRACE_INTENT_RECEIPT_MEMBERSHIP')
    by_profile=defaultdict(list)
    for key,case in expected_cases.items():
        p=e/'cases'/(key+'.json');body=load(p);report=case_reports[key,]
        need(sha(p)==complete['case_receipts'][key],'CASE_RECEIPT_HASH:'+key)
        need(body.get('schema')=='PCI-FAIR-EVALUATION-CASE-v1','PRODUCER_CASE_SCHEMA')
        need(body['status']=='COMPLETE' and body['case']==case and body['manifest_sha256']==manifest_hash,'CASE_SUMMARY_DESCRIPTOR')
        need(report['status']=='PASS' and report['trace_sha256']==body['trace_seal']['sha256'],'CASE_AUDIT_TRACE_BINDING')
        need(report['joint_steps']==body['totals']['joint_steps']==256 and report['agent_decisions']==1024,'CASE_HORIZON')
        policy=policies[case['candidate_id'],case['training_seed']]
        need(body['training_signature']==policy['training_signature'] and body['portable_export_sha256']==policy['portable_export_sha256'],'CASE_POLICY_BINDING')
        need(body['trace_seal']['publication_intent_sha256']==complete['trace_intent_receipts'][key],'CASE_PUBLICATION_INTENT')
        need(body['trace_seal']['records']==258,'CASE_TRACE_RECORD_COUNT')
        for name in COUNTS:
            val=body['totals'][name];need(type(val)is int and val>=0,'INVALID_COUNT:'+name)
        for name in COUNTS[:3]:need(body['totals'][name]<=1024,'REPLACEMENT_DECISION_BOUND:'+name)
        physical,institutional,total=(body['totals'][name] for name in COUNTS[:3])
        need(abs(physical-institutional)<=total<=physical+institutional,'REPLACEMENT_HAMMING_BOUNDS')
        if case['family']=='shieldaware':need(physical==institutional==total==0,'SHIELD_ACTUAL_PROPOSAL_REPLACEMENTS_MUST_BE_ZERO')
        need(body['totals']['deliveries']<=1024,'INVALID_NORMALIZED_COUNTS')
        expected_queries=(256*625,256) if case['family']=='shieldaware' else (0,0)
        need((body['deployment_policy_requested_rows'],body['deployment_policy_callback_calls'])==expected_queries,'PINNED_N4_POLICY_ORACLE_COUNTS')
        by_profile[case['candidate_id'],case['training_seed'],case['profile_id']].append(body)
    profile_rows=[];seed_rows=[];query_rows=[]
    for (candidate,seed),policy in policies.items():
        per=[]
        identity={k:policy[k] for k in ('candidate_id','family','training_seed','domain','n_agents')}
        for profile in sorted(PROFILES):
            summaries=by_profile[candidate,seed,profile];need(len(summaries)==10,'PROFILE_CASE_COUNT')
            counts={k:sum(r['totals'][k] for r in summaries) for k in COUNTS}
            profile_rows.append({**identity,'profile_id':profile,**counts});per.append(counts)
            query_rows.append({**identity,'profile_id':profile,'joint_steps':counts['joint_steps'],'requested_rows':sum(r['deployment_policy_requested_rows'] for r in summaries),'callback_calls':sum(r['deployment_policy_callback_calls'] for r in summaries)})
        counts={k:sum(p[k] for p in per) for k in COUNTS};steps=counts['joint_steps']
        seed_rows.append({**identity,**counts,'cohort_complete':True,'evaluation_complete':True,'profile_weights':{p:.25 for p in sorted(PROFILES)},'institutional_per_1000_agent_decisions':1000*counts['institutional_replacements']/(4*steps),'deliveries_per_1000_joint_steps':1000*counts['deliveries']/steps,'normalized_cost':counts['institutional_replacements']/(4*steps),'normalized_utility':counts['deliveries']/(4*steps)})
    def indexed(values,fields):return keys_exact(values,[tuple(x[k] for k in fields) for x in (seed_rows if len(fields)==2 else profile_rows)],fields,'AGGREGATE_ROWS')
    need(indexed(rows,('candidate_id','training_seed'))==indexed(seed_rows,('candidate_id','training_seed')),'SELECTION_ROWS_RECOMPUTATION')
    need(analysis['status']=='PASS_COMPLETE_COHORT' and analysis['stage']=='development' and analysis['policy_count']==144 and analysis['episode_count']==5760,'ANALYSIS_SCOPE')
    need(indexed(analysis['equal_profile_seed_rows'],('candidate_id','training_seed'))==indexed(seed_rows,('candidate_id','training_seed')),'ANALYSIS_SEED_RECOMPUTATION')
    need(indexed(analysis['profile_seed_rows'],('candidate_id','training_seed','profile_id'))==indexed(profile_rows,('candidate_id','training_seed','profile_id')),'ANALYSIS_PROFILE_RECOMPUTATION')
    expected_selection=frontier.select(rows,source_sha256=sha(e/'development_selection_rows.json'),study_domain='warehouse')
    need(selection==expected_selection,'DECLARED_SELECTION_RECOMPUTATION')
    return {'status':'PASS_RECEIPT_BINDINGS_AND_INDEPENDENT_SUMMARY_AGGREGATION','synthetic_validation_only':synthetic,'seed_rows':seed_rows,'profile_rows':profile_rows,'query_rows':query_rows,'selection':selection,'candidates':candidates,'source_map_sha256':digest(sources),'scope':'Descriptive development summary. Published replay-audit bindings verified; raw policies and trace replay not independently reexecuted by this reporter.'}

def csv_write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def two_point_redundant(points,cost_limit=.05):
    # At/beyond the cost boundary or zero utility contributes no area.
    # Otherwise, either weakly dominated point is redundant in a two-point set.
    a,b=points
    return any(c>=cost_limit or u==0 for c,u in points) or (a[0]<=b[0] and a[1]>=b[1]) or (b[0]<=a[0] and b[1]>=a[1])

def write_report(data,out,source):
    out=Path(out);need(not out.exists(),'OUTPUT_ALREADY_EXISTS');out.mkdir(parents=True)
    _,study,frontier=load_protocol(source);meta={x['candidate_id']:x for x in data['candidates']};rows=data['seed_rows'];selection=data['selection'];means={}
    seed_index={(r['candidate_id'],r['training_seed']):r for r in rows}
    for c in meta:
        cells=[seed_index[c,s] for s in study.DEVELOPMENT_SEEDS]
        means[c]={'candidate_id':c,'family':meta[c]['family'],'beta':meta[c]['penalty_coef'],'lambda':meta[c]['auxiliary_coef'],'institutional_per_1000_agent_decisions':sum(r['institutional_per_1000_agent_decisions'] for r in cells)/3,'deliveries_per_1000_joint_steps':sum(r['deliveries_per_1000_joint_steps'] for r in cells)/3,'selected_for_confirmation':c in selection['selected_candidate_ids'],'outside_primary_rectangle':sum(r['normalized_cost'] for r in cells)/3>study.COST_LIMITS['warehouse']}
    csv_write(out/'candidate_means.csv',list(means.values()))
    csv_write(out/'candidate_seed_points.csv',[{k:v for k,v in r.items() if k!='profile_weights'} for r in rows])
    csv_write(out/'profile_seed_counts.csv',data['profile_rows']);csv_write(out/'deployment_oracle_work.csv',data['query_rows'])
    hv=[]
    for family,details in selection['families'].items():
        chosen=details['selected'];mean_points=[(means[c]['institutional_per_1000_agent_decisions']/1000,means[c]['deliveries_per_1000_joint_steps']/4000) for c in chosen]
        # Analytic redundancy avoids float equality after split area integration.
        dominated_pair=two_point_redundant(mean_points,cost_limit=.05)
        seed_hv=[frontier.hypervolume([(seed_index[c,s]['normalized_cost'],seed_index[c,s]['normalized_utility']) for c in chosen],cost_limit=.05) for s in study.DEVELOPMENT_SEEDS]
        for s,value in zip(study.DEVELOPMENT_SEEDS,seed_hv):hv.append({'family':family,'candidate_1':chosen[0],'candidate_2':chosen[1],'training_seed':s,'selected_pair_seed_hypervolume':value,'selection_hypervolume_of_mean_points':details['development_hypervolume'],'mean_of_selected_pair_seed_hypervolumes':sum(seed_hv)/3,'mean_pair_has_redundant_or_tied_point':dominated_pair})
    csv_write(out/'frozen_selected_pair_hypervolumes.csv',hv)
    render(data,means,out)
    summary={k:v for k,v in data.items() if k not in ('seed_rows','profile_rows','query_rows','candidates')}
    summary.update(report_type='DESCRIPTIVE_DEVELOPMENT_ONLY',independent_training_clusters=3,confirmatory_inference_performed=False,task_equivalence_claim=False,seed_hypervolume_changes_selector=False)
    (out/'REPORT_VALIDATION.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'interpretation.txt').write_text('Development data tune candidates and describe behavior. Three training-seed clusters do not support a confirmatory efficacy claim. Episodes and four profiles are repeated observations within those clusters. All profiles were trained. Hypervolume of mean candidate points is the declared selector; mean seed-wise hypervolume is separately reported and never used to reselect. A redundant second selected point may reflect the declared lexicographic tie rule. Direct λ=0 cells repeat penalty-only objectives and are not extra independent replications. Fixed-versus-Direct candidate sets address finite-grid dose confounding; they do not isolate a novel gradient mechanism. Oracle work counts are recorded computational work, not calibrated deployment or proof latency. Training dual trajectories, gradient norms and training oracle diagnostics are not included in this small evaluation export and require the retained training records. Published replay-audit bindings are checked here; original raw trace/model replay remains in the separately retained cohort.\n')
    return summary

def render(data,means,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    title='SYNTHETIC VALIDATION — NOT RESEARCH RESULTS' if data['synthetic_validation_only'] else 'Warehouse development: 3 training seeds, all candidates retained'
    labels={'fixed':'Fixed penalty','direct':'Direct risk','lagrangian':'PPO-Lagrangian','shieldaware':'Shield-aware'}
    families=['fixed','direct','lagrangian','shieldaware'];colors=['.45','#21618c','.25','.1'];markers=['o','D','s','^']
    fig,ax=plt.subplots(figsize=(7.6,4.7),layout='constrained')
    for family,color,marker in zip(families,colors,markers):
        cells=[m for m in means.values() if m['family']==family]
        seeds=[r for r in data['seed_rows'] if r['family']==family]
        ax.scatter([r['institutional_per_1000_agent_decisions'] for r in seeds],[r['deliveries_per_1000_joint_steps'] for r in seeds],s=11,marker=marker,color=color,alpha=.22)
        ax.scatter([m['institutional_per_1000_agent_decisions'] for m in cells],[m['deliveries_per_1000_joint_steps'] for m in cells],s=38,marker=marker,color=color,label=labels[family])
        chosen=[m for m in cells if m['selected_for_confirmation']];ax.scatter([m['institutional_per_1000_agent_decisions'] for m in chosen],[m['deliveries_per_1000_joint_steps'] for m in chosen],s=110,marker='o',facecolors='none',edgecolors=color,linewidths=1.2)
    ax.set(xlabel='Institutional replacements / 1,000 agent decisions',ylabel='Deliveries / 1,000 joint steps',title=title);ax.legend(frameon=False,ncol=2);ax.grid(alpha=.15)
    fig.text(.5,-.015,'Small points: seeds. Large points: candidate means. Rings: declared selected pair.',ha='center',fontsize=8);save(fig,out,'all_candidate_tradeoffs');plt.close(fig)
    direct=[m for m in means.values() if m['family']=='direct'];betas=sorted({x['beta'] for x in direct});lambdas=sorted({x['lambda'] for x in direct})
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.7),layout='constrained')
    for ax,key,label in zip(axes,['institutional_per_1000_agent_decisions','deliveries_per_1000_joint_steps'],['Institutional replacements / 1,000 decisions','Deliveries / 1,000 joint steps']):
        a=np.array([[next(x[key] for x in direct if x['beta']==b and x['lambda']==l) for l in lambdas] for b in betas]);im=ax.imshow(a,cmap='Blues',aspect='auto');fig.colorbar(im,ax=ax,shrink=.8)
        for j in range(len(betas)):
            for k in range(len(lambdas)):ax.text(k,j,f'{a[j,k]:.2f}',ha='center',va='center',color='white' if a[j,k]>(a.min()+a.max())/2 else 'black',fontsize=8)
        ax.set(xticks=range(len(lambdas)),xticklabels=lambdas,yticks=range(len(betas)),yticklabels=betas,xlabel='Auxiliary coefficient λ',ylabel='Penalty coefficient β',title=label)
    fig.suptitle(title);save(fig,out,'direct_beta_lambda_grid');plt.close(fig)
    fixed=sorted([m for m in means.values() if m['family']=='fixed'],key=lambda x:x['beta'])
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.8),layout='constrained')
    for ax,key,label in zip(axes,['institutional_per_1000_agent_decisions','deliveries_per_1000_joint_steps'],['Replacements / 1,000 decisions','Deliveries / 1,000 joint steps']):
        for seed in [61001,61002,61003]:
            vv=[next(r[key] for r in data['seed_rows'] if r['candidate_id']==m['candidate_id'] and r['training_seed']==seed) for m in fixed]
            ax.plot(range(len(fixed)),vv,'.',color='.65',markersize=4)
        ax.plot(range(len(fixed)),[m[key] for m in fixed],'o-',color='.15',linewidth=.9,markersize=3)
        ax.set(xticks=range(len(fixed)),xticklabels=[str(m['beta']) for m in fixed],xlabel='Tested penalty coefficient β (categorical spacing)',ylabel=label)
        ax.tick_params(axis='x',labelrotation=60);ax.grid(alpha=.15)
    fig.suptitle(title);save(fig,out,'fixed_beta_sweep');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8.2,6.5),layout='constrained',sharex=True,sharey=True)
    for ax,profile in zip(axes.flat,PROFILES):
        for family,color,marker in zip(families,colors,markers):
            cells=[]
            for candidate in [k for k,m in means.items() if m['family']==family]:
                rr=[r for r in data['profile_rows'] if r['candidate_id']==candidate and r['profile_id']==profile]
                cells.append((sum(1000*r['institutional_replacements']/(4*r['joint_steps']) for r in rr)/3,sum(1000*r['deliveries']/r['joint_steps'] for r in rr)/3))
            ax.scatter([x for x,y in cells],[y for x,y in cells],s=25,color=color,marker=marker,label=labels[family])
        ax.set(title=profile,xlabel='Replacements / 1,000 decisions',ylabel='Deliveries / 1,000 joint steps');ax.grid(alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=7);fig.suptitle(title);save(fig,out,'per_profile_tradeoffs');plt.close(fig)

def save(fig,out,name):
    for suffix in ('pdf','svg'):fig.savefig(out/(name+'.'+suffix),bbox_inches='tight')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=validate_bundle(a.bundle,a.source_root);summary=write_report(result,a.out,a.source_root)
    print(json.dumps({'status':summary['status'],'out':str(a.out),'report_type':summary['report_type'],'independent_training_clusters':3}))
if __name__=='__main__':main()
