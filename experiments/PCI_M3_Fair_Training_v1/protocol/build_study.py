#!/usr/bin/env python3
"""Generate a bounded, venue-neutral development/confirmation design.

Default output is DRAFT and cannot authorize scientific training. A separate
root-reviewed source/design lock is required to emit runnable scientific cases.
Engineering fixtures are never analysis replications or tuning observations.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from bridge_design import design as execution_design

SCHEMA = 'PCI-COMPLETE-RESEARCH-PROTOCOL-DRAFT-v1'
FIXED_BETA = [0., .03125, .0625, .125, .25, .5, .75, 1., 1.5, 2., 4., 8.]
DIRECT_GRID = list(itertools.product([0., .25, 1.], [0., .05, .2, .8]))
LAGRANGE_GRID = list(itertools.product([.0025, .005, .01], [.01, .05, .2, 1.]))
SHIELD_GRID = list(itertools.product([1e-4, 3e-4, 1e-3], [0., .005, .02, .05]))
FAMILIES = ['fixed', 'direct', 'lagrangian', 'shieldaware']
COST_LIMITS = {'warehouse':.05, 'rware':.10}
DEVELOPMENT_SEEDS = [61001, 61002, 61003]
CONFIRMATION_SEEDS = list(range(62001, 62021))
REPLICATION_SEEDS = list(range(63001, 63011))
ESTIMATOR_LEARNING_SEEDS = list(range(64001, 64011))
RWARE_DEVELOPMENT_SEEDS = [74001,74002,74003]
RWARE_CONFIRMATION_SEEDS = list(range(75001,75021))
RWARE_REPLICATION_SEEDS = list(range(76001,76011))

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()

def source_map():
    root=Path(__file__).resolve().parents[1]
    paths=sorted(set((root/'runtime').rglob('*.py'))|set((root/'protocol').glob('*.py')))
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

def profiles(n=4, horizon=256):
    base={'layout':'twodoors','n_agents':n,'horizon':horizon,'quota_window':horizon//4,
          'quota':3,'capacity':2,'rule_change':False}
    return {'nominal':base,
            'quota_tight':dict(base,quota=1),
            'capacity_tight':dict(base,capacity=1),
            'rule_shift':dict(base,capacity=1,rule_change=True)}

def candidates():
    out=[]
    for family in FAMILIES:
        grid={'fixed':FIXED_BETA,'direct':DIRECT_GRID,'lagrangian':LAGRANGE_GRID,
              'shieldaware':SHIELD_GRID}[family]
        for i,params in enumerate(grid):
            row={'candidate_id':f'{family}_{i:02d}','family':family,'method':family,
                 'penalty_coef':.25,'auxiliary_coef':0.,'risk_estimator':'exact',
                 'risk_samples':32,'dual_initial':.25,'dual_lr':.05,'dual_target':.005,
                 'learning_rate':3e-4,'entropy_coef':.02}
            if family=='fixed':row['penalty_coef']=params
            elif family=='direct':row.update(penalty_coef=params[0],auxiliary_coef=params[1])
            elif family=='lagrangian':row.update(dual_target=params[0],dual_lr=params[1],penalty_coef=0.)
            else:row.update(learning_rate=params[0],entropy_coef=params[1],penalty_coef=0.)
            out.append(row)
    assert len(out)==48 and all(sum(r['family']==f for r in out)==12 for f in FAMILIES)
    return out

def protocol():
    return {
        'schema':SCHEMA,'status':'DRAFT_NOT_AUTHORIZATION','venue':'unselected',
        'scientific_launch_authorized':False,
        'question':'Does the expected-cost pathway improve a finite tuned task/intervention frontier and measured verified-execution work?',
        'prior_evidence':'M2 fixed configurations motivate design but are not confirmation/tuning observations.',
        'development':{'seeds':DEVELOPMENT_SEEDS,'candidates_per_family':12,
                       'families':FAMILIES,'policies':144,'joint_steps_per_policy':262144,
                       'total_joint_steps':144*262144,'candidate_grid':candidates(),
                       'profiles':profiles(),'training_profile_weights':{p:.25 for p in profiles()},
                       'profile_exposure':'All four profiles are trained; evaluation on new worlds is not unseen-profile generalization.',
                       'evaluation_worlds_per_profile_per_cluster':10,
                       'maximum_wall_seconds_proposed':86400,
                       'budget_status':'Requires engineering throughput estimate and root-reviewed lock before scientific launch.'},
        'selection':{'data':'Development seeds/worlds only.',
                     'rule':'Choose two candidates per family jointly maximizing finite-candidate dominated hypervolume; deterministic candidate_id tie-break.',
                     'identical_objective_cells':'Direct lambda=0 cells repeat penalty-only objectives; they are controls, not additional independent replications.',
                     'utility':'deliveries / (n_agents * joint_steps)',
                     'cost':'institutional replacements / (n_agents * joint_steps)',
                     'cost_limits_by_domain':COST_LIMITS,
                     'reference_rectangle':'Within each domain, cost in [0, declared domain bound], utility in [0,1]; normalize area by that bound. No pooled cross-domain hypervolume claim.',
                     'cost_rectangle_rationale':'Custom .05 brackets prior M2 task-only cost (~.0473); native .10 brackets the reference-only N4 RWARE cost (~.0544). These are ranges of scientific interest, not compliance standards. Both are fixed before any new M3 outcomes and shared by every family within a domain.',
                     'sensitivity':'Retain every raw point, outside-rectangle flags, and descriptive areas for cost limits .01,.025,.05,.1,1; do not select a bound after outcomes.',
                     'warning':'This is a finite, equally budgeted candidate family, not global optimization or an application-specific noninferiority margin.'},
        'confirmation':{'seeds':CONFIRMATION_SEEDS,'configurations_per_family':2,
                        'policies':160,'joint_steps_per_policy':524288,
                        'total_joint_steps':160*524288,
                        'checkpoint_rule':'Final prescribed step only; retained learning curves do not select test checkpoints.',
                        'evaluation_worlds_per_profile_per_cluster':20,
                        'primary_endpoint':'Within each independent training-seed cluster, dominated hypervolume of the two development-selected final policies.',
                        'primary_test_family':['Warehouse Direct minus Fixed','Warehouse Direct minus Lagrangian','Warehouse Direct minus shield-aware','RWARE Direct minus Fixed','RWARE Direct minus Lagrangian','RWARE Direct minus shield-aware'],
                        'inference':'Paired whole training-seed-cluster exact sign-flip tests within each domain; these six learning contrasts and six controlled-execution contrasts form one global Holm12 family. Cluster bootstrap intervals are descriptive. Until all12 contrasts close, partial-family tests are descriptive and cannot be called a completed familywise result. The historical M2 family is unchanged.',
                        'direction':'Report two-sided tests, raw differences, intervals and every seed. A nonsignificant result is not equivalence.',
                        'power_rationale':'Twenty clusters increases resolution and permits uncertainty assessment, but is not a prospective power guarantee; engineering/development variability must inform final lock before these seeds are opened.',
                        'maximum_wall_seconds_proposed':172800},
        'estimator_ablation':{'same_objective':'Same J, beta, lambda and optimizer on fixed states; exact, current-policy score with LOO baseline, and conditional enumeration.',
                              'K':[2,8,32,128],'N_reference':4,'N_scaling':[8,12],
                              'trained_controlled_comparison':{'seeds':ESTIMATOR_LEARNING_SEEDS,'estimators':['exact','score','conditional'],'samples_for_training':32,'configurations':'Two NONZERO-lambda Direct configurations jointly maximizing the same development-only two-point hypervolume among the nine nonzero-lambda cells; deterministic candidate-ID tie break. This separately declared mechanism-ablation selection does not alter primary frontier selection. Lambda-zero cells remain identity controls and never count as an active-estimator comparison.','joint_steps_per_policy':524288,'policies':60,'total_joint_steps':60*524288,'inference_scope':'Fresh matched training clusters; separate from fixed-state engineering. Exact-minus-score and exact-minus-conditional are a separately declared two-test Holm family. All scheduled cases run even if nonzero auxiliaries fail to improve development frontiers; such a result provides no algorithmic-advantage claim.'},
                              'signals':['gradient bias/error','variance','norm/cosine','oracle queries','checker seconds','optimizer work'],
                              'no_novelty_claim':'Finite conditional/score gradients are established machinery.'},
        'native_rware':{'development_seeds':RWARE_DEVELOPMENT_SEEDS,'confirmation_seeds':RWARE_CONFIRMATION_SEEDS,'replication_seeds':RWARE_REPLICATION_SEEDS,
                       'development':'Own equal12-candidate-per-family N4 tuning, three clusters,144policies;262144jointsteps perdevelopmentpolicy aswarehouse.',
                       'confirmation':'Own two development-selected configurations per family;20freshclusters;160policies×524288steps.',
                       'replication':'OwnN4-tuned configurations transferredtoN8/N12 usingdeclaredlambda normalization; tenfreshclusters, no claim globallyoptimizedlargeNcomparators.',
                       'task_and_institution':'NativeINDIVIDUALdeliveryreward, no shaping; qualifiedsmall-layouttransport-band simultaneous-entry quotas/capacity, notcustomstandingoccupancy.',
                       'profile':'Explicitreference-only-qualifiedsource-boundmanifestrequired; full54cell/180episodenegativescreenretained; no M3learnerused toselectprofile.',
                       'primary_multiplicity':'Three native contrasts join three warehouse learning contrasts and six controlled-execution contrasts in one global Holm12 family. No separate Holm3/Holm6 collection is presented as control of the whole integrated headline.'},
        'replication':{'seeds':REPLICATION_SEEDS,'domains':['custom N8','custom N12'],
                       'candidate_policy':'Two development-selected configurations per family, transferred unchanged unless a separately locked domain-specific tuning study is conducted.',
                       'estimator':'Conditional sampling at prequalified K; exact N4 retained as reference only.',
                       'coefficient_size_normalization':'lambda_N=lambda_selected_at_N4*4/N preserves explicit joint-risk/per-agent PPO averaging scale. Freeze before replication outcomes; log actual component-gradient ratios because optimization behavior need not be identical.',
                       'profiles':'Must qualify genuine task feasibility and institutional exposure before scientific outcomes; preserve native RWARE action/physics/rewards and explicit institution wrapper.',
                       'rware_profile_gate':'A separate source-bound native RWARE profile manifest and reference-only feasibility/exposure receipt is required; custom warehouse profiles are not silently reused.',
                       'sampled_work_budget':'Proposed N>=8: K2 and16 uniformly sampled auxiliary states per128-group minibatch; sequential shield entropy16 states. All are engineering variance/runtime admission choices, frozen before learning outcomes. N4 estimator learning ablation uses allstates andK32.',
                       'independence':'These seeds/worlds do not overlap development, confirmation, M2, or reserved historical A2 namespaces.'},
        'bridge':execution_design(),
        'integrity':{'required':['Original cold-image root-cause localization','Atomic consistent capture with identities and WAL handling','R1 file-lifecycle diagnosis','Named-boundary fault injection','Complete retained-record audit'],
                     'limits':'Process interruption is not physical power loss. New passing fixtures do not explain old failures.'},
        'common_controls':['same within-domain actor feature and observation access','two separately fitted critics in every arm','same native transition/task and institution cost','matched init/world/proposal namespaces','complete fixed budgets and final-checkpoint rule','component pre-clipping gradient diagnostics','actual oracle/CPU/GPU/wall work retained'],
        'exclusions':['No partial-cohort success claim','No outcome-based stopping or case replacement','No task equivalence/noninferiority inferred from nonsignificance','No guaranteed improvement before results','No submission operation in this protocol'],
    }

def train_config(row, seed, cluster, *, role, n=4, domain='warehouse', steps=262144, lock_hash=None, stage=None, domain_profiles=None, domain_slots=None):
    stage=stage or {'engineering':'engineering','development':'development','confirmatory':'confirmation'}[role]
    tag={'engineering':1,'development':2,'confirmation':3,'replication':4,'estimator_ablation':5}[stage]
    namespace=2_000_000_000+tag*100_000_000+{'warehouse':0,'rware':20_000_000}[domain]+n*1_000_000
    if domain=='rware' and domain_profiles is None:raise ValueError('EXPLICIT_NATIVE_RWARE_PROFILE_MANIFEST_REQUIRED')
    active_profiles=profiles(n) if domain_profiles is None else domain_profiles
    if any(v['n_agents']!=n for v in active_profiles.values()):raise ValueError('PROFILE_AGENT_COUNTS')
    if domain_slots is None:
        if 8%len(active_profiles):raise ValueError('EXPLICIT_EIGHT_ENVIRONMENT_PROFILE_SLOTS_REQUIRED')
        domain_slots=[p for p in active_profiles for _ in range(8//len(active_profiles))]
    if len(domain_slots)!=8 or any(p not in active_profiles for p in domain_slots):raise ValueError('EIGHT_VALID_PROFILE_SLOTS_REQUIRED')
    cfg={
        'schema':'PCI-FAIR-TRAIN-CONFIG-v1','run_id':f'm3_{stage}_{domain}_n{n}_{row["candidate_id"]}_e{row["risk_estimator"]}_s{seed}',
        'run_role':role,'domain':domain,'method':row['method'],'n_agents':n,
        'profiles':active_profiles,'env_profile_ids':domain_slots,
        'env_seeds':[namespace+1000*cluster+e for e in range(8)],
        'seed':seed,'init_seed':seed,'policy_seed':namespace+100000+cluster,
        'minibatch_seed':namespace+200000+cluster,'auxiliary_seed':namespace+300000+cluster,
        'entropy_seed':namespace+400000+cluster,'probe_seed':namespace+500000+cluster,
        'total_steps':steps,'rollout_steps':128,'hidden':64,'update_epochs':4,'minibatch_groups':128,
        'gamma':.99,'cost_gamma':.99,'gae_lambda':.95,'cost_gae_lambda':.95,'clip_coef':.2,
        'vf_coef':.5,'cost_vf_coef':.5,'max_grad_norm':.5,'target_kl':.03,
        'checkpoint_every_updates':1,'device':'cpu' if role=='engineering' else 'cuda','torch_threads':1,
        'dual_max':20.,'shield_policy':'exact' if n==4 else 'sequential',
    }
    cfg.update({k:v for k,v in row.items() if k not in ('candidate_id','family')})
    cfg['auxiliary_reference_n4']=float(cfg['auxiliary_coef']) if cfg['method']=='direct' else None
    cfg['auxiliary_coef']=float(cfg['auxiliary_coef'])*4/n
    if lock_hash is not None:cfg['protocol_sha256']=lock_hash
    if n>4 and cfg['method']=='direct':cfg.update(risk_estimator='conditional',risk_samples=2,risk_minibatch_groups=16)
    if n>4 and cfg['method']=='shieldaware':cfg['shield_entropy_groups']=16
    cfg['run_id']=f'm3_{stage}_{domain}_n{n}_{row["candidate_id"]}_e{cfg["risk_estimator"]}_s{seed}'
    return cfg

def emit(out:Path, *, stage='draft', lock:Path|None=None, selected:Path|None=None, engineering_n=4, engineering_domain='warehouse', native_profiles:Path|None=None, study_domain='warehouse'):
    out.mkdir(parents=True,exist_ok=True)
    p=protocol();body=canonical(p);(out/'PROTOCOL_DRAFT.json').write_bytes(body+b'\n')
    lock_hash=None;record={}
    if stage in ('development','confirmation','replication','estimator_ablation'):
        if lock is None:raise ValueError('Scientific case emission requires a separate reviewed source/design lock')
        record=json.loads(lock.read_text())
        if record.get('schema')!='PCI-COMPLETE-SOURCE-DESIGN-LOCK-v1' or record.get('scientific_launch_authorized') is not True:
            raise ValueError('Scientific source/design lock is not authorized')
        if record.get('protocol_canonical_sha256')!=hashlib.sha256(body).hexdigest():raise ValueError('Design lock does not match this protocol')
        if not record.get('engineering_complete'):raise ValueError('Engineering/source gate missing')
        current_sources=source_map()
        if record.get('source_sha256')!=current_sources or record.get('source_map_sha256')!=digest(current_sources):raise ValueError('CURRENT_SOURCE_DESIGN_LOCK_REQUIRED')
        lock_hash=hashlib.sha256(body).hexdigest()
    rows=[]
    def native_for(n,engineering=False):
        native=json.loads(native_profiles.read_text()) if native_profiles else record.get('native_rware_profiles_by_n',{}).get(str(n))
        if native and 'profiles_by_n' in native:native=native['profiles_by_n'].get(str(n))
        if not native:raise ValueError('EXPLICIT_NATIVE_RWARE_PROFILES_REQUIRED')
        if not engineering and (native.get('schema')!='PCI-NATIVE-RWARE-SCIENTIFIC-PROFILES-v1' or native.get('status')!='ROOT_APPROVED_FOR_SOURCE_DESIGN_LOCK' or not native.get('qualification_sha256')):
            raise ValueError('SOURCE_BOUND_QUALIFIED_NATIVE_RWARE_PROFILES_REQUIRED')
        if not engineering:
            from native_profiles import build
            if native!=build()['profiles_by_n'][str(n)]:raise ValueError('NATIVE_PROFILE_REFERENCE_QUALIFICATION_IDENTITY')
        return native
    if stage=='engineering':
        choices=[r for r in candidates() if r['candidate_id'] in ('fixed_04','direct_06','lagrangian_05','shieldaware_05')]
        native=native_for(engineering_n,True) if engineering_domain=='rware' else None
        for c,row in enumerate(choices):rows.append(train_config(row,71001,c,role='engineering',steps=2048,n=engineering_n,domain=engineering_domain,domain_profiles=native['profiles'] if native else None,domain_slots=native.get('env_profile_ids') if native else None))
    elif stage=='development':
        seeds=DEVELOPMENT_SEEDS if study_domain=='warehouse' else RWARE_DEVELOPMENT_SEEDS
        native=native_for(4) if study_domain=='rware' else None
        for c,seed in enumerate(seeds):
            for row in candidates():rows.append(train_config(row,seed,c,role='development',steps=262144,lock_hash=lock_hash,domain=study_domain,domain_profiles=native['profiles'] if native else None,domain_slots=native.get('env_profile_ids') if native else None))
    elif stage in ('confirmation','replication','estimator_ablation'):
        if selected is None:raise ValueError('Confirmation needs development-only selection receipt')
        sel=json.loads(selected.read_text())
        expected_seeds=DEVELOPMENT_SEEDS if study_domain=='warehouse' else RWARE_DEVELOPMENT_SEEDS
        if sel.get('schema')!='PCI-DEVELOPMENT-FRONTIER-SELECTION-v1' or sel.get('development_seeds')!=expected_seeds or sel.get('study_domain','warehouse')!=study_domain:
            raise ValueError('Invalid development-only selection receipt')
        if sel.get('cost_limit')!=COST_LIMITS[study_domain] or sel.get('primary_test_data_opened') is not False:
            raise ValueError('DEVELOPMENT_SELECTION_ENDPOINT_OR_HOLDOUT_MISMATCH')
        by_id={r['candidate_id']:r for r in candidates()};ids=sel.get('selected_candidate_ids')
        if not isinstance(ids,list) or len(ids)!=8 or len(set(ids))!=8:raise ValueError('Need exactly two candidates per family')
        chosen=[by_id[i] for i in ids]
        if any(sum(r['family']==f for r in chosen)!=2 for f in FAMILIES):raise ValueError('Unbalanced selection')
        settings=[(study_domain,4)] if stage in ('confirmation','estimator_ablation') else [(study_domain,8),(study_domain,12)]
        if stage=='estimator_ablation' and study_domain!='warehouse':raise ValueError('ESTIMATOR_LEARNING_ABLATION_IS_PRESPECIFIED_WAREHOUSE_ONLY')
        stage_seeds={'confirmation':CONFIRMATION_SEEDS,'replication':REPLICATION_SEEDS,'estimator_ablation':ESTIMATOR_LEARNING_SEEDS} if study_domain=='warehouse' else {'confirmation':RWARE_CONFIRMATION_SEEDS,'replication':RWARE_REPLICATION_SEEDS}
        seeds=stage_seeds[stage]
        if stage=='estimator_ablation':
            auxiliary_ids=sel.get('estimator_ablation_candidate_ids')
            if not isinstance(auxiliary_ids,list) or len(auxiliary_ids)!=2 or len(set(auxiliary_ids))!=2:
                raise ValueError('TWO_PREDECLARED_NONZERO_AUXILIARY_DEVELOPMENT_SELECTIONS_REQUIRED')
            active=[by_id[i] for i in auxiliary_ids]
            if any(r['family']!='direct' or r['auxiliary_coef']<=0 for r in active):
                raise ValueError('ZERO_AUXILIARY_IS_IDENTITY_CONTROL_NOT_ESTIMATOR_ABLATION')
            chosen=[dict(r,risk_estimator=e,candidate_id=r['candidate_id']+'_'+e) for r in active for e in ('exact','score','conditional')]
        for domain,n in settings:
            native=None
            if domain=='rware':native=native_for(n)
            for c,seed in enumerate(seeds):
                for row in chosen:rows.append(train_config(row,seed,c,role='confirmatory',n=n,domain=domain,steps=524288,lock_hash=lock_hash,stage=stage,domain_profiles=native['profiles'] if native else None,domain_slots=native.get('env_profile_ids') if native else None))
    index={'schema':'PCI-COMPLETE-CASE-MANIFEST-v1','stage':stage,'study_domain':engineering_domain if stage=='engineering' else study_domain,'status':'ENGINEERING_ONLY' if stage=='engineering' else ('DRAFT_NOT_RUNNABLE' if stage=='draft' else 'SOURCE_LOCK_REQUIRED_AT_CONTROLLER'),
           'scientific_launch_authorized':False,'protocol_canonical_sha256':hashlib.sha256(body).hexdigest(),
           'case_count':len(rows),'cases':[]}
    for cfg in rows:
        name=cfg['run_id']+'.json';data=canonical(cfg)+b'\n';destination=out/name
        if destination.exists() and destination.read_bytes()!=data:raise ValueError('EXISTING_CASE_CONFIGURATION_CHANGED_USE_NEW_NAMESPACE')
        destination.write_bytes(data)
        import re
        match=re.search(r'_(fixed|direct|lagrangian|shieldaware)_[0-9]{2}(?:_(score|conditional|exact))?_e',cfg['run_id'])
        if match is None:raise ValueError('Missing candidate identity')
        candidate_id=match.group(0)[1:-2]
        index['cases'].append({'case_id':cfg['run_id'],'candidate_id':candidate_id,'family':cfg['method'],'domain':cfg['domain'],'n_agents':cfg['n_agents'],'training_seed':cfg['seed'],'path':name,'sha256':hashlib.sha256(data).hexdigest(),'run_role':cfg['run_role'],'total_steps':cfg['total_steps'],'risk_estimator':cfg['risk_estimator'],'actual_auxiliary_coef':cfg['auxiliary_coef']})
    (out/'CASE_MANIFEST.json').write_text(json.dumps(index,sort_keys=True,indent=2)+'\n')
    return index

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',choices=['draft','engineering','development','confirmation','replication','estimator_ablation'],default='draft')
    ap.add_argument('--source-design-lock',type=Path);ap.add_argument('--development-selection',type=Path)
    ap.add_argument('--engineering-n',type=int,choices=[4,8,12],default=4)
    ap.add_argument('--engineering-domain',choices=['warehouse','rware'],default='warehouse')
    ap.add_argument('--native-profile-manifest',type=Path)
    ap.add_argument('--study-domain',choices=['warehouse','rware'],default='warehouse')
    a=ap.parse_args();r=emit(a.output,stage=a.stage,lock=a.source_design_lock,selected=a.development_selection,engineering_n=a.engineering_n,engineering_domain=a.engineering_domain,native_profiles=a.native_profile_manifest,study_domain=a.study_domain)
    print(json.dumps({k:v for k,v in r.items() if k!='cases'}))
