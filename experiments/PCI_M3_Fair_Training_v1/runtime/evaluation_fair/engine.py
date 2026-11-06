"""Final-policy evaluation and deterministic retained-action native replay.

Replay does not collect replacement episodes or sample a new policy trajectory.
It reexecutes retained actions in the pinned native implementation, checks task
and generator continuity, independently recomputes actor/sampling and reconciles
scalar stage counts. It is not an independent proof of upstream native physics.
"""
from __future__ import annotations
import argparse
import copy
import importlib.metadata
import itertools
import json
import os
import re
from pathlib import Path
import time
import numpy as np
import torch
from training.pci_risk_train.environment_adapters import make_environment,make_oracle,snapshot_oracle
from training.pci_risk_train.fair_trainer import runtime_sources,fair_system_meta
from .actor import PortableActor,verify_training_policy,LOGIT_ATOL,PROBABILITY_ATOL
from .common import (canonical,closed_json,digest,environment_state,plain,read_trace,
                     runtime_source_map,sha,ClosedTrace,trace_exists_or_pending,
                     verify_trace_structure,verify_trace_bytes)
from .policy import choose,scalar_stages

STAGE_BASE={'engineering':3000000000,'development':3200000000,
            'confirmation':3400000000,'replication':3600000000,'estimator_ablation':3800000000}
CORE_KEYS=('physical_replacements','institutional_replacements','total_replacements')


def _safe_relative(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('Safe relative artifact path required')
    return p


def evaluation_seeds(stage,domain,n,cluster,world):
    if stage not in STAGE_BASE or domain not in ('warehouse','rware') or not 1<=n<=32:
        raise ValueError('Explicit stage/domain/N namespace required')
    if not 0<=cluster<100 or not 0<=world<50:raise ValueError('Namespace bounds: <100 clusters/<50 worlds')
    base=STAGE_BASE[stage]+(100000000 if domain=='rware' else 0)+1000000*n+1000*cluster+world
    return base,base+50000000


def prepare_manifest(training_root,policies,*,stage,worlds_per_profile,protocol_sha256=None,
                     training_case_manifest=None):
    """Bind all final policies, fixed world streams and execution source closure.

    policies is a complete prespecified list with directory,candidate_id,family,
    training_seed. The caller supplies the whole training manifest, not a
    success-only filter. Scientific manifests additionally require source design
    lock/expected policy IDs, checked by the outer cohort controller.
    """
    if type(worlds_per_profile)is not int or not 1<=worlds_per_profile<=50:raise ValueError('1..50 complete worlds required')
    if not policies:raise ValueError('Empty policy cohort')
    if stage not in STAGE_BASE:raise ValueError('Unknown stage')
    if stage!='engineering':
        if not isinstance(training_case_manifest,dict) or training_case_manifest.get('schema')!='PCI-COMPLETE-CASE-MANIFEST-v1':
            raise ValueError('Scientific evaluation needs the complete prespecified training case ledger')
        if training_case_manifest.get('stage')!=stage or training_case_manifest.get('case_count')!=len(training_case_manifest.get('cases',[])):
            raise ValueError('Training ledger stage/count mismatch')
        if training_case_manifest.get('protocol_canonical_sha256')!=protocol_sha256:
            raise ValueError('Training ledger and evaluation protocol differ')
    root=Path(training_root);sources=runtime_sources()
    keys=[r['directory'] for r in policies]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate policy directories')
    seeds=sorted({int(r['training_seed']) for r in policies})
    if len(seeds)>=100:raise ValueError('Too many cluster namespaces')
    bound=[];cases=[]
    for row in policies:
        relative=_safe_relative(row['directory']);p=root/relative
        cfg,done,_=verify_training_policy(p,current_sources=sources)
        if cfg['seed']!=row['training_seed']:raise RuntimeError('Training seed descriptor mismatch')
        if stage=='engineering':
            if cfg['run_role']!='engineering':raise ValueError('Engineering must remain isolated from scientific policies')
        else:
            role='development' if stage=='development' else 'confirmatory'
            if cfg['run_role']!=role or not protocol_sha256 or cfg.get('protocol_sha256')!=protocol_sha256:
                raise ValueError('Scientific training role/protocol does not match evaluation')
        identity={'directory':str(relative),'candidate_id':row['candidate_id'],'family':row['family'],
            'training_seed':cfg['seed'],'run_id':cfg['run_id'],'domain':cfg['domain'],'n_agents':cfg['n_agents'],
            'train_complete_sha256':sha(p/'TRAIN_COMPLETE.json'),'config_sha256':sha(p/'config.json'),
            'closed_manifest_sha256':done['closed_manifest_sha256'],
            'portable_export_sha256':done['portable_export_sha256'],'training_signature':done['signature'],
            'model_specification':done['model']}
        bound.append(identity);cluster=seeds.index(cfg['seed'])
        for profile in sorted(cfg['profiles']):
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,50}',profile):raise ValueError('Safe profile ID required')
            config=cfg['profiles'][profile]
            for w in range(worlds_per_profile):
                ws,ps=evaluation_seeds(stage,cfg['domain'],cfg['n_agents'],cluster,w)
                case_id=f'{cfg["run_id"]}__{profile}__w{w:02d}'
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,180}',case_id):raise ValueError('Safe case ID required')
                cases.append({'case_id':case_id,'policy_directory':str(relative),
                    'candidate_id':row['candidate_id'],'family':row['family'],'method':cfg['method'],
                    'training_seed':cfg['seed'],'domain':cfg['domain'],'n_agents':cfg['n_agents'],
                    'profile_id':profile,'environment_config':config,'world_index':w,
                    'world_seed':ws,'policy_seed':ps})
    if len({r['case_id'] for r in cases})!=len(cases):raise ValueError('Duplicate evaluation cases')
    if stage!='engineering':
        expected=[r['case_id'] for r in training_case_manifest['cases']]
        if len(expected)!=len(set(expected)) or set(expected)!={r['run_id'] for r in bound}:
            raise ValueError('Every prescheduled final training policy must close before evaluation')
    return {'schema':'PCI-FAIR-EVALUATION-MANIFEST-v1','stage':stage,
        'run_role':'engineering' if stage=='engineering' else ('development' if stage=='development' else 'confirmatory'),
        'protocol_sha256':protocol_sha256,'source_sha256':runtime_source_map(),
        'training_case_manifest':training_case_manifest,
        'training_case_manifest_sha256':digest(training_case_manifest) if training_case_manifest is not None else None,
        'worlds_per_profile':worlds_per_profile,'training_seed_clusters':seeds,'policies':bound,'cases':cases,
        'complete_policy_count':len(bound),'complete_episode_count':len(cases),
        'sampling_contract':'CPU Torch float32 actor; independent NumPy float32 check; float64 renormalized inverse CDF; N+1 PCG64 uniforms at every step',
        'independence':'Stage/domain/N namespaces disjoint; policies and profiles in a cluster share worlds and uniform streams. Visited states can differ.',
        'audit_scope':'Recorded-action native transition replay plus independent actor/sampling/scalar correction reconciliation; no replacement episodes; no independent native-physics proof.'}


def verify_manifest(manifest,training_root):
    if manifest.get('schema')!='PCI-FAIR-EVALUATION-MANIFEST-v1':raise RuntimeError('Evaluation manifest schema')
    if manifest['source_sha256']!=runtime_source_map():raise RuntimeError('Evaluation/training executing source closure changed')
    if manifest['complete_policy_count']!=len(manifest['policies']) or manifest['complete_episode_count']!=len(manifest['cases']):
        raise RuntimeError('Evaluation cohort count mismatch')
    if len({r['case_id'] for r in manifest['cases']})!=len(manifest['cases']):raise RuntimeError('Duplicate case ledger')
    stage=manifest['stage']
    if stage not in STAGE_BASE or manifest['worlds_per_profile'] not in range(1,51):raise RuntimeError('Stage/world bounds')
    root=Path(training_root);loaded={};sources=runtime_sources()
    for b in manifest['policies']:
        p=root/_safe_relative(b['directory'])
        cfg,done,actor=verify_training_policy(p,current_sources=sources)
        actual={'train_complete_sha256':sha(p/'TRAIN_COMPLETE.json'),'config_sha256':sha(p/'config.json'),
            'closed_manifest_sha256':done['closed_manifest_sha256'],
            'portable_export_sha256':done['portable_export_sha256'],'training_signature':done['signature']}
        if any(b[k]!=v for k,v in actual.items()):raise RuntimeError('Final policy binding changed')
        if cfg['run_id']!=b['run_id'] or cfg['seed']!=b['training_seed'] or done['model']!=b['model_specification']:
            raise RuntimeError('Policy descriptor mismatch')
        if manifest['stage']!='engineering' and cfg.get('protocol_sha256')!=manifest['protocol_sha256']:
            raise RuntimeError('Scientific protocol identity mismatch')
        expected_role='engineering' if stage=='engineering' else ('development' if stage=='development' else 'confirmatory')
        if cfg['run_role']!=expected_role:raise RuntimeError('Training/evaluation role contamination')
        if b['directory'] in loaded:raise RuntimeError('Duplicate policy binding')
        loaded[b['directory']]=(cfg,done,actor)
    if manifest['training_seed_clusters']!=sorted({x[0]['seed'] for x in loaded.values()}):
        raise RuntimeError('Training cluster ledger mismatch')
    if stage!='engineering':
        ledger=manifest['training_case_manifest']
        if digest(ledger)!=manifest['training_case_manifest_sha256'] or ledger['case_count']!=len(ledger['cases']):
            raise RuntimeError('Prespecified training case ledger identity mismatch')
        if ledger['stage']!=stage or ledger['protocol_canonical_sha256']!=manifest['protocol_sha256']:
            raise RuntimeError('Training stage/protocol ledger mismatch')
        ids=[r['case_id'] for r in ledger['cases']]
        if len(ids)!=len(set(ids)) or set(ids)!={cfg['run_id'] for cfg,_,_ in loaded.values()}:
            raise RuntimeError('Missing scheduled policy from final cohort')
    expected_cases={}
    for b in manifest['policies']:
        cfg,_,_=loaded[b['directory']]
        cluster=manifest['training_seed_clusters'].index(cfg['seed'])
        for profile,config in sorted(cfg['profiles'].items()):
            for w in range(manifest['worlds_per_profile']):
                ws,ps=evaluation_seeds(stage,cfg['domain'],cfg['n_agents'],cluster,w)
                ident=f'{cfg["run_id"]}__{profile}__w{w:02d}'
                expected_cases[ident]={'case_id':ident,'policy_directory':b['directory'],
                    'candidate_id':b['candidate_id'],'family':b['family'],'method':cfg['method'],
                    'training_seed':cfg['seed'],'domain':cfg['domain'],'n_agents':cfg['n_agents'],
                    'profile_id':profile,'environment_config':config,'world_index':w,'world_seed':ws,'policy_seed':ps}
    if {r['case_id']:r for r in manifest['cases']}!=expected_cases:
        raise RuntimeError('Every scheduled profile/world/stream must appear exactly once')
    for case in manifest['cases']:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,180}',case['case_id']):raise RuntimeError('Unsafe case ledger path')
        if case['policy_directory'] not in loaded:raise RuntimeError('Case references an unbound policy')
        cfg,_,_=loaded[case['policy_directory']]
        if (case['training_seed'],case['domain'],case['n_agents'],case['method'])!=(cfg['seed'],cfg['domain'],cfg['n_agents'],cfg['method']):
            raise RuntimeError('Case policy identity mismatch')
        if cfg['profiles'][case['profile_id']]!=case['environment_config']:raise RuntimeError('Undeclared evaluation environment')
    return loaded


def _stage(oracle,actions,env,*,independent=True):
    result=oracle.evaluate(actions[None,None])
    row={k:int(result[k][0,0]) for k in CORE_KEYS}
    for key in ('physical_actions','admitted_actions','admitted_native_actions','physical_positions','admitted_positions'):
        if key in result:row[key]=plain(result[key][0,0])
    row.setdefault('admitted_native_actions',row['admitted_actions'])
    if independent:
        scalar=plain(scalar_stages(env,actions))
        if any(row[k]!=scalar[k] for k in scalar):raise RuntimeError('Independent scalar/native stage mismatch')
    return row


def _header(case,cfg,done,actor,env,manifest_hash):
    return {'kind':'header','schema':'PCI-FAIR-POLICY-TRACE-v1','case':case,
        'manifest_sha256':manifest_hash,'training_signature':done['signature'],
        'portable_export_sha256':done['portable_export_sha256'],'model_specification':actor.spec,
        'policy_configuration':{k:cfg[k] for k in ('method','domain','shield_policy')},
        'initial_world_state':environment_state(env),
        'authoritative_inference':'CPU Torch float32 actor-only',
        'native_audit_scope':'Pinned native recorded-action replay, independent scalar correction; not independent native physics or new policy rollout'}


def evaluate_case(case,cfg,done,actor,out,manifest_hash):
    out=Path(out);trace=out/'traces'/f'{case["case_id"]}.jsonl.gz'
    complete=out/'cases'/f'{case["case_id"]}.json'
    if complete.exists():raise RuntimeError('Use retained-record audit for an existing complete case')
    if trace_exists_or_pending(trace):raise RuntimeError('Unsealed trace/private staging requires explicit recovery/quarantine, not overwrite')
    totals={k:0 for k in CORE_KEYS};totals.update(deliveries=0,pickups=0,joint_steps=0,task_return=0.)
    deployment_rows=0;deployment_calls=0
    timings=[]
    start=time.perf_counter()
    with ClosedTrace(trace) as writer:
        # Permanent intent is installed before creating this evaluation world.
        env=make_environment(case['domain'],case['environment_config'],case['world_seed'])
        rng=np.random.Generator(np.random.PCG64(case['policy_seed']))
        writer.append(_header(case,cfg,done,actor,env,manifest_hash))
        for t in range(env.config.horizon):
            before=environment_state(env);features=env.features()
            light=snapshot_oracle(env);policy_oracle=make_oracle([light],domain=case['domain'])
            rng_before=copy.deepcopy(rng.bit_generator.state);uniforms=rng.random(env.n+1)
            astart=time.perf_counter();logits,probs=actor.authoritative(features)
            actor_check_s=time.perf_counter()-astart
            pstart=time.perf_counter()
            actions,logq,details=choose(logits,probs,policy_oracle,uniforms,cfg,check_torch=True)
            selection_check_s=time.perf_counter()-pstart
            if environment_state(env)!=before:raise RuntimeError('Inference/oracle mutated live world or RNG')
            # Static/scalar validation work is separate from deployed-policy
            # mask-query counts and cannot be called deployment latency.
            stages=_stage(make_oracle([light],domain=case['domain']),actions,env)
            if cfg['method']=='shieldaware' and stages['total_replacements']!=0:
                raise RuntimeError('A declared shielded proposal requires correction')
            nstart=time.perf_counter()
            _,task,_,is_done,metrics=env.transition(actions,enforce=True)
            transition_s=time.perf_counter()-nstart
            canonical_actions=plain(env.last.get('canonical_admitted_actions',env.last['actions']))
            native_commands=plain(env.last.get('admitted_actions',env.last['actions']))
            if canonical_actions!=stages['admitted_actions'] or native_commands!=stages['admitted_native_actions']:
                raise RuntimeError('Authoritative native action differs from stage prediction')
            after=environment_state(env)
            for k in CORE_KEYS:totals[k]+=stages[k]
            totals['joint_steps']+=1;totals['deliveries']+=int(metrics['delivered'])
            totals['pickups']+=int(metrics['pickups']);totals['task_return']+=float(np.sum(task,dtype=np.float64))
            deployment_rows+=int(policy_oracle.counters['requested_rows'])
            deployment_calls+=int(policy_oracle.counters['callback_calls'])
            writer.append({'kind':'step','step':t,'before':before,'features':features.tolist(),
                'features_sha256':digest(features.tolist()),'oracle_snapshot':light,
                'policy_rng_before':rng_before,'uniforms':uniforms.tolist(),'policy_rng_after':rng.bit_generator.state,
                'authoritative_logits':logits.tolist(),'authoritative_product_probabilities':probs.tolist(),
                'proposal':actions.tolist(),'deployed_joint_log_probability':logq,'deployed_policy_details':details,
                'policy_oracle_counters':policy_oracle.counters,'stages':stages,
                'task_rewards':np.asarray(task,dtype=np.float32).tolist(),'metrics':metrics,'done':bool(is_done),
                'native_record':env.last,'after':after})
            timings.append({'step':t,'actor_and_independent_numpy_check_seconds':actor_check_s,
                            'policy_selection_and_independent_torch_check_seconds':selection_check_s,
                            'native_simulation_transition_seconds':transition_s})
            if is_done!=(t+1==env.config.horizon):raise RuntimeError('Evaluation horizon mismatch')
        writer.append({'kind':'footer','totals':totals,'final_world_state_sha256':digest(environment_state(env)),
                       'final_policy_rng':rng.bit_generator.state})
    seal=writer.close();n=env.n;steps=totals['joint_steps']
    summary={'schema':'PCI-FAIR-EVALUATION-CASE-v1','status':'COMPLETE','case':case,
        'manifest_sha256':manifest_hash,'training_signature':done['signature'],
        'portable_export_sha256':done['portable_export_sha256'],'trace':trace.relative_to(out).as_posix(),
        'trace_seal':seal,'totals':totals,
        'institutional_per_1000_agent_decisions':1000*totals['institutional_replacements']/(n*steps),
        'deliveries_per_1000_joint_steps':1000*totals['deliveries']/steps,
        'normalized_cost':totals['institutional_replacements']/(n*steps),
        'normalized_utility':totals['deliveries']/(n*steps),
        'deployment_policy_requested_rows':deployment_rows,'deployment_policy_callback_calls':deployment_calls,
        'inference_max_logit_error':actor.max_logit_error,'inference_max_probability_error':actor.max_probability_error,
        'operational_measurements':f'operations/{case["case_id"]}.json',
        'scientific_performance_claim':False if cfg['run_role']=='engineering' else None}
    closed_json(out/summary['operational_measurements'],{
        'schema':'PCI-FAIR-EVALUATION-OPERATIONAL-MEASUREMENTS-v1',
        'case_id':case['case_id'],'manifest_sha256':manifest_hash,
        'training_signature':done['signature'],'trace_sha256':seal['sha256'],
        'elapsed_seconds_including_validation':time.perf_counter()-start,
        'step_timings':timings,
        'trace_publication':writer.lifecycle,
        'scope':'Measured simulation/reference-check work; validation overhead included. These are not isolated deployment or proof latencies.'})
    closed_json(complete,summary);return summary


def audit_case(summary,cfg,done,actor,out,manifest_hash):
    """Replay the *recorded* action stream; never sample a replacement episode."""
    if summary.get('schema')!='PCI-FAIR-EVALUATION-CASE-v1' or summary.get('status')!='COMPLETE':
        raise RuntimeError('Closed complete evaluation case required')
    case=summary['case'];out=Path(out)
    if summary['manifest_sha256']!=manifest_hash or summary['training_signature']!=done['signature']:
        raise RuntimeError('Case source/policy binding mismatch')
    operational_path=_safe_relative(summary['operational_measurements'])
    if str(operational_path)!=f'operations/{case["case_id"]}.json':
        raise RuntimeError('Operational measurement path mismatch')
    operations=json.loads((out/operational_path).read_text())
    if (operations.get('schema')!='PCI-FAIR-EVALUATION-OPERATIONAL-MEASUREMENTS-v1' or
        operations['case_id']!=case['case_id'] or operations['manifest_sha256']!=manifest_hash or
        operations['training_signature']!=done['signature'] or
        operations['trace_sha256']!=summary['trace_seal']['sha256']):
        raise RuntimeError('Operational measurement identity mismatch')
    if len(operations['step_timings'])!=case['environment_config']['horizon']:
        raise RuntimeError('Operational timing closure incomplete')
    for index,row in enumerate(operations['step_timings']):
        if row.get('step')!=index or any(not np.isfinite(v) or v<0 for k,v in row.items() if k!='step'):
            raise RuntimeError('Operational timing fields invalid')
    if not np.isfinite(operations['elapsed_seconds_including_validation']) or operations['elapsed_seconds_including_validation']<0:
        raise RuntimeError('Operational elapsed measurement invalid')
    rows=iter(read_trace(out/_safe_relative(summary['trace']),summary['trace_seal']))
    header=next(rows)
    env=make_environment(case['domain'],case['environment_config'],case['world_seed'])
    expected_header=_header(case,cfg,done,actor,env,manifest_hash)
    if canonical(header)!=canonical(expected_header):raise RuntimeError('Trace header/initial world mismatch')
    rng=np.random.Generator(np.random.PCG64(case['policy_seed']))
    totals={k:0 for k in CORE_KEYS};totals.update(deliveries=0,pickups=0,joint_steps=0,task_return=0.)
    deployment_rows=0;deployment_calls=0
    for t in range(env.config.horizon):
        row=next(rows)
        if row['kind']!='step' or row['step']!=t or canonical(row['before'])!=canonical(environment_state(env)):
            raise RuntimeError('Retained world-state continuity mismatch')
        features=env.features()
        if digest(features.tolist())!=row['features_sha256'] or not np.array_equal(features,np.asarray(row['features'],dtype=np.float32)):
            raise RuntimeError('Retained observations do not follow native state')
        light=snapshot_oracle(env)
        if canonical(light)!=canonical(row['oracle_snapshot']):raise RuntimeError('Light snapshot mismatch')
        if canonical(rng.bit_generator.state)!=canonical(row['policy_rng_before']):raise RuntimeError('Policy RNG prefix mismatch')
        uniforms=rng.random(env.n+1)
        if not np.array_equal(uniforms,row['uniforms']) or canonical(rng.bit_generator.state)!=canonical(row['policy_rng_after']):
            raise RuntimeError('Stored uniform stream/RNG successor mismatch')
        logits,probs=actor.authoritative(features)
        retained_logits=np.asarray(row['authoritative_logits'],dtype=np.float32)
        retained_probs=np.asarray(row['authoritative_product_probabilities'],dtype=np.float64)
        if (retained_logits.shape!=logits.shape or retained_probs.shape!=probs.shape or
            not np.isfinite(retained_probs).all() or np.any(retained_probs<0) or
            not np.allclose(retained_probs.sum(-1),1,atol=1e-12,rtol=0) or
            not np.allclose(logits,retained_logits,atol=LOGIT_ATOL,rtol=0) or
            not np.allclose(probs,retained_probs,atol=PROBABILITY_ATOL,rtol=0)):
            raise RuntimeError('Retained authoritative inference fails declared independent conformance')
        # The retained authoritative probabilities define the original draw.
        # A numerically conforming audit must not replace that draw when a
        # uniform lies near a CDF boundary on a different CPU/BLAS build.
        logits,probs=retained_logits,retained_probs
        policy_oracle=make_oracle([light],domain=case['domain'])
        actions,logq,details=choose(logits,probs,policy_oracle,uniforms,cfg,check_torch=True)
        if actions.tolist()!=row['proposal'] or abs(logq-row['deployed_joint_log_probability'])>1e-12 or canonical(details)!=canonical(row['deployed_policy_details']):
            raise RuntimeError('Stored proposal is not inverse-CDF of the declared deployed policy')
        if canonical(policy_oracle.counters)!=canonical(row['policy_oracle_counters']):
            raise RuntimeError('Retained deployed-policy oracle work mismatch')
        deployment_rows+=int(policy_oracle.counters['requested_rows'])
        deployment_calls+=int(policy_oracle.counters['callback_calls'])
        stages=_stage(make_oracle([light],domain=case['domain']),actions,env)
        if canonical(stages)!=canonical(row['stages']):raise RuntimeError('Retained scalar stage accounting mismatch')
        # Deterministic replay uses retained actions, not fresh policy draws.
        _,task,_,done_flag,metrics=env.transition(np.asarray(row['proposal'],dtype=np.int64),enforce=True)
        if canonical(env.last)!=canonical(row['native_record']) or canonical(environment_state(env))!=canonical(row['after']):
            raise RuntimeError('Native task/RNG/state replay mismatch')
        if not np.array_equal(np.asarray(task,dtype=np.float32),row['task_rewards']) or canonical(metrics)!=canonical(row['metrics']) or bool(done_flag)!=row['done']:
            raise RuntimeError('Task reward/termination/metric replay mismatch')
        for k in CORE_KEYS:totals[k]+=stages[k]
        totals['joint_steps']+=1;totals['deliveries']+=int(metrics['delivered'])
        totals['pickups']+=int(metrics['pickups']);totals['task_return']+=float(np.sum(task,dtype=np.float64))
    footer=next(rows)
    if canonical(footer)!=canonical({'kind':'footer','totals':totals,'final_world_state_sha256':digest(environment_state(env)),
                                     'final_policy_rng':rng.bit_generator.state}) or totals!=summary['totals']:
        raise RuntimeError('Retained final state/counts mismatch')
    if deployment_rows!=summary['deployment_policy_requested_rows'] or deployment_calls!=summary['deployment_policy_callback_calls']:
        raise RuntimeError('Retained complete deployed-policy work accounting mismatch')
    try:next(rows);raise RuntimeError('Unexpected extra trace records')
    except StopIteration:pass
    return {'case_id':case['case_id'],'status':'PASS','joint_steps':totals['joint_steps'],
            'agent_decisions':totals['joint_steps']*env.n,'trace_sha256':summary['trace_seal']['sha256'],
            'audit_scope':'Recorded-action pinned native replay; independent actor, inverse-CDF sampling, scalar corrections and task/RNG/state continuity'}


def aggregate(manifest,summaries):
    expected={c['case_id'] for c in manifest['cases']}
    if len(summaries)!=len(expected) or {r['case']['case_id'] for r in summaries}!=expected:
        raise RuntimeError('Complete exact episode cohort required for aggregation')
    by_policy={}
    for r in summaries:by_policy.setdefault(r['case']['policy_directory'],[]).append(r)
    rows=[];profile_rows=[]
    for binding in manifest['policies']:
        values=by_policy[binding['directory']]
        profiles=sorted({r['case']['profile_id'] for r in values})
        for profile in profiles:
            cell=[r for r in values if r['case']['profile_id']==profile]
            if len(cell)!=manifest['worlds_per_profile']:raise RuntimeError('Incomplete profile cell')
            counts={k:sum(r['totals'][k] for r in cell) for k in (*CORE_KEYS,'deliveries','pickups','joint_steps')}
            profile_rows.append({**{k:binding[k] for k in ('candidate_id','family','training_seed','domain','n_agents')},
                'profile_id':profile,**counts})
        counts={k:sum(r['totals'][k] for r in values) for k in (*CORE_KEYS,'deliveries','pickups','joint_steps')}
        per_profile=[r for r in profile_rows if r['candidate_id']==binding['candidate_id'] and r['training_seed']==binding['training_seed'] and r['domain']==binding['domain'] and r['n_agents']==binding['n_agents']]
        # Raw sums equal the macro profile estimand only with balanced horizons.
        if len({r['joint_steps'] for r in per_profile})!=1:raise RuntimeError('Unequal profile exposures require an explicit different aggregate, not raw pooling')
        n=binding['n_agents'];steps=counts['joint_steps']
        rows.append({**{k:binding[k] for k in ('candidate_id','family','training_seed','domain','n_agents')},
            **counts,'cohort_complete':True,'evaluation_complete':True,
            'profile_weights':{p:1/len(profiles) for p in profiles},
            'institutional_per_1000_agent_decisions':1000*counts['institutional_replacements']/(n*steps),
            'deliveries_per_1000_joint_steps':1000*counts['deliveries']/steps,
            'normalized_cost':counts['institutional_replacements']/(n*steps),
            'normalized_utility':counts['deliveries']/(n*steps)})
    return {'schema':'PCI-FAIR-EVALUATION-AGGREGATE-v1','status':'PASS_COMPLETE_COHORT',
        'stage':manifest['stage'],'scientific_performance_claim':False if manifest['stage']=='engineering' else None,
        'policy_count':len(manifest['policies']),'episode_count':len(summaries),
        'equal_profile_seed_rows':rows,'profile_seed_rows':profile_rows}


def verify_case_retention(summary,out,*,structure=True):
    out=Path(out);p=out/_safe_relative(summary['trace'])
    if structure:verify_trace_structure(p,summary['trace_seal'])
    else:verify_trace_bytes(p,summary['trace_seal'])
    # Read the actual published case, not just an in-memory success object.
    c=summary['case'];record=out/'cases'/f'{c["case_id"]}.json'
    if canonical(json.loads(record.read_text()))!=canonical(summary):
        raise RuntimeError('Retained case receipt differs from collected summary')
    return {'case_id':c['case_id'],'trace_sha256':summary['trace_seal']['sha256'],
            'records':summary['trace_seal']['records'],'status':'PASS_RETAINED_BYTES'}


def publish_completion(manifest,out,summaries,*,retained_structure_verified=False):
    """Publish only after the entire exact case cohort has closed."""
    out=Path(out);mh=digest(manifest)
    _retained_membership(manifest,out)
    expected={c['case_id']:c for c in manifest['cases']}
    if len(summaries)!=len(expected) or {r['case']['case_id'] for r in summaries}!=set(expected):
        raise RuntimeError('Complete exact summary cohort required before retention checks')
    for summary in summaries:
        if summary['case']!=expected[summary['case']['case_id']]:raise RuntimeError('Collected summary case descriptor differs from manifest')
    # A close-time hash alone cannot establish that the bytes survived later
    # work. Check every actual trace after all producers close, before analysis
    # or a global COMPLETE marker. Parallel mode performs structure readback in
    # a closed worker phase, then this parent repeats every raw byte identity.
    for summary in summaries:verify_case_retention(summary,out,structure=not retained_structure_verified)
    _retained_membership(manifest,out)
    _publication_runtime_guard(manifest,out)
    result=aggregate(manifest,summaries)
    closed_json(out/'ANALYSIS.json',result)
    closed_json(out/'development_selection_rows.json',result['equal_profile_seed_rows'])
    operation_receipts={r['case']['case_id']:sha(out/_safe_relative(r['operational_measurements'])) for r in summaries}
    closed_json(out/'EVALUATION_COMPLETE.json',{'schema':'PCI-FAIR-EVALUATION-COMPLETE-v1',
        'status':'COMPLETE','manifest_sha256':mh,'analysis_sha256':sha(out/'ANALYSIS.json'),
        'all_policy_count':len(manifest['policies']),'all_episode_count':len(summaries),
        'joint_steps':sum(r['totals']['joint_steps'] for r in summaries),
        'agent_decisions':sum(r['totals']['joint_steps']*r['case']['n_agents'] for r in summaries),
        'case_receipts':{r['case']['case_id']:sha(out/'cases'/f'{r["case"]["case_id"]}.json') for r in summaries},
        'trace_intent_receipts':{r['case']['case_id']:r['trace_seal']['publication_intent_sha256'] for r in summaries},
        'operational_measurement_receipts':operation_receipts})
    return result


def _retained_membership(manifest,out):
    """Reject pending writers or any extra/missing retained cohort object."""
    out=Path(out);cases={c['case_id'] for c in manifest['cases']}
    for folder,expected in (('cases',{f'{c}.json' for c in cases}),
                            ('traces',{f'{c}.jsonl.gz' for c in cases}),
                            ('trace_intents',{f'{c}.jsonl.gz.intent.json' for c in cases}),
                            ('operations',{f'{c}.json' for c in cases})):
        directory=out/folder
        if not directory.is_dir() or {p.name for p in directory.iterdir()}!=expected:
            raise RuntimeError('Unexpected, missing or partial retained '+folder)
        if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):
            raise RuntimeError('Retained cohort objects must be regular nonsymlink files')


def _publication_runtime_guard(manifest,out):
    """End-of-cohort source/dependency refusal, including the serial path."""
    if runtime_source_map()!=manifest['source_sha256']:
        raise RuntimeError('Publication source closure changed during cohort work')
    gate=json.loads((Path(out)/'ALL_POLICIES_VERIFIED.json').read_text())
    identities={d:{'versions':s['versions'],'dependency_source_sha256':s['dependency_source_sha256']}
                for d in sorted({c['domain'] for c in manifest['cases']}) for s in [fair_system_meta(d)]}
    if gate.get('all_closed_final_policies_verified')is not True or gate['manifest_sha256']!=digest(manifest) or gate['policy_count']!=len(manifest['policies']):
        raise RuntimeError('Publication requires the unchanged complete final-policy gate')
    if gate['evaluation_sources']!=manifest['source_sha256'] or gate['evaluation_runtime']!=identities:
        raise RuntimeError('Publication dependency runtime/source identity changed')


def publish_audit(manifest,out,reports,summaries):
    out=Path(out);mh=digest(manifest)
    _retained_membership(manifest,out)
    expected={c['case_id'] for c in manifest['cases']}
    if len(reports)!=len(expected) or {r['case_id'] for r in reports}!=expected:
        raise RuntimeError('Complete exact recorded-action audit cohort required')
    # Earlier per-case replay is not a claim that its retained inode survived
    # later workers. Repeat all raw identities after the whole audit closes.
    for summary in summaries:verify_case_retention(summary,out,structure=False)
    _retained_membership(manifest,out)
    _publication_runtime_guard(manifest,out)
    recomputed=aggregate(manifest,summaries)
    if canonical(recomputed)!=canonical(json.loads((out/'ANALYSIS.json').read_text())):
        raise RuntimeError('Recomputed complete aggregate differs')
    report={'schema':'PCI-FAIR-RETAINED-RECORD-AUDIT-v1','status':'PASS','manifest_sha256':mh,
        'policy_count':len(manifest['policies']),'episode_count':len(reports),
        'joint_steps':sum(r['joint_steps'] for r in reports),'agent_decisions':sum(r['agent_decisions'] for r in reports),
        'independent_numpy_actor':True,'independent_scalar_stage_checks':True,
        'recorded_action_native_replay':True,'replacement_policy_rollouts':0,
        'scope':manifest['audit_scope'],'case_reports':reports}
    closed_json(out/'RETAINED_RECORD_AUDIT.json',report);return report


def global_gate(manifest,training_root,out):
    """All policy bodies are verified once before any evaluation world."""
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);mh=digest(manifest)
    loaded=verify_manifest(manifest,training_root)
    # This gate precedes the first evaluation world for the entire cohort.
    closed_json(out/'MANIFEST.json',manifest)
    dependency={domain:fair_system_meta(domain) for domain in sorted({c['domain'] for c in manifest['cases']})}
    identities={k:{'versions':v['versions'],'dependency_source_sha256':v['dependency_source_sha256']} for k,v in dependency.items()}
    closed_json(out/'ALL_POLICIES_VERIFIED.json',{'schema':'PCI-FAIR-ALL-POLICIES-GATE-v1',
        'manifest_sha256':mh,'all_closed_final_policies_verified':True,'policy_count':len(loaded),
        'evaluation_sources':manifest['source_sha256'],'evaluation_runtime':identities})
    return loaded


def evaluate(manifest,training_root,out):
    out=Path(out);mh=digest(manifest)
    if (out/'EVALUATION_COMPLETE.json').exists():
        stored,loaded,summaries=audit_gate(training_root,out)
        if canonical(stored)!=canonical(manifest):raise RuntimeError('Closed cohort belongs to a different manifest')
        for summary in summaries:
            cfg,done,actor=loaded[summary['case']['policy_directory']]
            audit_case(summary,cfg,done,actor,out,mh)
        result=aggregate(manifest,summaries)
        if canonical(result)!=canonical(json.loads((out/'ANALYSIS.json').read_text())):
            raise RuntimeError('Closed complete aggregate differs')
        return result
    loaded=global_gate(manifest,training_root,out)
    summaries=[]
    for case in manifest['cases']:
        cfg,done,actor=loaded[case['policy_directory']]
        p=out/'cases'/f'{case["case_id"]}.json'
        if p.exists():
            summary=json.loads(p.read_text());audit_case(summary,cfg,done,actor,out,mh)
        else:
            if (out/'EVALUATION_COMPLETE.json').exists():
                raise RuntimeError('A sealed cohort cannot collect a replacement case')
            summary=evaluate_case(case,cfg,done,actor,out,mh)
        summaries.append(summary)
    return publish_completion(manifest,out,summaries)


def audit_gate(training_root,out):
    """Verify the complete cohort and immutable receipts before replay."""
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out);manifest=json.loads((out/'MANIFEST.json').read_text());mh=digest(manifest)
    loaded=verify_manifest(manifest,training_root)
    gate=json.loads((out/'ALL_POLICIES_VERIFIED.json').read_text())
    identities={d:{'versions':s['versions'],'dependency_source_sha256':s['dependency_source_sha256']}
                for d in sorted({c['domain'] for c in manifest['cases']}) for s in [fair_system_meta(d)]}
    if gate['manifest_sha256']!=mh or gate['evaluation_runtime']!=identities:raise RuntimeError('Audit runtime identity differs')
    complete=json.loads((out/'EVALUATION_COMPLETE.json').read_text())
    if complete.get('schema')!='PCI-FAIR-EVALUATION-COMPLETE-v1' or complete.get('status')!='COMPLETE':
        raise RuntimeError('Closed complete evaluation cohort required')
    if complete['manifest_sha256']!=mh or complete['analysis_sha256']!=sha(out/'ANALYSIS.json'):
        raise RuntimeError('Complete cohort identity mismatch')
    cases={c['case_id'] for c in manifest['cases']}
    if set(complete['case_receipts'])!=cases:raise RuntimeError('Incomplete case ledger')
    if set(complete.get('operational_measurement_receipts',{}))!=cases:
        raise RuntimeError('Incomplete operational measurement ledger')
    if set(complete.get('trace_intent_receipts',{}))!=cases:raise RuntimeError('Incomplete trace intent ledger')
    _retained_membership(manifest,out)
    summaries=[]
    for c in manifest['cases']:
        p=out/'cases'/f'{c["case_id"]}.json'
        if sha(p)!=complete['case_receipts'][c['case_id']]:raise RuntimeError('Retained case receipt changed')
        r=json.loads(p.read_text())
        if r['case']!=c:raise RuntimeError('Case descriptor differs')
        if complete['trace_intent_receipts'][c['case_id']]!=r['trace_seal'].get('publication_intent_sha256'):
            raise RuntimeError('Complete trace intent binding differs')
        if sha(out/_safe_relative(r['operational_measurements']))!=complete['operational_measurement_receipts'][c['case_id']]:
            raise RuntimeError('Operational measurement receipt changed')
        summaries.append(r)
    return manifest,loaded,summaries


def audit(training_root,out):
    out=Path(out);manifest,loaded,summaries=audit_gate(training_root,out);mh=digest(manifest)
    reports=[]
    for r in summaries:
        c=r['case']
        cfg,done,actor=loaded[c['policy_directory']]
        reports.append(audit_case(r,cfg,done,actor,out,mh))
    return publish_audit(manifest,out,reports,summaries)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('evaluate','audit'):
        q=sub.add_parser(name);q.add_argument('--training-root',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
        if name=='evaluate':q.add_argument('--manifest',type=Path,required=True)
    q=sub.add_parser('prepare');q.add_argument('--training-root',type=Path,required=True)
    q.add_argument('--policy-descriptors',type=Path,required=True);q.add_argument('--stage',choices=list(STAGE_BASE),required=True)
    q.add_argument('--worlds-per-profile',type=int,required=True);q.add_argument('--protocol-sha256');q.add_argument('--output',type=Path,required=True)
    q.add_argument('--training-case-manifest',type=Path)
    a=p.parse_args()
    if a.command=='prepare':
        result=prepare_manifest(a.training_root,json.loads(a.policy_descriptors.read_text()),stage=a.stage,
            worlds_per_profile=a.worlds_per_profile,protocol_sha256=a.protocol_sha256,
            training_case_manifest=json.loads(a.training_case_manifest.read_text()) if a.training_case_manifest else None);closed_json(a.output,result)
    elif a.command=='evaluate':result=evaluate(json.loads(a.manifest.read_text()),a.training_root,a.out)
    else:result=audit(a.training_root,a.out)
    print(json.dumps({k:v for k,v in result.items() if k in ('schema','status','policy_count','episode_count','joint_steps','agent_decisions')}))
