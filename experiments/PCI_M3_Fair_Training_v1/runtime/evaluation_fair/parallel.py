"""Bounded policy-block collection and retained-action audit, at most 4 workers.

The parent verifies the complete final-policy cohort before spawning any world.
Each spawned worker then verifies only its own policy body and the immutable
full-manifest/global-gate membership, and writes disjoint case paths. Completion
and audit receipts are published only by the parent after all workers close.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import contextmanager
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import torch
from training.pci_risk_train.fair_trainer import runtime_sources
from .actor import verify_training_policy
from .common import canonical,closed_json,digest,runtime_source_map,sha
from .engine import (global_gate,audit_gate,evaluate_case,audit_case,
                     publish_completion,publish_audit,verify_case_retention,_safe_relative)


@contextmanager
def execution_lock(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    f=(out/'EVALUATION_RUN.lock').open('a+')
    try:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:f.close()


def _workers(value):
    if type(value)is not int or not 1<=value<=4:raise ValueError('Exactly 1..4 evaluation workers required')
    return value


def _load_worker(job):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(job['out'])
    if sha(out/'MANIFEST.json')!=job['manifest_file_sha256'] or sha(out/'ALL_POLICIES_VERIFIED.json')!=job['gate_file_sha256']:
        raise RuntimeError('Worker full manifest/global policy gate identity changed')
    manifest=json.loads((out/'MANIFEST.json').read_text())
    if digest(manifest)!=job['manifest_sha256'] or runtime_source_map()!=manifest['source_sha256']:
        raise RuntimeError('Worker executing source closure changed')
    gate=json.loads((out/'ALL_POLICIES_VERIFIED.json').read_text())
    if gate.get('all_closed_final_policies_verified')is not True or gate['manifest_sha256']!=job['manifest_sha256'] or gate['policy_count']!=len(manifest['policies']):
        raise RuntimeError('Worker lacks a complete global final-policy gate')
    selected=[r for r in manifest['policies'] if r['directory']==job['directory']]
    if len(selected)!=1:raise RuntimeError('Worker policy is not a unique full-manifest member')
    b=selected[0];root=Path(job['training_root'])/_safe_relative(b['directory'])
    cfg,done,actor=verify_training_policy(root,current_sources=runtime_sources())
    for key,actual in {'train_complete_sha256':sha(root/'TRAIN_COMPLETE.json'),
            'config_sha256':sha(root/'config.json'),'closed_manifest_sha256':done['closed_manifest_sha256'],
            'portable_export_sha256':done['portable_export_sha256'],'training_signature':done['signature']}.items():
        if b[key]!=actual:raise RuntimeError('Worker policy binding mismatch: '+key)
    if cfg['run_id']!=b['run_id'] or cfg['seed']!=b['training_seed'] or done['model']!=b['model_specification']:
        raise RuntimeError('Worker model/cluster descriptor mismatch')
    if manifest['stage']!='engineering' and cfg.get('protocol_sha256')!=manifest['protocol_sha256']:
        raise RuntimeError('Worker scientific protocol mismatch')
    cases=[c for c in manifest['cases'] if c['policy_directory']==b['directory']]
    if not cases or [c['case_id'] for c in cases]!=job['case_ids']:
        raise RuntimeError('Worker cases do not exactly match the full cohort policy block')
    if job.get('engineering_exit'):
        if manifest['stage']!='engineering':raise RuntimeError('Crash injection is restricted to isolated engineering')
        os._exit(97)
    return out,manifest,cfg,done,actor,cases


def _worker(job):
    out,manifest,cfg,done,actor,cases=_load_worker(job)
    start=time.perf_counter();summaries=[];reports=[];mh=job['manifest_sha256']
    complete=None
    if job['mode']=='audit':
        if sha(out/'EVALUATION_COMPLETE.json')!=job['complete_file_sha256']:
            raise RuntimeError('Worker complete-cohort receipt changed')
        complete=json.loads((out/'EVALUATION_COMPLETE.json').read_text())
    for c in cases:
        p=out/'cases'/f'{c["case_id"]}.json'
        if job['mode']=='evaluate' and not p.exists():
            if not job['allow_new_cases'] or (out/'EVALUATION_COMPLETE.json').exists():
                raise RuntimeError('A sealed cohort cannot collect a replacement case')
            summary=evaluate_case(c,cfg,done,actor,out,mh)
        else:
            summary=json.loads(p.read_text())
            if summary['case']!=c:raise RuntimeError('Worker retained case differs from full manifest')
            if complete is not None:
                if sha(p)!=complete['case_receipts'][c['case_id']]:
                    raise RuntimeError('Worker case receipt differs from complete cohort')
                if sha(out/_safe_relative(summary['operational_measurements']))!=complete['operational_measurement_receipts'][c['case_id']]:
                    raise RuntimeError('Worker operational receipt differs from complete cohort')
            report=audit_case(summary,cfg,done,actor,out,mh)
            if job['mode']=='audit':reports.append(report)
        summaries.append(summary)
    if runtime_source_map()!=manifest['source_sha256']:raise RuntimeError('Worker source changed during its block')
    block={'schema':'PCI-FAIR-POLICY-BLOCK-CLOSED-v1','status':'CLOSED',
        'mode':job['mode'],'manifest_sha256':mh,'policy_directory':job['directory'],
        'case_ids':job['case_ids'],'all_block_cases_closed':True,
        'case_receipts':{c['case_id']:sha(out/'cases'/f'{c["case_id"]}.json') for c in cases},
        'scope':'One disjoint policy block; not a complete scientific cohort.'}
    name=f'{cfg["run_id"]}.{job["mode"]}.json'
    closed_json(out/'worker_receipts'/name,block)
    return {'directory':job['directory'],'summaries':summaries,'reports':reports,
            'worker_receipt':f'worker_receipts/{name}',
            'worker_elapsed_seconds':time.perf_counter()-start}


def _jobs(manifest,training_root,out,mode,*,engineering_fail_directory=None):
    out=Path(out);mh=digest(manifest)
    if engineering_fail_directory is not None:
        if manifest['stage']!='engineering':raise ValueError('Fault injection is engineering only')
        if engineering_fail_directory not in {b['directory'] for b in manifest['policies']}:
            raise ValueError('Unknown engineering fault-injection policy')
    jobs=[]
    for b in manifest['policies']:
        job={'mode':mode,'directory':b['directory'],'training_root':str(Path(training_root).resolve()),
            'out':str(out.resolve()),'manifest_sha256':mh,
            'manifest_file_sha256':sha(out/'MANIFEST.json'),'gate_file_sha256':sha(out/'ALL_POLICIES_VERIFIED.json'),
            'case_ids':[c['case_id'] for c in manifest['cases'] if c['policy_directory']==b['directory']],
            'allow_new_cases':mode=='evaluate' and not (out/'EVALUATION_COMPLETE.json').exists(),
            'engineering_exit':b['directory']==engineering_fail_directory}
        if mode=='audit':job['complete_file_sha256']=sha(out/'EVALUATION_COMPLETE.json')
        jobs.append(job)
    return jobs


def _retention_worker(job):
    """Post-producer full gzip/chain readback; never load or sample a policy."""
    out=Path(job['out'])
    if sha(out/'MANIFEST.json')!=job['manifest_file_sha256'] or sha(out/'ALL_POLICIES_VERIFIED.json')!=job['gate_file_sha256']:
        raise RuntimeError('Retained phase global manifest/gate identity changed')
    manifest=json.loads((out/'MANIFEST.json').read_text())
    if digest(manifest)!=job['manifest_sha256'] or runtime_source_map()!=manifest['source_sha256']:
        raise RuntimeError('Retained phase source or manifest changed')
    gate=json.loads((out/'ALL_POLICIES_VERIFIED.json').read_text())
    if gate.get('all_closed_final_policies_verified')is not True or gate['manifest_sha256']!=job['manifest_sha256'] or gate['policy_count']!=len(manifest['policies']):
        raise RuntimeError('Retained phase lacks complete policy gate')
    cases=[c for c in manifest['cases'] if c['policy_directory']==job['directory']]
    if [c['case_id'] for c in cases]!=job['case_ids']:raise RuntimeError('Retained phase block membership differs')
    reports=[]
    for c in cases:
        p=out/'cases'/f'{c["case_id"]}.json'
        if sha(p)!=job['case_receipts'][c['case_id']]:raise RuntimeError('Post-worker case bytes changed')
        summary=json.loads(p.read_text())
        if summary['case']!=c:raise RuntimeError('Post-worker case manifest differs')
        reports.append(verify_case_retention(summary,out,structure=True))
    if runtime_source_map()!=manifest['source_sha256']:raise RuntimeError('Retained phase source changed during readback')
    return {'directory':job['directory'],'case_ids':job['case_ids'],'retention_reports':reports}


def _run_jobs(jobs,max_workers,out,mode,*,worker=_worker):
    results=[];start=time.perf_counter()
    try:
        # Spawn avoids inheriting Torch/OpenMP state from the parent validator.
        with ProcessPoolExecutor(max_workers=_workers(max_workers),mp_context=mp.get_context('spawn')) as pool:
            pending={pool.submit(worker,j):j['directory'] for j in jobs}
            try:
                for future in as_completed(pending):results.append(future.result())
            except BaseException:
                for future in pending:future.cancel()
                raise
    except BaseException as e:
        # No successful global receipt is written here or by a worker.
        closed_json(Path(out)/f'PARALLEL_{mode.upper()}_FAILURE.json',{
            'schema':'PCI-FAIR-PARALLEL-FAILURE-v1','status':'FAILED_NO_COMPLETE_COHORT',
            'mode':mode,'error':repr(e),'closed_policy_blocks_observed':len(results)})
        raise
    if len(results)!=len(jobs) or {r['directory'] for r in results}!={j['directory'] for j in jobs}:
        raise RuntimeError('A worker policy block is missing')
    return results,time.perf_counter()-start


def _execution_receipt(out,name,value):
    """Retain the first execution and append measured retries separately."""
    p=Path(out)/name
    if not p.exists():closed_json(p,value)
    else:
        prior=json.loads(p.read_text())
        if prior['manifest_sha256']!=value['manifest_sha256']:
            raise RuntimeError('Execution retry belongs to a different manifest')
        closed_json(Path(out)/'execution_attempts'/f'{name}.{time.time_ns()}.{os.getpid()}.json',value)


def evaluate(manifest,training_root,out,*,max_workers=4,_engineering_fail_directory=None):
    _workers(max_workers);out=Path(out)
    with execution_lock(out):
        closed=(out/'EVALUATION_COMPLETE.json').exists()
        if closed:
            stored,loaded,_=audit_gate(training_root,out)
            if canonical(stored)!=canonical(manifest):raise RuntimeError('Closed cohort belongs to a different manifest')
        else:loaded=global_gate(manifest,training_root,out)
        # Parent verification is global; worker raw-file hashing is per-policy.
        del loaded
        jobs=_jobs(manifest,training_root,out,'audit' if closed else 'evaluate',
                   engineering_fail_directory=_engineering_fail_directory)
        blocks,elapsed=_run_jobs(jobs,max_workers,out,'evaluate')
        index={r['case']['case_id']:r for b in blocks for r in b['summaries']}
        expected=[c['case_id'] for c in manifest['cases']]
        if len(index)!=len(expected) or set(index)!=set(expected):
            raise RuntimeError('Complete exact case cohort required after worker closure')
        summaries=[index[k] for k in expected]
        # All producers are now joined. A separate read-only phase checks every
        # gzip EOF, complete record chain and header/footer before any analysis.
        retained_jobs=[]
        for job in jobs:
            retained=dict(job)
            retained['case_receipts']={k:sha(out/'cases'/f'{k}.json') for k in job['case_ids']}
            retained_jobs.append(retained)
        retention_blocks,retention_elapsed=_run_jobs(retained_jobs,max_workers,out,'retention',worker=_retention_worker)
        retained_ids=[r['case_id'] for b in retention_blocks for r in b['retention_reports']]
        if len(retained_ids)!=len(expected) or set(retained_ids)!=set(expected):raise RuntimeError('Incomplete post-worker retained trace readback')
        if closed:
            reports={r['case_id']:r for b in blocks for r in b['reports']}
            if set(reports)!=set(expected):raise RuntimeError('Closed resume requires every recorded-action audit')
            publish_audit(manifest,out,[reports[k] for k in expected],summaries)
        result=publish_completion(manifest,out,summaries,retained_structure_verified=True)
        _execution_receipt(out,'PARALLEL_EVALUATION_EXECUTION.json',{
            'schema':'PCI-FAIR-PARALLEL-EXECUTION-v1','status':'ALL_WORKERS_CLOSED',
            'manifest_sha256':digest(manifest),'max_workers':max_workers,'policy_blocks':len(blocks),
            'execution_kind':'retained_action_audit_resume' if closed else 'collection',
            'elapsed_seconds':elapsed,'post_worker_retention_seconds':retention_elapsed,
            'worker_receipts':{b['directory']:sha(out/b['worker_receipt']) for b in blocks},
            'worker_elapsed_seconds':{b['directory']:b['worker_elapsed_seconds'] for b in blocks}})
        return result


def audit(training_root,out,*,max_workers=4):
    _workers(max_workers);out=Path(out)
    with execution_lock(out):
        manifest,loaded,all_summaries=audit_gate(training_root,out);del loaded
        jobs=_jobs(manifest,training_root,out,'audit')
        blocks,elapsed=_run_jobs(jobs,max_workers,out,'audit')
        index={r['case_id']:r for b in blocks for r in b['reports']}
        expected=[c['case_id'] for c in manifest['cases']]
        if len(index)!=len(expected) or set(index)!=set(expected):raise RuntimeError('Missing audited worker/case block')
        reports=[index[k] for k in expected]
        result=publish_audit(manifest,out,reports,all_summaries)
        _execution_receipt(out,'PARALLEL_AUDIT_EXECUTION.json',{
            'schema':'PCI-FAIR-PARALLEL-EXECUTION-v1','status':'ALL_WORKERS_CLOSED',
            'manifest_sha256':digest(manifest),'max_workers':max_workers,'policy_blocks':len(blocks),
            'elapsed_seconds':elapsed,'worker_receipts':{b['directory']:sha(out/b['worker_receipt']) for b in blocks},
            'worker_elapsed_seconds':{b['directory']:b['worker_elapsed_seconds'] for b in blocks}})
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for mode in ('evaluate','audit'):
        q=sub.add_parser(mode);q.add_argument('--training-root',type=Path,required=True)
        q.add_argument('--out',type=Path,required=True);q.add_argument('--max-workers',type=int,default=4)
        if mode=='evaluate':q.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args()
    if a.command=='evaluate':r=evaluate(json.loads(a.manifest.read_text()),a.training_root,a.out,max_workers=a.max_workers)
    else:r=audit(a.training_root,a.out,max_workers=a.max_workers)
    print(json.dumps({k:v for k,v in r.items() if k in ('schema','status','policy_count','episode_count','joint_steps','agent_decisions')}))
