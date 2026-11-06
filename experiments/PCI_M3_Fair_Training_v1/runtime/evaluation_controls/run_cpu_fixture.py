"""Small one-thread CPU engineering fixture; never a scientific evaluation.

Creates four H16/W4 worlds, including a scheduled reset/rule change at step8.
All model parameters except fixed masks are zero; no optimizer is run. The
portable actor, native/scalar dynamics, complete chains and retained-record
reconciliation are checked. Corrupt summary and truncated trace controls must
fail. This is intentionally not routed through the 60/1200 scientific gate.
"""
from pathlib import Path
import argparse, copy, gzip, hashlib, json, os, shutil, sys, time


def main(project_root, output):
    project_root, output = Path(project_root).resolve(), Path(output).resolve()
    if not output.is_relative_to(Path('/tmp').resolve()) or output.exists():
        raise ValueError('NEW_OWNED_TMP_FIXTURE_DIRECTORY_REQUIRED')
    sys.path.insert(0,str(project_root))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from evaluation.common import digest, sha, read_json, write_new_json
    from evaluation.actor import SHAPES, VIEW_INDICES, CheckedTorchActor, load_export
    from evaluation.engine import PROFILES, checked_adapter, run_episode, source_closure
    from evaluation.retained_audit import audit_case
    output.mkdir(mode=0o700)
    start=time.perf_counter()
    arrays={k:np.zeros(shape,dtype=np.float32) for k,shape in SHAPES.items()}
    for i,cols in enumerate(VIEW_INDICES):arrays['masks'][i,list(cols)]=1
    export=output/'fixture_export.npz'
    np.savez(export,**arrays)
    signature=digest({'role':'ENGINEERING_ONLY_ZERO_WEIGHT_ACTOR','arrays':{k:list(v.shape) for k,v in arrays.items()}})
    checkpoint=output/'fixture_checkpoint.pt'
    torch.save({'model':{k:torch.from_numpy(v.copy()) for k,v in arrays.items()},'signature':signature},checkpoint)
    binding=dict(signature=signature,checkpoint_sha256=sha(checkpoint),portable_export_sha256=sha(export),export_path=str(export))
    actor=CheckedTorchActor(project_root,export,'task')
    adapter=checked_adapter(project_root)
    protocol_sha=sha(project_root/'design/M2_PROTOCOL.json')
    cases=[]
    for i,(profile,cfg) in enumerate(PROFILES.items()):
        cfg=copy.deepcopy(cfg);cfg['horizon']=16;cfg['quota_window']=4
        case=dict(case_id=f'engineering_eval_{profile}',method='task',training_seed=860000950,
                  cluster_index=0,profile_id=profile,world_index=i,world_seed=860000960+i,
                  policy_sampling_seed=860000970+i,horizon=16)
        summary=run_episode(case,cfg,actor,adapter,output,protocol_sha,binding,fixture=True)
        audited=audit_case(output,case,binding,arrays,adapter,protocol_sha,fixture=True,expected_config=cfg)
        if audited!=summary:raise AssertionError('FIXTURE_SUMMARY_RECONCILIATION')
        cases.append(dict(case=case,config=cfg,summary_sha256=sha(output/'cases'/f"{case['case_id']}.json"),trace_sha256=summary['trace_sha256']))
    # Each negative control retains the authentic trace and identities as far as
    # possible, then changes one declared property. No outcome is replaced.
    negative=[]
    chosen=cases[0]
    for kind in ['summary_count','truncated_trace']:
        bad=output/'negative_controls'/kind
        (bad/'cases').mkdir(parents=True);(bad/'traces').mkdir()
        case=chosen['case'];src=output/'cases'/f"{case['case_id']}.json";s=read_json(src)
        target=bad/s['trace_path'];shutil.copy2(output/s['trace_path'],target)
        if kind=='summary_count':s['counts']['institutional_stage_agent_replacements']+=1
        else:
            with gzip.open(target,'rb') as f:head=f.readline()
            with target.open('wb') as raw:
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:f.write(head)
            s['trace_sha256']=sha(target);s['trace_bytes']=target.stat().st_size
        write_new_json(bad/'cases'/f"{case['case_id']}.json",s)
        try:
            audit_case(bad,case,binding,arrays,adapter,protocol_sha,fixture=True,expected_config=chosen['config'])
        except ValueError as exc:
            expected='CASE_SUMMARY_RECONCILIATION' if kind=='summary_count' else 'MISSING_STEP_RECORD'
            if expected not in str(exc):raise
            negative.append(dict(control=kind,status='PASS_REJECTED',reason=str(exc)))
        else:raise AssertionError('CORRUPT_RETAINED_RECORD_ACCEPTED:'+kind)
    report=dict(schema='PCI-M2-EVALUATION-CPU-ENGINEERING-FIXTURE-v1',status='PASS',scientific=False,
        frozen_protocol_sha256=protocol_sha,source_sha256=source_closure(project_root),cases=cases,
        negative_controls=negative,CPU_threads=torch.get_num_threads(),Torch_version=torch.__version__,NumPy_version=np.__version__,
        authoritative_forward_states=actor.checked_states,forward_error_max=dict(logits=actor.max_logit_abs_error,probabilities=actor.max_probability_abs_error),
        writer_native_world_constructors=4,writer_native_task_transition_calls=64,writer_policy_seeded_streams=4,
        retained_audit_native_world_constructors=0,retained_audit_native_task_transition_calls=0,
        retained_audit_world_initialization_RNG_streams_replayed=4,retained_audit_policy_RNG_streams_replayed=4,
        scientific_training_episodes_or_new_scientific_evaluation_cases=0,optimizer_GPU_cloud_calls=0,
        elapsed_seconds=time.perf_counter()-start,scope='Numerical/end-to-end engineering validation on a zero-weight CPU actor and four explicitly shortened fixture worlds. This is not a fresh-training study, heldout evaluation, timing forecast for fullH256, CUDA validation or efficacy evidence.')
    # Negative controls replay one retained fixture twice more; count them explicitly.
    report['negative_control_additional_policy_RNG_replays']=2
    report['negative_control_additional_initialization_RNG_replays']=1
    write_new_json(output/'EVALUATION_CPU_FIXTURE.json',report)
    print(json.dumps({'status':'PASS','receipt':str(output/'EVALUATION_CPU_FIXTURE.json'),'sha256':sha(output/'EVALUATION_CPU_FIXTURE.json'),'writer_native_task_steps':64,'elapsed_seconds':report['elapsed_seconds']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.project_root,a.output)
