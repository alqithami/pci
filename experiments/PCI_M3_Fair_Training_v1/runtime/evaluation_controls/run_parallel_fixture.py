"""Engineering byte comparison using identical explicit serial/parallel inputs.

With --request, reuse frozen external models and optionally a retained serial
reference. Without it, create one zero-weight actor and eight H16 fixtures.
Never validates or executes the scientific cohort and never grants readiness.
"""
from pathlib import Path
import argparse, copy, json, math, os, sys, time


def main(project_root, output, request_path=None, serial_output=None, deadline_seconds=600):
    root, out = Path(project_root).resolve(), Path(output).resolve()
    if out.exists() or out.is_relative_to(root) or not Path(output).is_absolute():
        raise ValueError('NEW_ABSOLUTE_EXTERNAL_ENGINEERING_OUTPUT_REQUIRED')
    sys.path.insert(0,str(root))
    from evaluation.parallel import cpu_environment
    os.environ.update(cpu_environment())
    from evaluation.common import require, digest, sha, read_json, write_new_json
    require(math.isfinite(deadline_seconds) and deadline_seconds > 0, 'POSITIVE_FINITE_ENGINEERING_DEADLINE')
    from evaluation.engine import PROFILES, run_episode, checked_adapter, source_closure
    from evaluation.actor import CheckedTorchActor
    from evaluation.parallel import run_phase
    out.mkdir(mode=0o700)
    started=time.monotonic();deadline=started+deadline_seconds
    if request_path is None:
        import numpy as np
        import torch
        from evaluation.actor import SHAPES, VIEW_INDICES
        arrays={k:np.zeros(shape,dtype=np.float32) for k,shape in SHAPES.items()}
        for i,cols in enumerate(VIEW_INDICES):arrays['masks'][i,list(cols)]=1
        export=out/'fixture_export.npz';np.savez(export,**arrays)
        checkpoint=out/'fixture_checkpoint.pt'
        torch.save({'model':{k:torch.from_numpy(v.copy()) for k,v in arrays.items()}},checkpoint)
        key='engineering_zero_actor'
        bindings={key:dict(signature=digest({'role':'ENGINEERING_ZERO_WEIGHT'}),checkpoint_sha256=sha(checkpoint),portable_export_sha256=sha(export))}
        cfgs=copy.deepcopy(PROFILES)
        for cfg in cfgs.values():cfg.update(horizon=16,quota_window=4)
        cases=[]
        for i in range(8):
            profile=list(PROFILES)[i%4]
            cases.append(dict(case_id=f'engineering_parallel_{i:02d}_{profile}',training_case_id=key,
                method='task',training_seed=860000950,profile_id=profile,horizon=16,
                world_index=i,world_seed=860002000+i,policy_sampling_seed=860002100+i))
        request=dict(cases=cases,bindings=bindings,export_paths={key:str(export)},expected_configs=cfgs,
                     protocol_sha256=sha(root/'design/M2_PROTOCOL.json'))
    else:request=read_json(request_path)
    cases=request['cases'];bindings=request['bindings'];cfgs=request['expected_configs']
    paths=request.get('export_paths') or {k:b['export_path'] for k,b in bindings.items()}
    require(cases and all(c['case_id'].startswith('engineering_') for c in cases),'ENGINEERING_CASE_IDS_ONLY')
    require(all(all(type(c[k]) is int and 860000000 <= c[k] < 861000000
                for k in ('training_seed','world_seed','policy_sampling_seed')) for c in cases),'ENGINEERING_SEED_NAMESPACE_ONLY')
    require(all(2 <= c['horizon'] <= 256 and c['profile_id'] in cfgs for c in cases),'ENGINEERING_CONFIGS_REQUIRED')
    write_new_json(out/'EXPLICIT_INPUT_REQUEST.json',request)
    if serial_output is None:
        serial=out/'serial';serial.mkdir()
        actors={};adapter=checked_adapter(root)
        for c in cases:
            require(time.monotonic()<deadline,'ENGINEERING_COMPARISON_DEADLINE')
            key=c['training_case_id']
            if key not in actors:actors[key]=CheckedTorchActor(root,paths[key],c['method'])
            run_episode(c,cfgs[c['profile_id']],actors[key],adapter,serial,request['protocol_sha256'],bindings[key],fixture=True)
    else:
        serial=Path(serial_output).resolve()
        require(serial.is_dir(),'EXISTING_RETAINED_SERIAL_REFERENCE_REQUIRED')
    parallel=out/'parallel';parallel.mkdir()
    native=run_phase(root,parallel,cases,bindings,request['protocol_sha256'],'native',out/'native_operations',deadline,
        fixture=True,expected_configs=cfgs,export_paths=paths)
    audit=run_phase(root,parallel,cases,bindings,request['protocol_sha256'],'audit',out/'audit_operations',deadline,
        fixture=True,expected_configs=cfgs,export_paths=paths)
    compared=[];serial_summaries=[]
    for c in cases:
        names=(f"cases/{c['case_id']}.json",f"traces/{c['case_id']}.jsonl.gz")
        for name in names:
            require((serial/name).read_bytes()==(parallel/name).read_bytes(),'SERIAL_PARALLEL_BYTE_IDENTITY:'+name)
            compared.append(dict(path=name,bytes=(parallel/name).stat().st_size,sha256=sha(parallel/name)))
        serial_summaries.append(read_json(serial/names[0]))
    require(serial_summaries==native['summaries']==audit['summaries'],'ORDERED_SERIAL_PARALLEL_AUDIT_SUMMARY_IDENTITY')
    for phase in (native,audit):
        process=phase['receipt']['process']
        require(process['peak_parallel_workers']==4 and
                sum(x['seconds_lower_bound'] for x in process['actual_live_concurrency_intervals'])>0,
                'ACTUAL_FOUR_WORKER_OVERLAP_REQUIRED')
    result=dict(schema='PCI-M2-PARALLEL-ENGINEERING-BYTE-COMPARISON-v1',status='PASS',scientific=False,
        checked_cases=len(cases),checked_joint_steps=sum(c['horizon'] for c in cases),
        serial_output=str(serial),parallel_output=str(parallel),compared_files=compared,
        ordered_summaries_sha256=digest(serial_summaries),source_sha256=source_closure(root),
        native_phase=native['receipt'],independent_retained_audit_phase=audit['receipt'],
        exact_case_seeds_configs_models_bindings_reused=True,case_retries=0,
        elapsed_seconds=time.monotonic()-started,
        scope='Identical explicit engineering episodes in serial and four fresh CPU worker processes; exact retained trace/case byte equality and separate independent retained audit. No scientific cohort, GPU training, readiness authorization or timing extrapolation.')
    write_new_json(out/'PARALLEL_BYTE_COMPARISON.json',result)
    print(json.dumps({'status':'PASS','checked_cases':len(cases),'receipt':str(out/'PARALLEL_BYTE_COMPARISON.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--request',type=Path);p.add_argument('--serial-output',type=Path)
    p.add_argument('--deadline-seconds',type=float,default=600)
    a=p.parse_args();main(a.project_root,a.output,a.request,a.serial_output,a.deadline_seconds)
