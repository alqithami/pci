"""Four fixed fresh-Python CPU workers; arithmetic stays in engine/audit_case.

Operational manifests, logs and statuses live outside the closed primary tree.
Workers inherit the evaluator's session/process group, so its controller-owned
session includes every case worker. No process pool, respawn, retry or CUDA fork.
"""
from __future__ import annotations
import argparse, json, math, os, signal, subprocess, sys, time
from pathlib import Path
from .common import require, digest, sha, read_json, write_new_json

MAX_WORKERS = 4

class ParallelFailure(ValueError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('FIXED_WORKER_PHASE_FAILED:' + receipt.get('error', 'UNKNOWN'))


def write_operation_json(path, value):
    """Publish only fsynced complete operational records; never primary files."""
    path = Path(path)
    require(not path.exists(), 'NO_OPERATIONAL_RECORD_REPLACEMENT')
    temporary = path.with_name(path.name+'.writing')
    write_new_json(temporary,value)
    os.replace(temporary,path)
    fd = os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def cpu_environment():
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        env[key] = '1'
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
    return env


def run_processes(jobs, cwd, operations_root, deadline):
    """Start each fixed job once; join all exact child PIDs on every exit path."""
    require(1 <= len(jobs) <= MAX_WORKERS, 'FIXED_WORKER_CEILING')
    require(len({j['worker_id'] for j in jobs}) == len(jobs), 'UNIQUE_FIXED_WORKERS')
    case_ids = [case_id for j in jobs for case_id in j['case_ids']]
    require(len(set(case_ids)) == len(case_ids), 'NO_DUPLICATE_WORKER_CASE_ASSIGNMENTS')
    require(math.isfinite(deadline), 'FINITE_FIXED_WORKER_DEADLINE')
    out = Path(operations_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = [dict(worker_id=j['worker_id'], argv=j['argv'], case_ids=j['case_ids'], status='UNRUN') for j in jobs]
    processes, streams, intervals, samples = [], [], [], []
    started = time.monotonic()
    def cleanup():
        actions = []
        for p in processes:
            if p.poll() is None:
                try: p.terminate(); actions.append(dict(pid=p.pid, signal='SIGTERM'))
                except ProcessLookupError: pass
        grace = time.monotonic()+2
        while any(p.poll() is None for p in processes) and time.monotonic() < grace:
            time.sleep(.02)
        for p in processes:
            if p.poll() is None:
                try: p.kill(); actions.append(dict(pid=p.pid, signal='SIGKILL'))
                except ProcessLookupError: pass
        for p in processes: p.wait(timeout=5)
        return actions
    try:
        for j, row in zip(jobs, rows):
            require(time.monotonic() < deadline, 'FIXED_WORKER_DEADLINE')
            stdout = (out/('worker_'+j['worker_id']+'.stdout.log')).open('xb')
            streams.append(stdout)
            stderr = (out/('worker_'+j['worker_id']+'.stderr.log')).open('xb')
            streams.append(stderr)
            # Deliberately inherit PGID/SID: controller owns the parent session.
            p = subprocess.Popen(j['argv'], cwd=cwd, env=cpu_environment(),
                stdout=stdout, stderr=stderr, start_new_session=False)
            processes.append(p)
            row.update(status='RUNNING', pid=p.pid, pgid=os.getpgid(p.pid), sid=os.getsid(p.pid),
                       start_monotonic=time.monotonic())
            require(row['pgid'] == os.getpgrp() and row['sid'] == os.getsid(0), 'WORKER_MUST_INHERIT_PARENT_SESSION')
        while any(p.poll() is None for p in processes):
            require(time.monotonic() < deadline, 'FIXED_WORKER_DEADLINE')
            live = tuple(sorted(p.pid for p in processes if p.poll() is None))
            now = time.monotonic()
            if len(live) == len(jobs):
                if samples and samples[-1][0] == live:
                    elapsed = now-samples[-1][1]
                    if elapsed > 0: intervals.append(dict(pids=list(live), seconds_lower_bound=elapsed))
                samples.append((live, now))
            else: samples.clear()
            for p, row in zip(processes, rows):
                code = p.poll()
                if code is not None and row['status'] == 'RUNNING':
                    row.update(exit_code=code, end_monotonic=time.monotonic(), status='COMPLETE' if code == 0 else 'FAILED')
                    require(code == 0, 'FIXED_WORKER_EXIT:'+row['worker_id']+':'+str(code))
            time.sleep(.02)
        for p, row in zip(processes, rows):
            code = p.wait()
            if row['status'] == 'RUNNING':
                row.update(exit_code=code, end_monotonic=time.monotonic(), status='COMPLETE' if code == 0 else 'FAILED')
            require(code == 0, 'FIXED_WORKER_EXIT:'+row['worker_id']+':'+str(code))
        result = dict(status='PASS', rows=rows, peak_parallel_workers=len(processes),
            actual_live_concurrency_intervals=intervals, owned_cleanup_actions=[],
            all_owned_processes_closed=True, elapsed_seconds=time.monotonic()-started)
        write_operation_json(out/'PROCESS_COMPLETE.json', result)
        return result
    except BaseException as exc:
        actions, cleanup_error = [], None
        try: actions = cleanup()
        except BaseException as cleanup_exc: cleanup_error = repr(cleanup_exc)
        for p, row in zip(processes, rows):
            if row['status'] == 'RUNNING':
                row.update(status='STOPPED_AFTER_PHASE_FAILURE' if p.poll() is not None else 'STOP_UNVERIFIED',
                           exit_code=p.poll(), end_monotonic=time.monotonic())
        result = dict(status='FAIL', error=repr(exc), rows=rows,
            actual_live_concurrency_intervals=intervals, peak_parallel_workers=len(processes),
            owned_cleanup_actions=actions, all_owned_processes_closed=all(p.poll() is not None for p in processes),
            cleanup_error=cleanup_error, elapsed_seconds=time.monotonic()-started)
        write_operation_json(out/'PROCESS_FAILED.json', result)
        raise ParallelFailure(result) from exc
    finally:
        for stream in streams:
            if not stream.closed:
                stream.flush(); os.fsync(stream.fileno()); stream.close()


def phase_ledger(output, operations_root, cases, operation):
    rows = []
    for i, case in enumerate(cases):
        stem = Path(operations_root)/'case_status'/f'{i:06d}'
        complete, failed, begun = (Path(str(stem)+'.'+kind+'.json') for kind in ('complete','failed','started'))
        primary = Path(output)/'cases'/(case['case_id']+'.json')
        trace = Path(output)/'traces'/(case['case_id']+'.jsonl.gz')
        partial = Path(str(trace)+'.partial')
        closed, confirmed, errors = False, False, []
        if primary.exists():
            try:
                summary = read_json(primary)
                require(summary['case'] == case and summary['status'] == 'PASS', 'LEDGER_PRIMARY_CASE_IDENTITY')
                require(trace.is_file() and not partial.exists() and sha(trace) == summary['trace_sha256']
                        and trace.stat().st_size == summary['trace_bytes'], 'LEDGER_CLOSED_TRACE_IDENTITY')
                closed = True
                if complete.exists():
                    done = read_json(complete)
                    require(done['case_id'] == case['case_id'] and done['summary_sha256'] == digest(summary), 'LEDGER_COMPLETION_IDENTITY')
                    confirmed = True
            except Exception as exc: errors.append(repr(exc))
        elif complete.exists(): errors.append('COMPLETION_WITHOUT_PRIMARY_CASE')
        status = ('COMPLETE_CLOSED' if confirmed else 'FAILED_CURRENT_CASE' if failed.exists()
                  else 'STOPPED_STARTED_CASE' if begun.exists() else 'UNRUN_AFTER_PHASE_FAILURE')
        if operation == 'native' and closed and not confirmed:
            status = 'CLOSED_CASE_OPERATIONAL_COMPLETION_UNCONFIRMED'
        if primary.exists() and not closed:
            status = 'INVALID_OR_PARTIAL_CASE_FILE'
        row = dict(case_id=case['case_id'], status=status, started=begun.exists(),
                   closed_case=closed, case_file_exists=primary.exists(), closed_trace=trace.exists(),
                   partial_trace=partial.exists(), validation_errors=errors)
        rows.append(row)
    return rows


def validate_model_locator(model, case, binding, export_path):
    """Recheck actual retained bodies and their stored signature/method identity."""
    require(model['training_case_id'] == case['training_case_id'] and
            model['method'] == case['method'] and model['training_seed'] == case['training_seed'],
            'AUDIT_LOCATOR_CASE_IDENTITY')
    require(model['export_path'] == export_path and model['portable_export_sha256'] == binding['portable_export_sha256']
            and model['signature'] == binding['signature'], 'AUDIT_LOCATOR_BOUND_IDENTITY')
    for field, body in (('checkpoint','checkpoint.pt'),('config','config.json'),('train_complete','TRAIN_COMPLETE.json')):
        actual, expected = model[field+'_path'], model[field+'_sha256']
        require(type(actual) is str and Path(actual).is_absolute(), 'ABSOLUTE_AUDIT_LOCATOR_BODY')
        require(expected == binding.get(field+'_sha256',expected) and sha(actual) == expected, 'AUDIT_LOCATOR_BODY_IDENTITY:'+body)
    config, done = read_json(model['config_path']), read_json(model['train_complete_path'])
    require(config['method'] == done['method'] == case['method'] and
            config['seed'] == done['seed'] == case['training_seed'], 'AUDIT_LOCATOR_METHOD_AND_SEED')
    require(done['signature'] == binding['signature'], 'AUDIT_LOCATOR_MODEL_SIGNATURE')
    require(sha(export_path) == binding['portable_export_sha256'], 'AUDIT_LOCATOR_EXPORT_IDENTITY')


def run_phase(project_root, output_root, cases, bindings, protocol_sha256, operation,
              operations_root, deadline, *, fixture=False, expected_configs=None, workers=4, export_paths=None, expected_source_sha256=None, validated_models=None):
    """Explicit immutable inputs; ordered results despite static worker sharding."""
    from .engine import source_closure
    root, output, ops = map(lambda p: Path(p).resolve(), (project_root, output_root, operations_root))
    require(operation in ('native','audit'), 'FIXED_WORKER_OPERATION')
    require(type(workers) is int and 1 <= workers <= MAX_WORKERS, 'FIXED_WORKER_CEILING')
    require(cases and len({c['case_id'] for c in cases}) == len(cases), 'UNIQUE_PHASE_CASES')
    require(not ops.is_relative_to(output) and not output.is_relative_to(ops), 'OPERATIONAL_FILES_OUTSIDE_PRIMARY_TREE')
    require(not ops.is_relative_to(root), 'OPERATIONAL_FILES_OUTSIDE_SOURCE_TREE')
    ops.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        source = source_closure(root)
        require(expected_source_sha256 is None or source == expected_source_sha256, 'PARENT_FROZEN_PHASE_SOURCE_IDENTITY')
        paths = {} if export_paths is None else dict(export_paths)
        for c in cases:
            key = c['training_case_id']; b = bindings[key]
            path = paths.setdefault(key,b.get('export_path'))
            require(type(path) is str and Path(path).is_absolute(), 'ABSOLUTE_WORKER_EXPORT_PATH_REQUIRED')
            if validated_models is not None:
                require(operation == 'audit', 'VALIDATED_RELOCATION_AUDIT_ONLY')
                require(fixture or {'checkpoint_sha256','config_sha256','train_complete_sha256'} <= set(b),
                        'COMPLETE_SCIENTIFIC_RELOCATION_BINDING_REQUIRED')
                validate_model_locator(validated_models[key],c,b,path)
            else:
                require('export_path' not in b or b['export_path'] == path, 'BINDING_EXPORT_PATH_MATCH')
            require(sha(path) == b['portable_export_sha256'], 'PHASE_MODEL_EXPORT_IDENTITY')
        request = dict(schema='PCI-M2-FIXED-CPU-WORKER-REQUEST-v1', project_root=str(root),
            output_root=str(output), operation=operation, cases=cases, bindings=bindings, export_paths=paths,validated_models=validated_models,
            protocol_sha256=protocol_sha256, fixture=fixture, expected_configs=expected_configs,
            source_sha256=source, workers=workers, deadline_monotonic=deadline)
        write_operation_json(ops/'REQUEST.json', request)
        (ops/'case_status').mkdir()
        jobs = [dict(worker_id=f'{i:02d}', case_ids=[c['case_id'] for c in cases[i::workers]],
            argv=[sys.executable,'-B','-m','evaluation.parallel','--worker-request',str(ops/'REQUEST.json'),
                  '--worker-index',str(i)]) for i in range(workers)]
        process = run_processes(jobs, root, ops/'processes', deadline)
        statuses = phase_ledger(output, ops, cases, operation)
        require(all(r['status'] == 'COMPLETE_CLOSED' for r in statuses), 'ALL_FIXED_CASES_COMPLETED_ONCE')
        require(source_closure(root) == source, 'PHASE_SOURCE_CHANGED')
        summaries = []
        for i, c in enumerate(cases):
            summary = read_json(output/'cases'/(c['case_id']+'.json'))
            complete = read_json(ops/'case_status'/f'{i:06d}.complete.json')
            require(summary['case'] == c and digest(summary) == complete['summary_sha256'], 'ORDERED_WORKER_RESULT_IDENTITY')
            summaries.append(summary)
        result = dict(schema='PCI-M2-FIXED-CPU-PHASE-v1', status='PASS', operation=operation,
            workers=workers, case_statuses=statuses, process=process,
            elapsed_seconds=time.monotonic()-started, primary_results_order=[c['case_id'] for c in cases],
            source_sha256=source, scientific=not fixture, case_retry_count=0)
        write_operation_json(ops/'PHASE_COMPLETE.json', result)
        return dict(summaries=summaries, receipt=result)
    except BaseException as exc:
        result = dict(schema='PCI-M2-FIXED-CPU-PHASE-FAILURE-v1', status='FAIL', operation=operation,
            error=repr(exc), case_statuses=phase_ledger(output, ops, cases, operation),
            failed_cases_not_replaced=True, case_retry_count=0, elapsed_seconds=time.monotonic()-started)
        write_operation_json(ops/'PHASE_FAILED.json', result)
        raise ParallelFailure(result) from exc


def worker(request_path, worker_index):
    """Internal fixed worker. One actor/export cache, no helper subprocesses."""
    request_path = Path(request_path).resolve()
    request = read_json(request_path)
    from .engine import source_closure, checked_adapter, run_episode, PROFILES
    from .actor import CheckedTorchActor, load_export
    from .retained_audit import audit_case
    root, output = Path(request['project_root']), Path(request['output_root'])
    require(Path(__file__).resolve() == root/'evaluation/parallel.py', 'FIXED_WORKER_IMPORT_PATH')
    require(source_closure(root) == request['source_sha256'], 'WORKER_SOURCE_IDENTITY')
    require(type(worker_index) is int and 0 <= worker_index < request['workers'] <= MAX_WORKERS, 'FIXED_WORKER_INDEX')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')), 'WORKER_ONE_CPU_THREAD')
    if request['operation'] == 'native':
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
    adapter = checked_adapter(root)
    actors, arrays = {}, {}
    for i in range(worker_index, len(request['cases']), request['workers']):
        case = request['cases'][i]
        status = request_path.parent/'case_status'/f'{i:06d}'
        require(time.monotonic() < request['deadline_monotonic'], 'FIXED_WORKER_DEADLINE')
        write_operation_json(Path(str(status)+'.started.json'), dict(case_id=case['case_id'],pid=os.getpid(),operation=request['operation']))
        try:
            key = case['training_case_id']; binding = request['bindings'][key]
            export_path = request['export_paths'][key]
            if request.get('validated_models') is not None:
                require(request['operation'] == 'audit', 'VALIDATED_RELOCATION_AUDIT_ONLY')
                require(request['fixture'] or {'checkpoint_sha256','config_sha256','train_complete_sha256'} <= set(binding),
                        'COMPLETE_SCIENTIFIC_RELOCATION_BINDING_REQUIRED')
                validate_model_locator(request['validated_models'][key],case,binding,export_path)
            require(sha(export_path) == binding['portable_export_sha256'], 'WORKER_EXPORT_IDENTITY')
            cfg = (request['expected_configs'] or PROFILES)[case['profile_id']]
            if request['operation'] == 'native':
                if key not in actors: actors[key] = CheckedTorchActor(root,export_path,case['method'])
                summary = run_episode(case,cfg,actors[key],adapter,output,request['protocol_sha256'],binding,fixture=request['fixture'])
            else:
                if key not in arrays: arrays[key] = load_export(export_path)
                summary = audit_case(output,case,binding,arrays[key],adapter,request['protocol_sha256'],
                    fixture=request['fixture'],expected_config=cfg if request['fixture'] else None)
            write_operation_json(Path(str(status)+'.complete.json'), dict(case_id=case['case_id'],pid=os.getpid(),summary_sha256=digest(summary)))
        except BaseException as exc:
            write_operation_json(Path(str(status)+'.failed.json'),dict(case_id=case['case_id'],pid=os.getpid(),error=repr(exc)))
            raise
    require(source_closure(root) == request['source_sha256'], 'WORKER_SOURCE_CHANGED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-request',type=Path,required=True)
    parser.add_argument('--worker-index',type=int,required=True)
    args = parser.parse_args()
    worker(args.worker_request,args.worker_index)

if __name__ == '__main__': main()
