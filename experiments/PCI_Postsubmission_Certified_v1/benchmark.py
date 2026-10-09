#!/usr/bin/env python3
"""Post-submission research. Preserves frozen studies; never invents completion."""
from __future__ import annotations
import argparse, base64, copy, csv, functools, gzip, hashlib, itertools, json
import math, os, platform, random, re, shutil, subprocess, sys, tarfile, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path(os.environ.get('PCI_BASE', '/home/ubuntu/PCI_AAMAS27_Rebuild_v1')).resolve()
OLD_VARIANT = Path(os.environ.get('PCI_OLD_BLINDED', '/home/ubuntu/PCI_Blinded_Action_v2')).resolve()
FOLLOWUP = Path(os.environ.get('PCI_FOLLOWUP', '/home/ubuntu/PCI_AAMAS27_Followup_v1')).resolve()
VARIANT = ROOT / 'variant'
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
sys.path.insert(0, str(VARIANT))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_csv(path, rows):
    if not rows: raise ValueError('Refusing schema-less CSV')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    fields = list(rows[0])
    with temp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='raise'); w.writeheader()
        for row in rows:
            if set(row) != set(fields): raise ValueError('CSV schema changed')
            w.writerow(row)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def utc(): return datetime.now(timezone.utc).isoformat()


def state(stage, **values):
    v = dict(stage=stage, utc=utc(), pid=os.getpid(), **values)
    atomic_json(ROOT / 'validation' / 'STATE.json', v)
    print(stage, json.dumps(values, sort_keys=True), 'UTC=' + v['utc'], flush=True)


def json_file(path): return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f, strict=True)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Bad CSV header: ' + str(path))
        rows = list(reader)
        if any(None in r or any(v is None for v in r.values()) for r in rows):
            raise ValueError('Bad CSV row: ' + str(path))
        return rows


def available_space():
    free = shutil.disk_usage(ROOT).free / 2**30
    if free < PROTOCOL['disk_stop_free_gib']:
        raise RuntimeError(f'Only {free:.2f} GiB free: preserve artifacts, do not continue')
    return free


def freeze_json(path, value):
    if Path(path).exists():
        if json_file(path) != value: raise RuntimeError('Immutable specification changed: ' + str(path))
    else: atomic_json(path, value)


def exported_sources():
    out = {p.name: sha(p) for p in ROOT.iterdir()
           if p.is_file() and p.suffix in ('.py', '.sh', '.md', '.json')
           and p.name not in ('PREPARATION.json', 'SOURCE_MANIFEST.json', 'bundle_manifest.json')}
    for folder in ('pci_bench', 'crypto'):
        for p in sorted((VARIANT / folder).glob('*')):
            if p.is_file() and not p.is_symlink(): out['variant/' + folder + '/' + p.name] = sha(p)
    return out


def prepare():
    for name, folder, archive in (
        ('base', BASE, 'PCI_deadline_results.tar.gz'),
        ('followup', FOLLOWUP, 'PCI_followup_results.tar.gz')):
        if sha(folder / archive) != PROTOCOL['frozen_archives'][name]:
            raise RuntimeError('Frozen archive identity failed: ' + name)
    if (OLD_VARIANT / 'validation/STATE.txt').read_text().strip() != 'CRYPTO_SMOKE_COMPLETE_NOT_BENCHMARKED':
        raise RuntimeError('Prior blinded variant did not finish validation')
    for rel, expected in [('validation/real_crypto/CRYPTO_TESTS_PASSED.json', 21),
                          ('validation/blinded_binding/REAL_BINDING_CHECKS.json', 10)]:
        d = json_file(OLD_VARIANT / rel)
        if d['status'] != 'PASS' or d['count'] != expected: raise RuntimeError('Invalid prior smoke evidence')
    wanted = {}
    for directory in ('pci_bench', 'configs'):
        for src in sorted((OLD_VARIANT / directory).glob('*')):
            if src.is_file() and src.suffix in ('.py', '.json'):
                wanted[directory + '/' + src.name] = sha(src)
    for name in ('worker.cjs', 'primitives.circom', 'package.json', 'package-lock.json'):
        wanted['crypto/' + name] = sha(OLD_VARIANT / 'crypto' / name)
    # Copy only scientific sources, never results or old keys. Links are dependencies.
    if not VARIANT.exists():
        VARIANT.mkdir()
        for rel in wanted:
            dest = VARIANT / rel; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(OLD_VARIANT / rel, dest)
        for rel in ('crypto/node_modules', 'crypto/bin', 'crypto/setup'):
            src = OLD_VARIANT / rel
            if not src.exists(): raise RuntimeError('Dependency absent: ' + str(src))
            (VARIANT / rel).symlink_to(src.resolve(), target_is_directory=True)
    for rel, expected in wanted.items():
        if sha(VARIANT / rel) != expected: raise RuntimeError('Copied source changed: ' + rel)
    for name in ('model.py', 'environment.py', 'evaluate.py'):
        if sha(VARIANT / 'pci_bench' / name) != sha(BASE / 'pci_bench' / name):
            raise RuntimeError('Frozen policy/environment incompatibility')
    checkpoints = {}
    for method in PROTOCOL['methods']:
        for seed in PROTOCOL['training_seeds']:
            path = BASE / 'results/deadline/runs' / f'{method}_twodoors_l0.2_s{seed}'
            mark = json_file(path / 'TRAIN_COMPLETE.json')
            if sha(path / 'checkpoint.pt') != mark['checkpoint_sha256']:
                raise RuntimeError('Checkpoint identity failed: ' + str(path))
            checkpoints[f'{method}_s{seed}'] = mark['checkpoint_sha256']
    freeze_json(ROOT / 'PREPARATION.json', dict(source_hashes=wanted, checkpoints=checkpoints,
        base_archives=PROTOCOL['frozen_archives'], old_blinded=str(OLD_VARIANT), schema_version=1))
    state('INPUTS_AND_FROZEN_CHECKPOINTS_VERIFIED', checkpoints=len(checkpoints), disk_free_gib=available_space())


# Separate scalar operational oracle; does not call environment monitor/transition.
def oracle(record):
    if record['config']['layout'] != 'twodoors': raise ValueError('Oracle is scoped to twodoors')
    old = record['before']['positions']; counts = record['before']['counts']; acts = record['actions']
    n = record['config']['n_agents']; q = record['quotas']; cap = record['capacities']
    if any(len(x) != n for x in (old, counts, acts, q)) or len(cap) != 2: raise ValueError('Shape mismatch')
    moves = ((0,0),(1,0),(-1,0),(0,-1),(0,1)); errors=[]; nxt=[]; entries=[[0,0] for _ in old]
    floor = lambda p: 1 <= p[0] <= 11 and 1 <= p[1] <= 11 and (p[0] != 6 or p[1] in (3,9))
    for i, a in enumerate(acts):
        if type(a) is not int or not 0 <= a < 5: raise ValueError('Bad action')
        if len(old[i]) != 2 or len(counts[i]) != 2 or len(q[i]) != 2: raise ValueError('Bad shape')
        if not all(type(v) is int for v in old[i] + counts[i] + q[i]): raise ValueError('Noninteger state')
        v=[old[i][0]+moves[a][0],old[i][1]+moves[a][1]]; nxt.append(v)
        if not floor(old[i]): errors.append(f'old_floor:{i}')
        if not floor(v): errors.append(f'floor:{i}')
        for r,y in enumerate((3,9)):
            was=5<=old[i][0]<=7 and old[i][1]==y; now=5<=v[0]<=7 and v[1]==y
            entries[i][r]=int(now and not was)
            if counts[i][r]<0 or counts[i][r]+entries[i][r]>q[i][r]: errors.append(f'quota:{i}:{r}')
    for i in range(n):
        for j in range(i):
            if old[i]==old[j]: errors.append(f'old_vertex:{j}:{i}')
            if nxt[i]==nxt[j]: errors.append(f'vertex:{j}:{i}')
            if nxt[i]==old[j] and nxt[j]==old[i]: errors.append(f'swap:{j}:{i}')
    for r,y in enumerate((3,9)):
        if sum(5<=v[0]<=7 and v[1]==y for v in nxt)>cap[r]: errors.append(f'capacity:{r}')
    return dict(compliant=not errors, errors=errors, next_positions=nxt, entries=entries)


@functools.lru_cache(maxsize=12)
def abi_order(sym: str, vkey: str):
    n=int(json_file(vkey)['nPublic']); fields={}
    expected={'ok','record_hash','run_tag','episode','step','nonce','policy_version','state_root','action_root','rules_root'}
    for line in Path(sym).read_text().splitlines():
        p=line.split(',')
        if len(p)==4 and p[3].startswith('main.') and p[3][5:] in expected:
            idx=int(p[1])
            if 1<=idx<=n: fields[idx]=p[3][5:]
    if n!=10 or set(fields)!=set(range(1,11)) or set(fields.values())!=expected:
        raise RuntimeError('Public ABI is incomplete or exposes unexpected fields')
    return tuple(fields[i] for i in range(1,11))


def vector(expected, paths): return [str(expected[k]) for k in abi_order(paths['sym'],paths['vkey'])]


def signing_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    return Ed25519PrivateKey.generate()


def pubkey(key):
    from cryptography.hazmat.primitives import serialization
    return base64.b64encode(key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)).decode()


def verify_signature(envelope):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    Ed25519PublicKey.from_public_bytes(base64.b64decode(envelope['key'])).verify(
        base64.b64decode(envelope['signature']), canonical(envelope['payload']))


def sign_envelope(payload,key):
    encoded=canonical(payload); start=time.perf_counter_ns(); sig=key.sign(encoded)
    elapsed=(time.perf_counter_ns()-start)/1e6
    return dict(payload=payload,key=pubkey(key),signature=base64.b64encode(sig).decode()),elapsed,len(sig),len(encoded)


def source_plan(smoke=False):
    import numpy as np
    cells=[]
    for method,seed,n,regime,ep in itertools.product(PROTOCOL['methods'],PROTOCOL['training_seeds'],
                  PROTOCOL['populations'],PROTOCOL['regimes'],range(PROTOCOL['episodes_per_cell'])):
        cells.append(dict(method=method,seed=seed,n=n,regime=regime,episode=ep,
                          eval_seed=PROTOCOL['evaluation_seeds'][ep],horizon=PROTOCOL['horizon']))
    if smoke:
        cells=[dict(method=m,seed=0,n=n,regime='nominal',episode=0,eval_seed=810001,horizon=4)
               for m,n in itertools.product(PROTOCOL['methods'],PROTOCOL['populations'])]
    np.random.default_rng(PROTOCOL['order_seed']).shuffle(cells)
    for i,c in enumerate(cells):
        c['case_id']=f"{c['method']}_n{c['n']}_s{c['seed']}_{c['regime']}_e{c['episode']}"
        c['run_tag']=92000000+int(smoke)*10000+i
        c['order_offset']=i%24
    prep=json_file(ROOT/'PREPARATION.json')
    return dict(profile='smoke' if smoke else 'full', protocol=PROTOCOL, cells=cells,
                checkpoints=prep['checkpoints'], sources=exported_sources())


def expect_circuit_rejection(worker, paths, protocol, inp):
    from pci_bench.crypto import CryptoError
    try: worker.request('prove',protocol=protocol,input=inp,wasm=paths['wasm'],zkey=paths['zkey'],vkey=paths['vkey'])
    except CryptoError as e:
        if not re.search(r'assert|constraint|template',str(e),re.I): raise
    else: raise AssertionError('An unsatisfied witness was accepted')


def preflight():
    from pci_bench.crypto import NodeWorker, build_witness, artifact_paths, public_vector, FIELD
    from pci_bench.crypto_check import samples
    from pci_bench.crypto_build import build
    state('BUILDING_NEW_BLINDED_CIRCUITS')
    for n in PROTOCOL['populations']:
        build('twodoors',n)
    good,bad=samples(); passed=[]
    pads=[[1,11],[3,11],[5,11],[7,11],[9,11],[11,11]]
    out=ROOT/'validation/preflight'; out.mkdir(parents=True,exist_ok=True)
    with NodeWorker(timeout=1200) as worker:
        for n in PROTOCOL['populations']:
            def expand(r):
                r=copy.deepcopy(r); r['config']['n_agents']=n
                r['before']['positions']+=pads[:n-2];r['before']['counts']+=[[0,0] for _ in range(n-2)]
                r['actions']+=[0]*(n-2);r['quotas']+=[[3,3] for _ in range(n-2)];return r
            for protocol in ('groth16','plonk'):
                paths=artifact_paths('twodoors',n,protocol)
                witness=build_witness(expand(good),worker,run_tag=910000+n)
                if 'action_salt' not in witness['input'] or 'action_salt' in witness['expected']:
                    raise AssertionError('Not the private-opening variant')
                if vector(witness['expected'],paths)!=public_vector(witness['expected'],paths):
                    raise AssertionError('Cached ABI differs from source implementation')
                result=worker.request('prove',protocol=protocol,input=witness['input'],wasm=paths['wasm'],zkey=paths['zkey'],vkey=paths['vkey'])
                if result['valid'] is not True or result['publicSignals']!=vector(witness['expected'],paths):
                    raise AssertionError('Honest blinded proof failed')
                atomic_json(out/f'n{n}_{protocol}_honest.json',dict(witness=witness,result=result))
                passed += [f'n{n}_{protocol}_honest_and_private_abi']
                changed=result['publicSignals'].copy();changed[-1]=str((int(changed[-1])+1)%FIELD)
                if worker.request('verify',protocol=protocol,vkey=paths['vkey'],proof=result['proof'],publicSignals=changed)['valid']:
                    raise AssertionError('Changed public root accepted')
                passed.append(f'n{n}_{protocol}_public_tamper')
                wrong=copy.deepcopy(witness['input']);wrong['action_salt']=str((int(wrong['action_salt'])+1)%FIELD)
                expect_circuit_rejection(worker,paths,protocol,wrong);passed.append(f'n{n}_{protocol}_opening_substitution')
                for name,r in bad.items():
                    r=expand(r)
                    if oracle(r)['compliant']:raise AssertionError('Bad negative fixture')
                    w=build_witness(r,worker,run_tag=910000+n,strict=False)
                    expect_circuit_rejection(worker,paths,protocol,w['input'])
                    passed.append(f'n{n}_{protocol}_{name}')
                w=copy.deepcopy(witness['input']);w['used'][0][0]='1'
                expect_circuit_rejection(worker,paths,protocol,w);passed.append(f'n{n}_{protocol}_private_substitution')
                # Library-level range checks retain a valid altered commitment while rejecting alias arithmetic.
                w=copy.deepcopy(witness['input']);w['used'][0][0]=str(FIELD-1)
                vals=[int(w['run_tag']),int(w['episode']),int(w['step'])]
                for i in range(n):vals += [int(w['x'][i]),int(w['y'][i]),int(w['used'][i][0]),int(w['used'][i][1])]
                w['state_root']=str(worker.commit(1001,vals,int(w['salt'])))
                expect_circuit_rejection(worker,paths,protocol,w);passed.append(f'n{n}_{protocol}_range_alias')
                # Same cryptographically valid proof cannot authorize the token twice.
                from pci_bench.crypto import AdmissionVerifier
                gate=AdmissionVerifier(worker,paths,protocol)
                if not gate.accept(result['proof'],result['publicSignals'],witness['expected']):raise AssertionError('Fresh rejected')
                if gate.accept(result['proof'],result['publicSignals'],witness['expected']):raise AssertionError('Replay accepted')
                passed.append(f'n{n}_{protocol}_replay')
        # Signed false claim: authentic signature is deliberately not predicate assurance.
        r=bad['capacity'];key=signing_key()
        envelope,_,_,_=sign_envelope(dict(kind='asserted_compliance',claim=True,record=r),key)
        verify_signature(envelope)
        if oracle(r)['compliant']:raise AssertionError('False-assertion fixture unexpectedly compliant')
        atomic_json(out/'signed_false_assertion.json',dict(envelope=envelope,signature_valid=True,oracle_compliant=False,
            meaning='Signature authenticates an incorrect assertion when its signer is malicious; this is not signature forgery.'))
        passed.append('authentic_false_assertion_is_not_predicate_assurance')
    atomic_json(out/'COMPLETE.json',dict(status='PASS',count=len(passed),checks=passed,
        scope='Real repaired circuits at 2,4,8 agents; finite functional tests, not a privacy or formal security proof.'))
    state('CRYPTOGRAPHIC_PREFLIGHT_PASSED',checks=len(passed))


def case_files(folder):
    return {str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name not in ('COMPLETE.json','FAILED.json')}


def run_case(c, folder, worker, paths_by_n):
    import numpy as np
    import torch
    from pci_bench.crypto import build_witness
    from pci_bench.environment import EnvConfig, Warehouse
    from pci_bench.model import ContextPolicy
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    expected_sig=hashlib.sha256(canonical(dict(case=c,protocol=PROTOCOL,sources=exported_sources()))).hexdigest()
    marker=folder/'COMPLETE.json'
    if marker.exists():
        saved=json_file(marker)
        if saved['signature']!=expected_sig or case_files(folder)!=saved['files']:raise RuntimeError('Completed case changed')
        return saved
    if any(folder.iterdir()):
        # Preserve the incomplete attempt, with a distinct name; never mix timings.
        old=folder.with_name(folder.name+'_interrupted_'+str(time.time_ns()));os.replace(folder,old);folder.mkdir()
    freeze_json(folder/'case.json',c)
    torch.set_num_threads(1)
    run=BASE/'results/deadline/runs'/f"{c['method']}_twodoors_l0.2_s{c['seed']}"
    cfg=json_file(run/'config.json');checkpoint=run/'checkpoint.pt'
    if sha(checkpoint)!=json_file(ROOT/'PREPARATION.json')['checkpoints'][f"{c['method']}_s{c['seed']}"]:
        raise RuntimeError('Checkpoint changed before execution')
    model=ContextPolicy(cfg['method'],cfg['hidden'])
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
    env=Warehouse(EnvConfig(layout='twodoors',n_agents=c['n'],horizon=c['horizon'],quota_window=64,
                  quota=1 if c['regime']=='quota_tight' else 3,capacity=2,rule_change=c['regime']=='rule_shift'),c['eval_seed'])
    rng=np.random.default_rng(c['eval_seed']+7000000);key=signing_key();sums=defaultdict(float)
    rows=[];consumed=set();orders=list(itertools.permutations(PROTOCOL['backends']))
    sample_steps=PROTOCOL['paired_steps'] if c['horizon']==256 else [0]
    started=time.perf_counter();resets=[];rule_versions=[]
    public_stream=gzip.open(folder/'snark_public.jsonl.gz','wt')
    private_stream=gzip.open(folder/'private_witness.jsonl.gz','wt')
    signature_stream=gzip.open(folder/'signature_views.jsonl.gz','wt')
    trace_stream=gzip.open(folder/'trace.jsonl.gz','wt')
    try:
        while env.t<c['horizon']:
            available_space();t=env.t
            ob=torch.from_numpy(env.features()[None])
            start=time.perf_counter_ns()
            with torch.no_grad():prob=model(ob)[0].probs[0].numpy().astype(float)
            actions=np.array([rng.choice(5,p=v/v.sum()) for v in prob],dtype=np.int64)
            sums['policy_inference_ms']+=(time.perf_counter_ns()-start)/1e6
            before=env.pos.copy();counts_before=env.counts.copy()
            candidate,_=env.candidates(actions);physical,_=env.physical_resolve(candidate);admitted=env.shield(candidate)
            physical_actions=np.array(env.actions_for_positions(physical));admitted_actions=np.array(env.actions_for_positions(admitted))
            physical_replacements=int(np.sum(actions!=physical_actions))
            institutional_replacements=int(np.sum(physical_actions!=admitted_actions))
            sums['physical_agent_replacements']+=physical_replacements
            sums['institutional_agent_replacements']+=institutional_replacements
            sums['institutional_changed_joint_steps']+=int(institutional_replacements>0)
            step_admission_ms=0.
            def certify(record):
                nonlocal step_admission_ms
                step_start=time.perf_counter_ns()
                if record['actions']!=admitted_actions.tolist():raise AssertionError('Correction prediction differs from execution')
                check=oracle(record)
                if not check['compliant']:raise AssertionError('Noncompliant admitted action')
                if t>0 and t%64==0:
                    if np.any(counts_before):raise AssertionError('Counters not reset')
                    resets.append(t)
                begin=time.perf_counter_ns();built=build_witness(record,worker,run_tag=c['run_tag'])
                binding_ms=(time.perf_counter_ns()-begin)/1e6
                rule_versions.append(int(built['expected']['policy_version']))
                token=(built['expected']['run_tag'],built['expected']['episode'],built['expected']['step'])
                if token in consumed:raise AssertionError('Repeated interaction')
                private_stream.write(canonical(dict(step=t,record=record,input=built['input'],expected=built['expected'])).decode()+'\n')
                paired=t in sample_steps
                order=orders[(sample_steps.index(t)+c['order_offset'])%24] if paired else ('groth16',)
                for oi,protocol in enumerate(order):
                    start=time.perf_counter_ns();extra_ms=0.
                    if protocol in ('groth16','plonk'):
                        paths=paths_by_n[c['n']][protocol]
                        out=worker.request('prove',protocol=protocol,input=built['input'],wasm=paths['wasm'],zkey=paths['zkey'],vkey=paths['vkey'])
                        if out['valid'] is not True or out['publicSignals']!=vector(built['expected'],paths):
                            raise AssertionError('Proof/public statement rejected')
                        # fullProve worker verifies; admit only this verified, expected statement.
                        if protocol=='groth16':consumed.add(token)
                        prove_ms=out['witness_and_prove_ms'];verify_ms=out['verify_ms']
                        proof_bytes=out['proof_json_bytes'];public_bytes=out['public_json_bytes']
                        public_stream.write(canonical(dict(step=t,protocol=protocol,expected=built['expected'],
                                      proof=out['proof'],publicSignals=out['publicSignals'])).decode()+'\n')
                        encoding='snarkjs_compact_json'
                    else:
                        if protocol=='signed_assertion':
                            payload=dict(kind='pci.signed_assertion.v1',expected=built['expected'],compliant=True)
                        else:
                            payload=dict(kind='pci.signed_disclosure.v1',expected=built['expected'],record=record,
                                         openings=dict(state=built['input']['salt'],action=built['input']['action_salt']))
                        envelope,prove_ms,proof_bytes,public_bytes=sign_envelope(payload,key)
                        vb=time.perf_counter_ns();verify_signature(envelope)
                        if envelope['payload']['expected']!=built['expected']:raise AssertionError('Signature binding failure')
                        if protocol=='signed_disclosure':
                            opened=build_witness(record,worker,run_tag=c['run_tag'],nonce=int(built['expected']['nonce']),
                                      salt=int(payload['openings']['state']),action_salt=int(payload['openings']['action']))
                            if opened['expected']!=built['expected'] or not oracle(record)['compliant']:
                                raise AssertionError('Disclosure failed predicate/commitment reevaluation')
                        verify_ms=(time.perf_counter_ns()-vb)/1e6;encoding='raw_ed25519_signature'
                        signature_stream.write(canonical(dict(step=t,protocol=protocol,envelope=envelope)).decode()+'\n')
                    rows.append(dict(case_id=c['case_id'],method=c['method'],seed=c['seed'],agents=c['n'],regime=c['regime'],
                         eval_episode=c['episode'],step=t,protocol=protocol,paired=int(paired),order_index=oi,order='>'.join(order),
                         record_binding_ms=binding_ms,generation_ms=prove_ms,verification_ms=verify_ms,
                         call_wall_ms=(time.perf_counter_ns()-start)/1e6,proof_bytes=proof_bytes,public_payload_bytes=public_bytes,
                         encoding=encoding,load1=os.getloadavg()[0]))
                if token not in consumed:raise AssertionError('No Groth16 admission')
                step_admission_ms=(time.perf_counter_ns()-step_start)/1e6
                return True
            _,_,_,_,metrics=env.transition(actions,enforce=True,certifier=certify)
            if metrics['executed_compliant']!=1:raise AssertionError('Executed compliance failure')
            for name,value in metrics.items():sums[name]+=float(value)
            sums['changed_joint_steps']+=int(metrics['interventions']>0)
            sums['all_backend_admission_ms']+=step_admission_ms
            trace_stream.write(canonical(dict(step=t,proposal=actions.tolist(),physical_actions=physical_actions.tolist(),
                  record=env.last,metrics=metrics,physical_agent_replacements=physical_replacements,
                  institutional_agent_replacements=institutional_replacements,counts_after_reset=env.counts.tolist(),
                  cumulative_deliveries=env.delivered.tolist())).decode()+'\n')
            if (t+1)%16==0 or t+1==c['horizon']:
                for f in (private_stream,public_stream,signature_stream,trace_stream):f.flush()
                atomic_json(ROOT/'validation/PROGRESS.json',dict(case_id=c['case_id'],step=t+1,horizon=c['horizon'],
                           profile='full' if c['horizon']==256 else 'smoke',measurements=len(rows),utc=utc()))
                print(f"CERTIFIED {c['case_id']} {t+1}/{c['horizon']} records={len(rows)}",flush=True)
    finally:
        for f in (private_stream,public_stream,signature_stream,trace_stream):f.close()
    atomic_csv(folder/'measurements.csv',rows)
    if c['horizon']==256:
        if resets!=[64,128,192]:raise AssertionError('Reset coverage missing')
        if c['regime']=='rule_shift' and (rule_versions[:128]!=[1]*128 or rule_versions[128:]!=[2]*128):
            raise AssertionError('Rule version transition missing')
        paired_orders=Counter(r['order'] for r in rows if r['paired'])
        if len(paired_orders)!=24 or set(paired_orders.values())!={4}:raise AssertionError('Unbalanced protocol order')
    summary=dict(case_id=c['case_id'],method=c['method'],seed=c['seed'],agents=c['n'],regime=c['regime'],
       eval_episode=c['episode'],joint_steps=c['horizon'],agent_decisions=c['horizon']*c['n'],
       deliveries=int(sums['delivered']),throughput=1000*sums['delivered']/c['horizon'],
       proposed_compliance=sums['proposed_compliant']/c['horizon'],executed_compliance=sums['executed_compliant']/c['horizon'],
       institutional_intervention_rate=1000*sums['institutional_agent_replacements']/(c['horizon']*c['n']),
       physical_agent_replacements=int(sums['physical_agent_replacements']),institutional_agent_replacements=int(sums['institutional_agent_replacements']),
       all_agent_replacements=int(sums['interventions']),changed_joint_steps=int(sums['changed_joint_steps']),
       institutional_changed_joint_steps=int(sums['institutional_changed_joint_steps']),
       quota_violations=int(sums['quota_violations']),capacity_excess=int(sums['capacity_excess']),
       reset_count=len(resets),rule_changes=sum(x!=y for x,y in zip(rule_versions,rule_versions[1:])),
       all_backend_admission_ms=sums['all_backend_admission_ms'],episode_wall_s=time.perf_counter()-started,
       benchmark_note='Includes all paired backend costs; not Groth16-only runtime')
    atomic_json(folder/'episode.json',summary)
    audit=audit_case(c,folder,worker,paths_by_n)
    atomic_json(folder/'AUDIT.json',audit)
    saved=dict(status='COMPLETE',signature=expected_sig,files=case_files(folder),summary=summary,audit=audit)
    atomic_json(marker,saved)
    return saved


def gzrows(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def audit_case(c,folder,worker,paths_by_n):
    import numpy as np
    from pci_bench.crypto import build_witness
    witness=list(gzrows(folder/'private_witness.jsonl.gz'));traces=list(gzrows(folder/'trace.jsonl.gz'))
    if len(witness)!=c['horizon'] or len(traces)!=c['horizon']:raise AssertionError('Missing trace')
    for t,(w,r) in enumerate(zip(witness,traces)):
        if w['step']!=t or r['step']!=t:raise AssertionError('Wrong step order')
        check=oracle(w['record'])
        if not check['compliant'] or check['next_positions']!=r['record']['after_positions']:
            raise AssertionError('Independent oracle rejected archived transition')
        if t:
            if w['record']['before']['positions']!=traces[t-1]['record']['after_positions']:
                raise AssertionError('Position history mismatch')
            if w['record']['before']['counts']!=traces[t-1]['counts_after_reset']:
                raise AssertionError('Counter history mismatch')
        counts=(np.array(w['record']['before']['counts'])+np.array(check['entries'])).tolist()
        if (t+1)%64==0:counts=[[0,0] for _ in range(c['n'])]
        if counts!=r['counts_after_reset']:raise AssertionError('Counter reset incorrect')
        quota=1 if c['regime']=='quota_tight' else 2 if c['regime']=='rule_shift' and t>=128 else 3
        if w['record']['quotas']!=[[quota,quota] for _ in range(c['n'])] or w['record']['capacities']!=[2,2]:
            raise AssertionError('Wrong authentic rules')
        opened=build_witness(w['record'],worker,run_tag=c['run_tag'],nonce=int(w['input']['nonce']),
                  salt=int(w['input']['salt']),action_salt=int(w['input']['action_salt']))
        if opened['expected']!=w['expected']:raise AssertionError('Archived commitment reconstruction failed')
    proofs=0;seen=set()
    for r in gzrows(folder/'snark_public.jsonl.gz'):
        key=(r['step'],r['protocol'])
        if key in seen:raise AssertionError('Duplicate proof');
        seen.add(key);p=paths_by_n[c['n']][r['protocol']]
        if r['expected']!=witness[r['step']]['expected'] or r['publicSignals']!=vector(r['expected'],p):raise AssertionError('Archived ABI mismatch')
        if worker.request('verify',protocol=r['protocol'],vkey=p['vkey'],proof=r['proof'],publicSignals=r['publicSignals'])['valid'] is not True:
            raise AssertionError('Archived proof re-verification failed')
        proofs+=1
    samples=PROTOCOL['paired_steps'] if c['horizon']==256 else [0]
    expected={(t,'groth16') for t in range(c['horizon'])}|{(t,'plonk') for t in samples}
    if seen!=expected:raise AssertionError('Missing or extra proof coverage')
    signatures=0;sigseen=set()
    for r in gzrows(folder/'signature_views.jsonl.gz'):
        key=(r['step'],r['protocol'])
        if key in sigseen:raise AssertionError('Duplicate signature')
        sigseen.add(key);verify_signature(r['envelope'])
        if r['envelope']['payload']['expected']!=witness[r['step']]['expected']:raise AssertionError('Wrong signature binding')
        if r['protocol']=='signed_disclosure':
            payload=r['envelope']['payload'];w=witness[r['step']]
            if payload['record']!=w['record'] or payload['openings']!={'state':w['input']['salt'],'action':w['input']['action_salt']}:
                raise AssertionError('Disclosure record changed')
            if not oracle(payload['record'])['compliant']:raise AssertionError('Disclosed predicate false')
        signatures+=1
    if sigseen!={(t,p) for t in samples for p in ('signed_assertion','signed_disclosure')}:raise AssertionError('Signature coverage incomplete')
    summary=json_file(folder/'episode.json')
    if sum(r['metrics']['delivered'] for r in traces)!=summary['deliveries'] or sum(traces[-1]['cumulative_deliveries'])!=summary['deliveries']:
        raise AssertionError('Task ledger mismatch')
    if sum(r['institutional_agent_replacements'] for r in traces)!=summary['institutional_agent_replacements']:
        raise AssertionError('Intervention ledger mismatch')
    return dict(status='PASS',transitions=len(traces),proofs_reverified=proofs,signatures_reverified=signatures,
                scope='Post-run archive replay, separate scalar transition oracle; same cryptographic library, not an external security audit.')


def full_run(smoke=False):
    from pci_bench.crypto import NodeWorker, artifact_paths
    profile='smoke' if smoke else 'full';out=ROOT/'results'/profile;out.mkdir(parents=True,exist_ok=True)
    plan=source_plan(smoke);freeze_json(out/'plan.json',plan)
    if not smoke:
        old=json_file(ROOT/'results/smoke/SUITE_COMPLETE.json')
        if old['status']!='COMPLETE' or old['sources']!=exported_sources():raise RuntimeError('Smoke gate failed or source changed')
    paths={n:{p:artifact_paths('twodoors',n,p) for p in ('groth16','plonk')} for n in PROTOCOL['populations']}
    state('SMOKE_ACTIVE' if smoke else 'FULL_BENCHMARK_ACTIVE',episodes=len(plan['cells']))
    summaries=[];complete=0
    with NodeWorker(timeout=1200) as worker:
        for c in plan['cells']:
            folder=out/'cases'/c['case_id']
            try:res=run_case(c,folder,worker,paths)
            except Exception as exc:
                atomic_json(folder/'FAILED.json',dict(error=str(exc),utc=utc()))
                raise
            complete+=1;summaries.append(res['summary'])
            atomic_csv(out/'episode_summary.csv',summaries)
            print(f"EPISODE_COMPLETE {complete}/{len(plan['cells'])} {c['case_id']} deliveries={res['summary']['deliveries']}",flush=True)
    report=dict(status='COMPLETE',profile=profile,episodes=complete,transitions=sum(x['joint_steps'] for x in summaries),
                sources=exported_sources(),completed_utc=utc())
    atomic_json(out/'SUITE_COMPLETE.json',report)
    if not smoke:analyze(out)
    state('SMOKE_COMPLETE' if smoke else 'EXPERIMENT_AND_ANALYSIS_COMPLETE',episodes=complete)


def analyze(out):
    import numpy as np
    from scipy import stats
    out=Path(out);summary=read_rows(out/'episode_summary.csv');plan=json_file(out/'plan.json')
    if len(summary)!=PROTOCOL['full_episodes']:raise AssertionError('Incomplete analysis')
    fields=('throughput','institutional_intervention_rate','proposed_compliance','physical_agent_replacements')
    g=defaultdict(list)
    for r in summary:g[r['method'],int(r['seed']),int(r['agents']),r['regime']].append(r)
    seedmeans={};flat=[]
    for k,rows in g.items():
        if len(rows)!=2:raise AssertionError('Episode pairing failed')
        x={name:float(np.mean([float(r[name]) for r in rows])) for name in fields}
        seedmeans[k]=x;flat.append(dict(method=k[0],seed=k[1],agents=k[2],regime=k[3],**x))
    a=out/'analysis';a.mkdir(exist_ok=True);atomic_csv(a/'seed_means.csv',flat)
    contrasts=[]
    for n,regime,baseline,metric in itertools.product(PROTOCOL['populations'],PROTOCOL['regimes'],
                            ['ppo_penalty','consensus'],['throughput','institutional_intervention_rate']):
        diff=np.array([seedmeans['pci',s,n,regime][metric]-seedmeans[baseline,s,n,regime][metric] for s in PROTOCOL['training_seeds']])
        mean=float(diff.mean());sd=float(diff.std(ddof=1))
        if sd==0:lo=hi=mean;tp=1. if mean==0 else 0.
        else:
            half=float(stats.t.ppf(.975,4))*sd/math.sqrt(5);lo=mean-half;hi=mean+half
            tp=float(stats.ttest_1samp(diff,0).pvalue)
        signed=np.array(list(itertools.product([-1,1],repeat=5)))@diff/5
        fp=float(np.mean(np.abs(signed)>=abs(mean)-1e-12))
        rng=np.random.default_rng(921701);draws=diff[rng.integers(0,5,(5000,5))].mean(1)
        contrasts.append(dict(agents=n,regime=regime,baseline=baseline,endpoint=metric,difference=mean,
            paired_t_low=lo,paired_t_high=hi,t_p=tp,signflip_p=fp,
            bootstrap_low=float(np.quantile(draws,.025)),bootstrap_high=float(np.quantile(draws,.975)),
            paired_seed_differences=json.dumps(diff.tolist()),n_seeds=5))
    if len(contrasts)!=36:raise AssertionError('Wrong primary family')
    for p in ('t_p','signflip_p'):
        order=sorted(range(36),key=lambda i:contrasts[i][p]);current=0.
        for rank,i in enumerate(order):
            current=max(current,(36-rank)*contrasts[i][p]);contrasts[i][p+'_holm36']=min(1.,current)
    atomic_csv(a/'paired_operational_comparisons.csv',contrasts)
    count=0;proofs=signatures=0;timing=defaultdict(list)
    for c in plan['cells']:
        folder=out/'cases'/c['case_id'];d=json_file(folder/'COMPLETE.json')
        if case_files(folder)!=d['files']:raise AssertionError('Case changed before analysis')
        proofs+=d['audit']['proofs_reverified'];signatures+=d['audit']['signatures_reverified']
        for r in read_rows(folder/'measurements.csv'):
            count+=1
            if int(r['paired']):timing[int(r['agents']),r['protocol']].append(r)
    result=[]
    for (n,p),rows in sorted(timing.items()):
        for m in ('generation_ms','verification_ms','call_wall_ms','proof_bytes','public_payload_bytes'):
            arr=np.array([float(r[m]) for r in rows])
            result.append(dict(agents=n,protocol=p,metric=m,paired_samples=len(arr),median=float(np.median(arr)),
                          p95=float(np.quantile(arr,.95)),scope='Descriptive on shared hardware; not independent training samples'))
    atomic_csv(a/'paired_backend_summary.csv',result)
    if count!=PROTOCOL['backend_measurements'] or proofs!=PROTOCOL['saved_snark_proofs']:
        raise AssertionError('Evidence totals do not match protocol')
    atomic_json(a/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',episodes=len(summary),transitions=PROTOCOL['certified_transitions'],
        measured_backend_calls=count,proofs_reverified=proofs,signatures_reverified=signatures,primary_comparisons=36,
        interpretation='New frozen-policy experiment, five training seeds. No general superiority or privacy theorem follows from completion.'))


def package():
    out=ROOT/'results/full';d=json_file(out/'analysis/ANALYSIS_COMPLETE.json')
    if d['status']!='COMPLETE':raise RuntimeError('Analysis incomplete')
    if exported_sources()!=json_file(out/'plan.json')['sources']:raise RuntimeError('Sources changed during full experiment')
    # Verify old archive identities again. Do not edit/delete old data.
    for name,folder,archive in [('base',BASE,'PCI_deadline_results.tar.gz'),('followup',FOLLOWUP,'PCI_followup_results.tar.gz')]:
        if sha(folder/archive)!=PROTOCOL['frozen_archives'][name]:raise RuntimeError('Original evidence changed')
    state('PACKAGING_RESULTS')
    paths=[]
    for folder in (ROOT/'results/full',ROOT/'results/smoke',ROOT/'validation',VARIANT/'pci_bench',VARIANT/'crypto/build',VARIANT/'crypto/generated'):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.name not in ('launch.log','controller.pid','STATE.json','controller.lock'):
                paths.append((p,str(p.relative_to(ROOT))))
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.py','.sh','.md','.json') and p.name!='bundle_manifest.json':paths.append((p,p.name))
    for n in ('package.json','package-lock.json','primitives.circom','worker.cjs'):
        p=VARIANT/'crypto'/n;paths.append((p,str(p.relative_to(ROOT))))
    for m,s in itertools.product(PROTOCOL['methods'],PROTOCOL['training_seeds']):
        folder=BASE/'results/deadline/runs'/f'{m}_twodoors_l0.2_s{s}'
        for n in ('checkpoint.pt','config.json','TRAIN_COMPLETE.json'):
            paths.append((folder/n,'frozen_checkpoints/'+folder.name+'/'+n))
    manifest={name:sha(p) for p,name in paths}
    if len(manifest)!=len(paths):raise AssertionError('Archive duplicate paths')
    atomic_json(ROOT/'bundle_manifest.json',dict(files=manifest,protocol_sha256=sha(ROOT/'protocol.json'),
        note='Includes synthetic private witnesses/openings and trusted checkpoints; not auditor-only or production archive.'))
    target=ROOT/'PCI_postsubmission_certified_results.tar.gz';temp=target.with_suffix('.partial')
    with tarfile.open(temp,'w:gz',compresslevel=3) as tar:
        for p,name in paths:tar.add(p,arcname=name,recursive=False)
        tar.add(ROOT/'bundle_manifest.json',arcname='bundle_manifest.json',recursive=False)
    os.replace(temp,target)
    with tarfile.open(target,'r:gz') as tar:
        for name,digest in manifest.items():
            f=tar.extractfile(name);h=hashlib.sha256()
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
            if h.hexdigest()!=digest:raise RuntimeError('Archive byte verification failed: '+name)
    digest=sha(target);target.with_suffix(target.suffix+'.sha256').write_text(digest+'  '+target.name+'\n')
    state('POSTSUBMISSION_FULL_BENCHMARK_ARCHIVE_VERIFIED',archive=str(target),sha256=digest,files=len(manifest))


def status():
    for p in ('validation/STATE.json','validation/PROGRESS.json','results/full/analysis/ANALYSIS_COMPLETE.json'):
        if (ROOT/p).exists():print(p,(ROOT/p).read_text())
    root=ROOT/'results/full'
    print('Full episodes complete:',len(list((root/'cases').glob('*/COMPLETE.json'))),'/',PROTOCOL['full_episodes'])
    for f in (root/'cases').glob('*/FAILED.json'):print('FAILED:',f.parent.name,f.read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight','smoke','full','package','status'])
    args=p.parse_args()
    try:
        if args.command=='prepare':prepare()
        elif args.command=='preflight':preflight()
        elif args.command=='smoke':full_run(True)
        elif args.command=='full':full_run(False)
        elif args.command=='package':package()
        else:status()
    except Exception as e:
        state('STOPPED_ON_ERROR',stage_command=args.command,error=str(e))
        raise
