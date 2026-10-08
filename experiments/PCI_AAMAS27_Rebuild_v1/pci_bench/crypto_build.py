from __future__ import annotations
import argparse, hashlib, json, os, secrets, shutil, subprocess, time
from pathlib import Path
from .crypto_generate import generate
from .io import atomic_json,sha256

ROOT=Path(__file__).resolve().parents[1]
PTAU_HASH='6a6277a2f74e1073601b4f9fed6e1e55226917efb0f0db8a07d98ab01df1ccf43eb0e8c3159432acd4960e2f29fe84a4198501fa54c8dad9e43297453efec125'
PTAU_URL='https://circom.info/powersOfTau28_hez_final_16.ptau'


def run(cmd,log,timeout=3600,stdin=None,show=True):
    if show:print(' '.join(map(str,cmd)),flush=True)
    with Path(log).open('a') as f:
        f.write('\nCOMMAND: '+(' '.join(map(str,cmd)) if show else '[contribution; entropy deliberately not logged]')+'\n');f.flush()
        p=subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,input=stdin,text=True)
    if p.returncode:raise RuntimeError(f'Command failed ({p.returncode}). See {log}')


def ensure_ptau():
    folder=ROOT/'crypto/setup';folder.mkdir(parents=True,exist_ok=True);ptau=folder/'powersOfTau28_hez_final_16.ptau'
    if not ptau.exists():
        tmp=ptau.with_suffix('.part')
        run(['curl','--fail','--location','--retry','4','--output',tmp,PTAU_URL],folder/'download.log')
        os.replace(tmp,ptau)
    h=hashlib.blake2b()
    with ptau.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    if h.hexdigest()!=PTAU_HASH:raise RuntimeError('PTAU BLAKE2b mismatch. Refusing setup.')
    verified=folder/'verified.json';snark=ROOT/'crypto/node_modules/.bin/snarkjs'
    if not verified.exists():
        run([snark,'powersoftau','verify',ptau],folder/'verify.log',timeout=7200)
        atomic_json(verified,{'source':PTAU_URL,'blake2b':h.hexdigest(),'sha256':sha256(ptau)})
    return ptau


def build(layout,n):
    circom=os.environ.get('CIRCOM') or str(ROOT/'crypto/bin/circom')
    if not Path(circom).is_file():circom=shutil.which('circom')
    if not circom:raise RuntimeError('Circom missing: run setup_ubuntu.sh')
    snark=ROOT/'crypto/node_modules/.bin/snarkjs'
    if not snark.exists():raise RuntimeError('snarkjs missing: run setup_ubuntu.sh')
    source=generate(layout,n,ROOT/'crypto/generated');name=source.stem
    out=ROOT/'crypto/build'/name;out.mkdir(parents=True,exist_ok=True)
    compiler_version=subprocess.check_output([circom,'--version'],text=True).strip()
    wanted={'source':sha256(source),'primitives':sha256(ROOT/'crypto/primitives.circom'),
            'compiler':compiler_version,'npm_lock':sha256(ROOT/'crypto/package-lock.json')}
    manifest=out/'manifest.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        if meta['build_inputs']!=wanted:raise RuntimeError(f'Crypto sources changed; archive/remove only {out} and rebuild')
        for k,v in meta['files'].items():
            if sha256(out/k)!=v:raise RuntimeError(f'Crypto artifact changed: {out/k}')
        return out
    ptau=ensure_ptau();log=out/'build.log';started=time.perf_counter()
    run([circom,source,'--r1cs','--wasm','--sym','--inspect','-l',ROOT/'crypto/node_modules','-l',ROOT/'crypto','-o',out],log)
    r1cs=out/f'{name}.r1cs';run([snark,'r1cs','info',r1cs],log)
    run([snark,'groth16','setup',r1cs,ptau,out/'initial.zkey'],log)
    # A private, OS-random contribution by the experiment operator. This is a
    # benchmark trust model, not a claim of an independent production MPC.
    entropy=secrets.token_hex(64)
    run([snark,'zkey','contribute',out/'initial.zkey',out/'groth16.zkey','--name=PCI-benchmark',f'-e={entropy}'],log,show=False)
    run([snark,'zkey','verify',r1cs,ptau,out/'groth16.zkey'],log)
    run([snark,'zkey','export','verificationkey',out/'groth16.zkey',out/'groth16.vkey.json'],log)
    run([snark,'plonk','setup',r1cs,ptau,out/'plonk.zkey'],log)
    run([snark,'zkey','export','verificationkey',out/'plonk.zkey',out/'plonk.vkey.json'],log)
    filenames=[f'{name}.r1cs',f'{name}.sym',f'{name}_js/{name}.wasm','groth16.zkey','groth16.vkey.json','plonk.zkey','plonk.vkey.json']
    atomic_json(manifest,{'build_inputs':wanted,'files':{k:sha256(out/k) for k in filenames},
                          'setup_seconds':time.perf_counter()-started,'ptau_blake2b':PTAU_HASH,
                          'phase2_trust':'single local OS-random contribution; benchmark only'})
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--agents',nargs='+',type=int,default=[2]);p.add_argument('--layout',default='twodoors')
    a=p.parse_args()
    for n in a.agents:print(build(a.layout,n))
