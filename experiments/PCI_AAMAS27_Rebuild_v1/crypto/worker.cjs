#!/usr/bin/env node
'use strict';
// JSONL worker. No cryptographic fallback. stdout contains protocol messages only.
const readline = require('readline');
const fs = require('fs');
const { performance } = require('perf_hooks');
const FIELD = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;
let poseidon;
let snark;
const vkeys = new Map();
function integer(v) {
  if (typeof v === 'number') {
    if (!Number.isSafeInteger(v)) throw Error('Unsafe JSON number: use an integer string');
    v = BigInt(v);
  } else if (typeof v === 'string' && /^-?(0x[0-9a-fA-F]+|[0-9]+)$/.test(v)) {
    v = BigInt(v);
  } else if (typeof v !== 'bigint') throw Error('Expected an integer, never a float');
  return (v % FIELD + FIELD) % FIELD;
}
async function hash(v) {
  if (!poseidon) poseidon = await require('circomlibjs').buildPoseidon();
  if (!Array.isArray(v) || v.length < 1 || v.length > 16) throw Error('Poseidon arity must be 1..16');
  return poseidon.F.toString(poseidon(v.map(integer)));
}
async function commit(domain, vals, salt) {
  let h = await hash([domain, vals.length, salt, 1]);
  for(let i=0; i<vals.length; i+=8) {
    let block=vals.slice(i,i+8);
    while(block.length<8)block.push('0');
    h=await hash([h,...block]);
  }
  return h;
}
function loadVkey(p) {
  if(!vkeys.has(p))vkeys.set(p, JSON.parse(fs.readFileSync(p,'utf8')));
  return vkeys.get(p);
}
async function handle(r) {
  if(r.op==='hash')return {hash:await hash(r.values)};
  if(r.op==='commit')return {hash:await commit(r.domain,r.values,r.salt)};
  if(r.op==='ping') {
    if (!snark)snark=require('snarkjs');
    return {node:process.version,poseidon_1_2:await hash(['1','2'])};
  }
  if(!snark)snark=require('snarkjs');
  if(!['groth16','plonk'].includes(r.protocol))throw Error('Unsupported proof protocol');
  if(r.op==='prove') {
    const before=performance.now();
    const result=await snark[r.protocol].fullProve(r.input,r.wasm,r.zkey);
    const after=performance.now();
    const vk=loadVkey(r.vkey);
    const t0=performance.now();
    const valid=await snark[r.protocol].verify(vk,result.publicSignals,result.proof);
    const t1=performance.now();
    return {proof:result.proof,publicSignals:result.publicSignals.map(String),valid:valid===true,
            witness_and_prove_ms:after-before,verify_ms:t1-t0,
            proof_json_bytes:Buffer.byteLength(JSON.stringify(result.proof),'utf8'),
            public_json_bytes:Buffer.byteLength(JSON.stringify(result.publicSignals),'utf8')};
  }
  if(r.op==='verify') {
    const t=performance.now();
    const valid=await snark[r.protocol].verify(loadVkey(r.vkey),r.publicSignals,r.proof);
    return {valid:valid===true,verify_ms:performance.now()-t};
  }
  throw Error('Unknown operation');
}
(async()=>{
  const rl=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
  for await(const line of rl) {
    let r;
    try {r=JSON.parse(line); const value=await handle(r); process.stdout.write(JSON.stringify({id:r.id,ok:true,result:value})+'\n');}
    catch(e) {process.stdout.write(JSON.stringify({id:r&&r.id,ok:false,error:String(e.stack||e)})+'\n');}
  }
  process.exit(0);
})().catch(e=>{process.stderr.write(String(e.stack||e)+'\n');process.exit(1);});
