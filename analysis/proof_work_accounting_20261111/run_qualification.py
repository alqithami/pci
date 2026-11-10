"""Run synthetic ledger controls and retain a source-bound qualification receipt."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = ('check_proof_ledger.py', 'test_check_proof_ledger.py', 'run_qualification.py')
    before = {n: sha(ROOT/n) for n in files}
    r = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v',
        'test_check_proof_ledger.py'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    log = r.stdout + r.stderr
    (ROOT/'QUALIFICATION.log').write_text(log)
    assert before == {n: sha(ROOT/n) for n in files}, 'Qualification source changed'
    assert r.returncode == 0 and 'Ran 37 tests' in log and log.rstrip().endswith('OK'), log
    result = {'schema': 'PCI-PROOF-LEDGER-COUNTER-QUALIFICATION-v1',
        'status': 'PASS_SYNTHETIC_CONTRACT_AND_RETAINED_BYTE_CONTROLS',
        'synthetic_fixtures_only': True, 'genuine_proof_verification_executed': False,
        'scientific_result': False, 'tests': 37, 'exit_code': r.returncode,
        'source_sha256': before, 'log_sha256': sha(ROOT/'QUALIFICATION.log'),
        'qualified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python': sys.version,
        'example_partition': {'offers': 6, 'proof_rpc_invocations': 5,
          'valid_generated_proofs': 5, 'distinct_durably_applied_proof_attempts': 3,
          'valid_unapplied_proofs': 2, 'valid_stale_unapplied_proofs': 1,
          'valid_host_encoding_fault_unapplied_proofs': 1,
          'preproof_rejections': 1, 'duplicate_retries': 1, 'task_units': 6},
        'limits': 'This qualifies the counter/byte-binding contract. Actual-vkey/native proof audits must be independently executed and supplied under explicit trusted source/key pins; this module does not execute them.'}
    (ROOT/'QUALIFICATION_RECEIPT.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'tests', 'scientific_result')}))


if __name__ == '__main__':
    main()
