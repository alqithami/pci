#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results',nargs='?',default='results/deadline');a=p.parse_args()
r=Path(a.results);plan=json.loads((r/'plan.json').read_text());trained=evaluated=failed=0
for cfg in plan['runs']:
    d=r/'runs'/cfg['run_id'];trained+=(d/'TRAIN_COMPLETE.json').exists();evaluated+=(d/'EVAL_COMPLETE.json').exists()
    if (d/'FAILED.json').exists():print('FAILED:',cfg['run_id'],(d/'FAILED.json').read_text());failed+=1
controllers=sum((r/'controllers'/x['run_id']/'EVAL_COMPLETE.json').exists() for x in plan['controllers'])
print(f"Training: {trained}/{len(plan['runs'])}; evaluation: {evaluated}/{len(plan['runs'])}; controllers: {controllers}/{len(plan['controllers'])}; failed: {failed}")
print('Crypto preflight:',(r/'preflight/crypto/CRYPTO_TESTS_PASSED.json').exists())
print('Crypto benchmark:',(r/'crypto_benchmark/CRYPTO_BENCH_COMPLETE.json').exists())
if (r/'SUITE_COMPLETE.json').exists():print((r/'SUITE_COMPLETE.json').read_text())
else:print('NOT COMPLETE. Preserve this results directory; rerun the same launch command to resume.')
