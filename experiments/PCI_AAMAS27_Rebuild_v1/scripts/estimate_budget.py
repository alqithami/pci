#!/usr/bin/env python3
"""A measured training-only estimate, explicitly not a wall-clock guarantee."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pci_bench.suite import create_plan
p=argparse.ArgumentParser();p.add_argument('smoke',nargs='?',default='results/smoke');p.add_argument('--jobs',type=int,default=12);a=p.parse_args()
records=[]
for f in Path(a.smoke).glob('runs/*/TRAIN_COMPLETE.json'):
    d=json.loads(f.read_text());records.append(d['env_steps']/d['elapsed_s'])
if not records:raise SystemExit('No completed smoke training records; run the smoke test first.')
speed=min(records);plan=create_plan('deadline')
hours=plan['total_joint_training_steps']/speed/a.jobs/3600
print(f'Measured slowest smoke training rate: {speed:.1f} joint env steps/s per worker')
print(f'Optimistic training-only scaling estimate at {a.jobs} workers: {hours:.2f} h')
print(f'Planning allowance at 2x-4x this estimate: {2*hours:.2f}-{4*hours:.2f} h, PLUS held-out evaluations and proof setup/benchmarks.')
print('This is not a runtime promise. Contention, CPU clock, larger populations and proof construction are measured separately.')
