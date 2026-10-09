#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p validation
exec 9>validation/controller.lock
flock -n 9 || { echo 'A controller already owns this study'; exit 2; }
export PCI_BASE="${PCI_BASE:-/home/ubuntu/PCI_AAMAS27_Rebuild_v1}"
export PCI_OLD_BLINDED="${PCI_OLD_BLINDED:-/home/ubuntu/PCI_Blinded_Action_v2}"
export PCI_FOLLOWUP="${PCI_FOLLOWUP:-/home/ubuntu/PCI_AAMAS27_Followup_v1}"
PY="${PCI_PYTHON:-$PCI_FOLLOWUP/.venv/bin/python}"
export PATH="$PCI_BASE/.toolchain/node/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
# A limited affinity mask is inherited by all new processes; other workloads are untouched.
CPUS="$($PY -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[-6:])))')"
taskset -pc "$CPUS" "$$"
printf '%s\n' "CONTROLLER_PID=$$ CPU_AFFINITY=$CPUS UTC=$(date -u +%FT%TZ)"
printf '%s\n' "$$" > validation/controller.pid
finish() {
  code=$?
  printf 'exit=%s UTC=%s\n' "$code" "$(date -u +%FT%TZ)" > validation/EXIT.txt
  if [ "$code" -ne 0 ]; then printf 'STOPPED_ON_ERROR exit=%s\n' "$code"; fi
}
trap finish EXIT
sha256sum -c SOURCE.sha256
{
  date -u
  uname -a
  lscpu
  free -h
  df -h "$ROOT"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
} > validation/initial_machine.txt
"$PY" benchmark.py prepare
"$PY" tests.py
"$PY" profile_cpu.py
"$PY" benchmark.py preflight
"$PY" benchmark.py smoke
"$PY" benchmark.py full
"$PY" benchmark.py package
sha256sum -c PCI_postsubmission_certified_results.tar.gz.sha256
