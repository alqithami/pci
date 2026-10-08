#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
PY="$("$PY" -c 'import sys; print(sys.executable)')"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PCI_BASE="$ROOT/experiments/PCI_AAMAS27_Rebuild_v1"
"$PY" "$ROOT/scripts/verify_sources.py"
(cd "$PCI_BASE" && "$PY" -m pytest -q tests)
(cd "$ROOT/experiments/PCI_AAMAS27_Followup_v1" && "$PY" -m pytest -q tests)
printf '\nPython suites finished. Real cryptographic preflight is a separate operation.\n'
