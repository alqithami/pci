#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [ -n "${PCI_PYTHON:-}" ]; then PY="$PCI_PYTHON";
elif [ -x .venv/bin/python ]; then PY="$ROOT/.venv/bin/python";
else PY=python3; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$PY" -u -m pci_bench.suite "$@"
