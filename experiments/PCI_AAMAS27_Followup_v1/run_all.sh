#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PCI_BASE="${PCI_BASE:-/home/ubuntu/PCI_AAMAS27_Rebuild_v1}"
export PATH="$PCI_BASE/.toolchain/node/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$ROOT/.venv/bin/python" -u -m pci_followup.suite "$@"
