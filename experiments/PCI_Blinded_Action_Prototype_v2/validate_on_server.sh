#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${PCI_BASE:-/home/ubuntu/PCI_AAMAS27_Rebuild_v1}"
OUT="${PCI_BLINDED_OUT:-/home/ubuntu/PCI_Blinded_Action_v2}"
PY="${PCI_PYTHON:-/home/ubuntu/PCI_AAMAS27_Followup_v1/.venv/bin/python}"
export PATH="$BASE/.toolchain/node/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -x "$PY" ] || { echo "Missing Python environment: $PY" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "Refusing an existing destination: $OUT" >&2; exit 2; }
"$PY" "$HERE/prepare_blinded_variant.py" --base "$BASE" --out "$OUT"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "STOPPED_ON_ERROR exit=%s\n" "$rc" > "$OUT/validation/STATE.txt"; fi' EXIT
"$PY" "$HERE/check_host_variant.py" --variant "$OUT"
cd "$OUT"
"$PY" -m pci_bench.crypto_build --agents 2 > validation/build.log 2>&1
"$PY" -m pci_bench.crypto_check --out validation/real_crypto > validation/real_crypto.log 2>&1
"$PY" "$HERE/check_real_binding.py" --variant "$OUT" > validation/blinded_binding.log 2>&1
printf 'CRYPTO_SMOKE_COMPLETE_NOT_BENCHMARKED\n' | tee validation/STATE.txt
