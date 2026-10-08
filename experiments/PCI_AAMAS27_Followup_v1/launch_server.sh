#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p validation
exec 9>validation/controller.lock
flock -n 9 || { echo 'Controller already running'; exit 2; }
state(){ printf '%s UTC=%s PID=%s\n' "$1" "$(date -u +%FT%TZ)" "$$" | tee validation/STATE.txt; }
finish(){ rc=$?; printf 'exit=%s UTC=%s\n' "$rc" "$(date -u +%FT%TZ)" > validation/EXIT.txt; if [ "$rc" -ne 0 ]; then state STOPPED_ON_ERROR; fi; }
trap finish EXIT
export PCI_BASE=/home/ubuntu/PCI_AAMAS27_Rebuild_v1
export PATH="$PCI_BASE/.toolchain/node/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$HOME/.local/bin/uv" pip freeze --python "$ROOT/.venv/bin/python" > requirements-lock.txt
.venv/bin/python - <<'PY_CHECK'
import hashlib,json
from pathlib import Path
for name,digest in json.loads(Path('SOURCE_MANIFEST.json').read_text()).items():
    p=Path(name)
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        raise SystemExit('Source transfer mismatch: '+name)
from pci_followup.io import versions
v=versions()
for name,want in {'torch':'2.10.0+cpu','numpy':'2.3.5','rware':'2.0.0','gymnasium':'1.2.1','networkx':'3.4.2'}.items():
    if v.get(name)!=want:raise SystemExit(f'Dependency mismatch {name}: {v.get(name)} != {want}')
from pci_followup import BASE
from pci_bench.io import source_hashes,canonical_hash,sha256
original=json.loads((BASE/'results/deadline/SUITE_COMPLETE.json').read_text())
if original.get('status')!='COMPLETE' or original.get('source_digest')!=canonical_hash(source_hashes(BASE)):
    raise SystemExit('Frozen original study source/completion mismatch')
protocol=json.loads(Path('protocol.json').read_text())
if sha256(BASE/'PCI_deadline_results.tar.gz')!=protocol['original_archive_sha256']:
    raise SystemExit('Original archive changed')
print('SOURCE_AND_DEPENDENCIES_VERIFIED',v,flush=True)
print('ORIGINAL_STUDY_PRESERVED_AND_VERIFIED',flush=True)
PY_CHECK
state SMOKE_ACTIVE
bash run_all.sh --smoke --jobs 2 --results results/smoke
state FULL_STUDY_ACTIVE
bash run_all.sh --jobs 12 --results results/full
state PACKAGING
.venv/bin/python -m pci_followup.package
sha256sum -c PCI_followup_results.tar.gz.sha256
state FOLLOWUP_COMPLETE_ARCHIVE_VERIFIED
