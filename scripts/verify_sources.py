#!/usr/bin/env python3
"""Verify the versioned source export; no dependencies, network, or training."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ('PCI_AAMAS27_Rebuild_v1', 'PCI_AAMAS27_Followup_v1')
EXCLUDED_DIRS = {'__pycache__', '.pytest_cache', 'node_modules', '.venv',
                 '.toolchain', 'build', 'bin', 'vendor', 'setup', 'generated'}
EXPECTED_FOLLOWUP = 'ba8d57b73079993a77269e357094e4d7c23694b16a392f0ccc1be9dc4ca79b5e'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify() -> int:
    checked = {}
    errors = []
    for study in STUDIES:
        base = ROOT / 'experiments' / study
        manifest = json.loads((base / 'SOURCE_MANIFEST.json').read_text())
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError(f'Invalid manifest: {study}')
        for name, expected in manifest.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise ValueError(f'Unsafe manifest path: {name}')
            path = base / name
            if path.is_symlink() or not path.is_file():
                errors.append(f'{study}/{name}: absent or symlink')
                continue
            actual = digest(path)
            if actual != expected:
                errors.append(f'{study}/{name}: SHA-256 mismatch')
        dirs = ('pci_bench', 'tests', 'configs', 'scripts', 'docs', 'crypto') if 'Rebuild' in study else ('pci_followup', 'tests', 'docs')
        for directory in dirs:
            for path in (base / directory).rglob('*'):
                rel = path.relative_to(base)
                if set(rel.parts) & EXCLUDED_DIRS or not path.is_file():
                    continue
                # Installation produces this lock; the exact original is in the raw archive.
                if str(rel) == 'crypto/package-lock.json':
                    continue
                if str(rel) not in manifest:
                    errors.append(f'{study}/{rel}: unlisted source file')
        checked[study] = manifest
        print(f'{study}: {len(manifest)} listed source files checked')
    base = checked[STUDIES[0]]
    followup = checked[STUDIES[1]]
    scientific = dict(followup)
    scientific.update({'frozen_base/' + Path(name).name: value
                       for name, value in base.items()
                       if name.startswith('pci_bench/') and name.endswith('.py')})
    encoded = json.dumps(scientific, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != EXPECTED_FOLLOWUP:
        errors.append('Follow-up scientific fingerprint differs from the frozen run')
    if errors:
        for error in errors:
            print('ERROR:', error, file=sys.stderr)
        return 1
    print(f'FOLLOWUP_SCIENTIFIC_DIGEST_OK {actual}')
    print('SOURCE_EXPORT_VERIFIED; this is not a scientific or cryptographic result')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(verify())
    except (OSError, ValueError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
