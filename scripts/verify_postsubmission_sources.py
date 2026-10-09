#!/usr/bin/env python3
"""Read-only source/input checks; no network, imports of experiment code, or launch."""
from __future__ import annotations
import ast
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    'experiments/PCI_Postsubmission_Certified_v1':
        ('b7e1d5c5941263c3dd997080115be745ca6916e8744cf3659d284fbd0dd0e11f', 6),
    'experiments/PCI_Blinded_Action_Prototype_v2':
        ('08f7564b5ccd3574385d26c8e8c7d1d714fbff918b3977f71e1b8249644ce3e8', 4),
    'analysis/nominal_sensitivity_20261009':
        ('8d0931fdc9f9cc323335c4e50a7bc5e5dbbcc786fa5d189f626e130301f4769b', 5),
}
COUNT_DIGEST = '91c68f761cdc483beddf1622b36e70e91fca91314557126af0a25baf77531524'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(root: Path = ROOT) -> int:
    count = 0
    for directory, (manifest_digest, expected_count) in MANIFESTS.items():
        base = root / directory
        manifest = base / 'SOURCE.sha256'
        require(digest(manifest) == manifest_digest, f'Manifest changed: {directory}')
        seen = set()
        for line in manifest.read_text(encoding='utf-8').splitlines():
            match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
            require(match is not None, f'Malformed manifest row: {directory}')
            expected, name = match.groups()
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name,
                    f'Unsafe path: {name}')
            require(name not in seen, f'Duplicate path: {name}')
            seen.add(name)
            path = base / name
            require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base.resolve()),
                    f'Missing or indirect input: {directory}/{name}')
            require(digest(path) == expected, f'Content mismatch: {directory}/{name}')
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            if path.suffix == '.json':
                json.loads(path.read_text(encoding='utf-8'))
            count += 1
        require(len(seen) == expected_count, f'Wrong manifest count: {directory}')
        print(f'{directory}: {len(seen)} payload identities verified')
    protocol = json.loads((root / 'experiments/PCI_Postsubmission_Certified_v1/protocol.json').read_text())
    cells = (len(protocol['methods']) * len(protocol['training_seeds']) *
             len(protocol['populations']) * len(protocol['regimes']) * protocol['episodes_per_cell'])
    require(cells == protocol['full_episodes'] == 270, 'Wrong episode matrix')
    transitions = cells * protocol['horizon']
    paired = cells * len(protocol['paired_steps'])
    require(transitions == protocol['certified_transitions'] == 69120, 'Wrong transition count')
    require(paired == protocol['paired_records'] == 6480, 'Wrong paired-record count')
    require(transitions + 3 * paired == protocol['backend_measurements'] == 88560, 'Wrong call count')
    require(transitions + paired == protocol['saved_snark_proofs'] == 75600, 'Wrong proof count')
    require(len(set(itertools.permutations(protocol['backends']))) == len(protocol['paired_steps']) == 24,
            'Wrong protocol-order design')
    data = json.loads((root / 'analysis/nominal_sensitivity_20261009/data/reanalysis/nominal_counts.json').read_text())
    encoded = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
    require(hashlib.sha256(encoded).hexdigest() == COUNT_DIGEST, 'Wrong canonical nominal counts')
    require(len(data) == 25 and all([r[0] for r in rows] == [101, 102, 103, 104, 105]
                                  for rows in data.values()), 'Wrong nominal replication matrix')
    print(f'POSTSUBMISSION_SOURCE_EXPORT_VERIFIED: {count} listed source/input files')
    print('This checks source identity, syntax and design arithmetic, not experiment completion or security.')
    return count


if __name__ == '__main__':
    try:
        verify()
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
