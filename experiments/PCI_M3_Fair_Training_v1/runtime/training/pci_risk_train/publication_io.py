"""Held-descriptor publication for Linux training artifacts.

Growing artifacts have no pathname: O_TMPFILE is mandatory. Immutable bodies
are installed with no-replace linkat. Mutable metadata uses an already closed,
content-qualified promotion alias and an immutable intent. A resurrected alias
is ancillary only, and is accepted only with the exact registered bytes/hash.
These checks assume faithful Linux/filesystem operations; they are not a
power-loss theorem or Byzantine-storage protection.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

ALIAS_PURPOSE = 'Closed ancillary promotion alias; never authoritative checkpoint state'


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try: os.fsync(fd)
    finally: os.close(fd)


def _identity(s):
    return (s.st_dev, s.st_ino, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _sha_stream(stream):
    stream.seek(0); h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    stream.seek(0); return h.hexdigest()


def _same_named(stream, path, baseline):
    current = os.fstat(stream.fileno()); named = Path(path).lstat()
    if _identity(current) != _identity(baseline) or not stat.S_ISREG(named.st_mode):
        raise RuntimeError('Held publication inode changed: ' + str(path))
    if (current.st_dev, current.st_ino) != (named.st_dev, named.st_ino):
        raise RuntimeError('Publication pathname changed inode: ' + str(path))


@contextmanager
def held_existing(path):
    """One non-symlink FD/baseline across validation, hashing and directory sync."""
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as stream:
        baseline = os.fstat(stream.fileno())
        if not stat.S_ISREG(baseline.st_mode): raise RuntimeError('Regular artifact required')
        _same_named(stream, path, baseline)
        try: yield stream
        finally: _same_named(stream, path, baseline)


def verified_digest(path, *, expected=None, expected_bytes=None):
    with held_existing(path) as stream:
        if expected_bytes is not None and os.fstat(stream.fileno()).st_size != expected_bytes:
            raise RuntimeError('Retained artifact byte count differs: ' + str(path))
        result = _sha_stream(stream)
        if expected is not None and result != expected:
            raise RuntimeError('Retained artifact digest differs: ' + str(path))
        fsync_directory(Path(path).parent)
        return result


def load_json_verified(path, *, expected=None):
    with held_existing(path) as stream:
        data = stream.read(); result = hashlib.sha256(data).hexdigest()
        if expected is not None and result != expected: raise RuntimeError('JSON digest differs')
        value = json.loads(data); fsync_directory(Path(path).parent)
        return value, result


def load_torch_verified(path, *, expected=None):
    import torch
    with held_existing(path) as stream:
        result = _sha_stream(stream)
        if expected is not None and result != expected: raise RuntimeError('Checkpoint digest differs')
        value = torch.load(stream, map_location='cpu', weights_only=False)
        if _sha_stream(stream) != result: raise RuntimeError('Checkpoint changed during decode')
        fsync_directory(Path(path).parent)
        return value, result


def _anonymous_fd(directory):
    if not hasattr(os, 'O_TMPFILE'): raise RuntimeError('O_TMPFILE required; no named growing-file fallback')
    try: return os.open(directory, os.O_TMPFILE | os.O_RDWR | os.O_CLOEXEC, 0o600)
    except OSError as e: raise RuntimeError('Anonymous publication unsupported on target filesystem') from e


def _link_anonymous(fd, path):
    """Publish the held inode without requiring CAP_DAC_READ_SEARCH.

    Linux documents /proc/self/fd plus AT_SYMLINK_FOLLOW as the unprivileged
    O_TMPFILE publication route. The source remains an open anonymous inode;
    linkat still refuses an occupied destination. No named growing file is used.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    libc.linkat.restype = ctypes.c_int
    source = os.fsencode('/proc/self/fd/' + str(fd))
    if libc.linkat(-100, source, -100, os.fsencode(os.path.abspath(path)), 0x400) != 0:
        number = ctypes.get_errno()
        if number == 17: raise FileExistsError(number, os.strerror(number), str(path))
        raise RuntimeError('Anonymous no-replace linkat unsupported: ' + os.strerror(number))


def _rename_noreplace(source, destination):
    """Preserve both names on collision; never overwrite quarantine evidence."""
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, 'renameat2'):
        raise RuntimeError('Linux renameat2 required for no-replace quarantine')
    libc.renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                              ctypes.c_char_p, ctypes.c_uint]
    libc.renameat2.restype = ctypes.c_int
    if libc.renameat2(-100, os.fsencode(os.path.abspath(source)), -100,
                      os.fsencode(os.path.abspath(destination)), 1) != 0:
        number = ctypes.get_errno()
        if number == 17: raise FileExistsError(number, os.strerror(number), str(destination))
        raise RuntimeError('No-replace quarantine rename unsupported: ' + os.strerror(number))


def quarantine_existing(source, directory):
    """Hash one held inode and move it without replacing any retained body.

    A matching occupied content-qualified name is retained. A resurrected
    duplicate is moved to a separate no-replace duplicate name; mismatching
    existing evidence fails closed with both original paths untouched.
    """
    source = Path(source); directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, 'rb') as stream:
        baseline = os.fstat(stream.fileno())
        if not stat.S_ISREG(baseline.st_mode): raise RuntimeError('Regular orphan required')
        _same_named(stream, source, baseline); digest = _sha_stream(stream)
        _same_named(stream, source, baseline)
        destination = directory / (source.name + '.' + digest)
        try:
            _rename_noreplace(source, destination)
        except FileExistsError:
            with held_existing(destination) as previous:
                if _sha_stream(previous) != digest:
                    raise RuntimeError('Occupied quarantine destination has different retained bytes')
                fsync_directory(directory)
            # Preserve the matching prior body as well as this duplicate. The
            # no-replace syscall closes the name-allocation race.
            for duplicate in range(1, 10001):
                candidate = destination.with_name(destination.name + f'.duplicate_{duplicate:06}')
                _same_named(stream, source, baseline)
                try: _rename_noreplace(source, candidate)
                except FileExistsError: continue
                destination = candidate; break
            else: raise RuntimeError('Quarantine duplicate namespace exhausted')
        fsync_directory(source.parent); fsync_directory(directory)
        current = os.fstat(stream.fileno())
        if (current.st_dev, current.st_ino, current.st_nlink, current.st_size, current.st_mtime_ns) != (
                baseline.st_dev, baseline.st_ino, baseline.st_nlink, baseline.st_size, baseline.st_mtime_ns):
            raise RuntimeError('Orphan inode changed during quarantine')
        _check_published(stream, destination, digest)
        return destination, digest


def _readback(stream, validator):
    baseline = os.fstat(stream.fileno()); stream.seek(0); validator(stream)
    result = _sha_stream(stream)
    if _identity(os.fstat(stream.fileno())) != _identity(baseline):
        raise RuntimeError('Anonymous artifact changed during readback')
    return result, baseline.st_size


def _check_published(stream, path, expected):
    baseline = os.fstat(stream.fileno())
    _same_named(stream, path, baseline)
    if _sha_stream(stream) != expected: raise RuntimeError('Published body differs from readback')
    fsync_directory(Path(path).parent); _same_named(stream, path, baseline)


def _existing(path, validator):
    with held_existing(path) as stream:
        stream.seek(0); validator(stream); result = _sha_stream(stream)
        fsync_directory(Path(path).parent)
        return result


def _immutable(path, writer, validator, *, verify_existing=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if not verify_existing: raise FileExistsError('Immutable artifact exists: ' + str(path))
        return _existing(path, validator)
    with os.fdopen(_anonymous_fd(path.parent), 'w+b') as stream:
        writer(stream); stream.flush(); os.fsync(stream.fileno())
        result, _ = _readback(stream, validator)
        try: _link_anonymous(stream.fileno(), path)
        except FileExistsError:
            if not verify_existing: raise
            return _existing(path, validator)
        fsync_directory(path.parent); _check_published(stream, path, result)
        return result


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _bytes_validator(expected):
    def validate(stream):
        stream.seek(0)
        if stream.read() != expected: raise RuntimeError('Closed artifact content differs')
    return validate


def immutable_json(path, value, *, verify_existing=True):
    data = _json_bytes(value)
    return _immutable(path, lambda s: s.write(data), _bytes_validator(data), verify_existing=verify_existing)


def immutable_npz(path, arrays, *, verify_existing=False):
    import numpy as np
    arrays = {name: np.asarray(array) for name, array in arrays.items()}
    if any(a.dtype.hasobject for a in arrays.values()): raise ValueError('Object arrays cannot be retained')
    def validate(stream):
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(old[k].dtype != a.dtype or old[k].shape != a.shape or
                    not np.array_equal(old[k], a, equal_nan=False) for k, a in arrays.items()):
                raise RuntimeError('Closed NPZ arrays differ')
    return _immutable(path, lambda s: np.savez_compressed(s, **arrays), validate, verify_existing=verify_existing)


def immutable_torch(path, value):
    import torch
    def validate(stream):
        stream.seek(0); restored = torch.load(stream, map_location='cpu', weights_only=False)
        if isinstance(value, dict):
            if not isinstance(restored, dict) or set(restored) != set(value):
                raise RuntimeError('Checkpoint root schema changed during serialization')
            for name in ('signature', 'update', 'step'):
                if name in value and restored[name] != value[name]:
                    raise RuntimeError('Checkpoint generation changed during serialization')
            original_model = value.get('model', value)
            restored_model = restored.get('model', restored)
            if original_model and all(isinstance(v, torch.Tensor) for v in original_model.values()):
                if any(not torch.equal(v.detach().cpu(), restored_model[k]) for k, v in original_model.items()):
                    raise RuntimeError('Checkpoint tensors differ after serialization')
    return _immutable(path, lambda s: torch.save(value, s), validate, verify_existing=False)


def _mutable_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    validator = _bytes_validator(data); desired = hashlib.sha256(data).hexdigest()
    if path.exists() or path.is_symlink():
        with held_existing(path) as previous:
            if previous.read() == data:
                fsync_directory(path.parent); return desired
    namespace = path.parent / '.publication'
    intents = namespace / 'intents'; aliases = namespace / 'closed'
    intents.mkdir(parents=True, exist_ok=True); aliases.mkdir(exist_ok=True)
    target = hashlib.sha256(path.name.encode()).hexdigest()
    basename = target + '.' + desired
    alias = aliases / (basename + '.ready')
    intent = {'schema': 'PCI-CLOSED-MUTABLE-PROMOTION-INTENT-v1', 'target_name': path.name,
              'closed_alias_name': alias.name, 'sha256': desired, 'bytes': len(data),
              'purpose': ALIAS_PURPOSE}
    immutable_json(intents / (basename + '.json'), intent, verify_existing=True)
    with os.fdopen(_anonymous_fd(path.parent), 'w+b') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno()); _readback(stream, validator)
        try:
            _link_anonymous(stream.fileno(), alias)
            fsync_directory(aliases); _check_published(stream, alias, desired)
        except FileExistsError:
            # A specifically registered alias can be an interrupted promotion or
            # a resurrected complete body. Hold that inode across promotion.
            descriptor = os.open(alias, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            with os.fdopen(descriptor, 'rb') as existing:
                baseline = os.fstat(existing.fileno()); _same_named(existing, alias, baseline)
                validator(existing); fsync_directory(aliases); _same_named(existing, alias, baseline)
                os.replace(alias, path); fsync_directory(aliases); fsync_directory(path.parent)
                _check_published(existing, path, desired)
        else:
            os.replace(alias, path); fsync_directory(aliases); fsync_directory(path.parent)
            _check_published(stream, path, desired)
    return desired


def mutable_json(path, value):
    return _mutable_bytes(path, _json_bytes(value))


def mutable_csv(path, rows, fields=None):
    if not rows and fields is None: raise ValueError('Schema-less empty CSV is forbidden')
    fields = list(fields or rows[0].keys()); buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='raise'); writer.writeheader()
    for row in rows:
        if set(row) != set(fields): raise ValueError('CSV schema changed')
        writer.writerow(row)
    return _mutable_bytes(path, buffer.getvalue().encode())


def publication_inventory(root):
    """Validate specifically registered ancillary aliases; unknown ones fail.

    The resulting optional map is bound by CLOSED_FILES. Required scientific
    artifacts and immutable intents remain mandatory. Aliases are never read
    as checkpoint pointers and can be absent or reappear only with exact bytes.
    """
    root = Path(root); optional = {}; present = set()
    for directory in root.rglob('.publication'):
        if not directory.is_dir() or directory.is_symlink(): raise RuntimeError('Invalid publication namespace')
        intents = directory / 'intents'; aliases = directory / 'closed'; known = set()
        for path in intents.iterdir():
            if not path.name.endswith('.json'): raise RuntimeError('Unknown publication-intent member')
            value, _ = load_json_verified(path)
            if set(value) != {'schema', 'target_name', 'closed_alias_name', 'sha256', 'bytes', 'purpose'} or value['schema'] != 'PCI-CLOSED-MUTABLE-PROMOTION-INTENT-v1':
                raise RuntimeError('Invalid ancillary promotion intent')
            target = value['target_name']; digest = value['sha256']
            if Path(target).name != target or target in ('', '.', '..') or not re.fullmatch(r'[0-9a-f]{64}', digest) or value['purpose'] != ALIAS_PURPOSE:
                raise RuntimeError('Nonlocal promotion intent')
            prefix = hashlib.sha256(target.encode()).hexdigest() + '.' + digest
            if path.name != prefix + '.json' or value['closed_alias_name'] != prefix + '.ready' or type(value['bytes']) is not int or value['bytes'] < 0:
                raise RuntimeError('Promotion intent identity mismatch')
            alias = aliases / value['closed_alias_name']; name = alias.relative_to(root).as_posix()
            item = {'bytes': value['bytes'], 'sha256': digest, 'publication_intent': path.relative_to(root).as_posix(),
                    'role': 'optional closed ancillary alias, never scientific evidence or authoritative state'}
            optional[name] = item; known.add(alias.name)
            if alias.exists() or alias.is_symlink():
                verified_digest(alias, expected=digest, expected_bytes=value['bytes']); present.add(name)
        for alias in aliases.iterdir():
            if alias.name not in known: raise RuntimeError('Unregistered publication alias: ' + str(alias))
        for entry in directory.iterdir():
            if entry.name not in ('intents', 'closed'): raise RuntimeError('Unknown publication namespace member')
    return optional, present


def verify_optional_aliases(root, optional):
    expected, present = publication_inventory(root)
    if expected != optional: raise RuntimeError('Bound optional alias inventory differs from immutable intents')
    return present
