"""Strict identities and closed-file operations used by the new evaluation."""
from pathlib import Path
import hashlib, json, os, re

class EvaluationError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise EvaluationError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read_json(path):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'DUPLICATE_JSON_KEY:' + k)
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(EvaluationError('NONFINITE_JSON:' + x)))

def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

def local(root, relative):
    require(type(relative) is str and bool(relative), 'RELATIVE_PATH_REQUIRED')
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in relative, 'UNSAFE_RELATIVE_PATH')
    path = (Path(root) / p).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'PATH_ESCAPE')
    return path

def hash_string(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'SHA256_REQUIRED')
    return value
