"""Independent float32 portable forward pass and checked Torch evaluation bridge.

Torch is imported lazily. This module can be inspected/tested in the executor
without installing or importing Torch. Actual evaluation samples from the CPU
Torch actor only after every state's independent portable forward check passes.
"""
from pathlib import Path
import importlib.util, sys
import numpy as np
from .common import require, sha

MODEL_SHA = 'e9a7ab8721d457b60f54fe577ab03841390a57af5bd85f55e83c9f0bbefa2665'
METHOD_MODEL = {'task': 'ppo_task', 'fixed': 'ppo_penalty', 'adaptive': 'ppo_penalty',
                'semantic': 'pci', 'consensus': 'consensus', 'directrisk': 'ppo_penalty'}
LOGIT_ATOL, PROBABILITY_ATOL = 2e-5, 2e-6
ACTOR_SHAPES = {f'heads.{h}.{layer}.{kind}': shape
    for h in range(3)
    for layer, w, b in ((0, (64, 46), (64,)), (2, (64, 64), (64,)), (4, (1, 64), (1,)))
    for kind, shape in (('weight', w), ('bias', b))}
CRITIC_SHAPES = {f'critic.{layer}.{kind}': shape
    for layer, w, b in ((0, (64, 230), (64,)), (2, (64, 64), (64,)), (4, (1, 64), (1,)))
    for kind, shape in (('weight', w), ('bias', b))}
SHAPES = ACTOR_SHAPES | CRITIC_SHAPES | {'masks': (3, 23)}
VIEW_INDICES = ((0,1,2,3,4,5,6,10,11,16,17,18,19,20,21,22),
                (0,1,2,3,4,9,10,11,16,17,18,19,20),
                (0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))

def load_export(path):
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == set(SHAPES), 'EXACT_EXPORTED_STATE_DICT_KEYS')
        arrays = {k: z[k].copy() for k in z.files}
    for k, shape in SHAPES.items():
        a = arrays[k]
        require(a.dtype == np.float32 and a.shape == shape and np.isfinite(a).all(), 'EXPORT_ARRAY:' + k)
    masks = np.zeros((3, 23), dtype=np.float32)
    for i, cols in enumerate(VIEW_INDICES):
        masks[i, list(cols)] = 1
    require(np.array_equal(arrays['masks'], masks), 'EXACT_FIXED_MASKS')
    return arrays

def portable_forward(arrays, features):
    x = np.asarray(features)
    require(x.dtype == np.float32 and x.shape == (4, 5, 23) and np.isfinite(x).all(), 'EVAL_FEATURES')
    heads = []
    for i in range(3):
        z = np.concatenate((x * arrays['masks'][i], np.zeros_like(x)), axis=-1)
        a = np.tanh(z @ arrays[f'heads.{i}.0.weight'].T + arrays[f'heads.{i}.0.bias'])
        b = np.tanh(a @ arrays[f'heads.{i}.2.weight'].T + arrays[f'heads.{i}.2.bias'])
        y = (b @ arrays[f'heads.{i}.4.weight'].T + arrays[f'heads.{i}.4.bias'])[..., 0]
        heads.append(y)
    logits = np.stack(heads, axis=-2).mean(axis=-2, dtype=np.float32)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    require(logits.dtype == p.dtype == np.float32 and np.isfinite(p).all(), 'PORTABLE_FINITE_F32')
    return logits, p

def sampling_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    require(p.shape == (4, 5) and np.isfinite(p).all() and (p >= 0).all(), 'SAMPLING_PROBABILITIES')
    sums = p.sum(axis=-1, keepdims=True)
    require((sums > 0).all(), 'NONEMPTY_PROBABILITY_SIMPLEX')
    return p / sums

def inverse_cdf_actions(probabilities, uniforms):
    p = sampling_probabilities(probabilities)
    u = np.asarray(uniforms, dtype=np.float64)
    require(u.shape == (4,) and np.isfinite(u).all() and (u >= 0).all() and (u < 1).all(), 'FOUR_UNIFORMS')
    cumulative = p.cumsum(axis=-1)
    cumulative[:, -1] = 1.0
    actions = np.asarray([np.searchsorted(cumulative[i], u[i], side='right') for i in range(4)], dtype=np.int64)
    require(((actions >= 0) & (actions < 5)).all(), 'ACTION_SAMPLING_RANGE')
    return actions

class CheckedTorchActor:
    """Authoritative CPU Torch probabilities; separate portable check per state."""
    def __init__(self, project_root, export_path, method):
        require(method in METHOD_MODEL, 'EVALUATION_METHOD')
        import torch
        root = Path(project_root).resolve()
        model_source = root / 'training/pci_risk_train/model.py'
        require(sha(model_source) == MODEL_SHA, 'IMMUTABLE_MODEL_SOURCE')
        if str(root / 'training') not in sys.path:
            sys.path.insert(0, str(root / 'training'))
        from pci_risk_train.model import ContextPolicy
        require(Path(sys.modules['pci_risk_train.model'].__file__).resolve() == model_source, 'MODEL_IMPORT_PATH')
        self.arrays = load_export(export_path)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        # Construction is numerical validation, not a scientific policy RNG stream.
        with torch.random.fork_rng(devices=[]):
            cpu_generator = torch.Generator(device='cpu')
            cpu_generator.manual_seed(860000900)
            torch.set_rng_state(cpu_generator.get_state())
            self.model = ContextPolicy(METHOD_MODEL[method], 64).cpu()
        state = {k: torch.from_numpy(v.copy()) for k, v in self.arrays.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.torch = torch
        self.checked_states = 0
        self.max_logit_abs_error = self.max_probability_abs_error = 0.0

    def probabilities(self, features):
        torch = self.torch
        with torch.no_grad():
            f = torch.as_tensor(np.asarray(features), dtype=torch.float32).unsqueeze(0)
            dist, _, _ = self.model(f)
            # Categorical.logits are log-softmax-normalized; raw head means are
            # reconstructed directly for the logit check without changing sampling.
            global_slot = torch.zeros_like(f)
            raw = torch.stack([head(torch.cat((f*self.model.masks[i], global_slot), dim=-1)).squeeze(-1)
                               for i, head in enumerate(self.model.heads)], dim=-2).mean(dim=-2)[0]
            logits = raw.cpu().numpy().copy()
            probs = dist.probs[0].cpu().numpy().copy()
        pl, pp = portable_forward(self.arrays, np.asarray(features, dtype=np.float32))
        le = float(np.max(np.abs(logits.astype(np.float64) - pl.astype(np.float64))))
        pe = float(np.max(np.abs(probs.astype(np.float64) - pp.astype(np.float64))))
        require(le <= LOGIT_ATOL and pe <= PROBABILITY_ATOL, 'TORCH_PORTABLE_FORWARD_PARITY')
        self.checked_states += 1
        self.max_logit_abs_error = max(self.max_logit_abs_error, le)
        self.max_probability_abs_error = max(self.max_probability_abs_error, pe)
        return logits, probs, {'logit_abs_error': le, 'probability_abs_error': pe}
