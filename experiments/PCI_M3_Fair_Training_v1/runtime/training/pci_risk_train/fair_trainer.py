"""Executable, separately versioned fair institutional-learning controls.

This is an engineering candidate, not a completed experiment or a reproduction
of a named published shield-learning algorithm. Frozen M2 trainer.py is intact.
All arms use the same actor and two critics. The constrained arm has separate
reward/cost GAE, a pessimistic clipped cost surrogate, and projected dual ascent.
Shield-aware PPO uses the likelihood of the actual joint shielded policy.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from .baseline_controls import (clipped_policy_surrogates, component_gradient_diagnostics,
                                gradient_norm, normalize_reward_and_scale_cost,
                                projected_dual_update)
from .constrained_models import ConstrainedContextPolicy
from .environment import D, VIEW_INDICES
from .io import canonical_hash, sha256, system_meta
from . import publication_io as publication
from .model import compatibility_energy, gae
from .risk_estimators import estimate_product_risk

VERSION = 'PCI-FAIR-INSTITUTIONAL-PPO-ENGINEERING-v1'
METHODS = ('fixed', 'direct', 'lagrangian', 'shieldaware', 'compatibility', 'consensus')
DEFAULTS = dict(schema='PCI-FAIR-TRAIN-CONFIG-v1', domain='warehouse', n_agents=4,
    hidden=64, device='cpu', torch_threads=1, penalty_coef=.25, auxiliary_coef=.2,
    auxiliary_reference_n4=None,
    risk_estimator='exact', risk_samples=16, risk_baseline='loo', max_joint_actions=65536,
    risk_minibatch_groups=None,
    query_chunk_size=1024, dual_initial=.25, dual_lr=.05, dual_target=.005,
    dual_max=20., shield_policy='exact', shield_entropy_samples=1,
    shield_entropy_groups=None, shield_prefix_cache=True,
    learning_rate=3e-4, critic_learning_rate=3e-4, cost_critic_learning_rate=3e-4,
    update_epochs=4, minibatch_groups=128, gamma=.99, cost_gamma=.99,
    gae_lambda=.95, cost_gae_lambda=.95, clip_coef=.2, vf_coef=.5,
    cost_vf_coef=.5, entropy_coef=.02, max_grad_norm=.5, target_kl=.03,
    checkpoint_every_updates=1, keep_checkpoint_generations=2)


def atomic_json(path, value):
    return publication.mutable_json(path, value)


def fsync_directory(path):
    return publication.fsync_directory(path)


atomic_npz = publication.immutable_npz
atomic_torch = publication.immutable_torch
atomic_csv = publication.mutable_csv


def configuration(value):
    c = DEFAULTS | copy.deepcopy(value)
    required = ('run_id', 'run_role', 'method', 'profiles', 'env_profile_ids', 'env_seeds',
                'seed', 'init_seed', 'policy_seed', 'minibatch_seed', 'auxiliary_seed',
                'entropy_seed', 'probe_seed', 'total_steps', 'rollout_steps')
    if any(k not in c for k in required): raise ValueError('Missing explicit run configuration')
    if c['schema'] != DEFAULTS['schema'] or c['method'] not in METHODS:
        raise ValueError('Unknown schema or comparison method')
    if c['method'] not in ('direct', 'compatibility', 'consensus'):
        if 'auxiliary_coef' in value and value['auxiliary_coef'] != 0:
            raise ValueError('This method does not use an auxiliary coefficient')
        c['auxiliary_coef'] = 0.
    if c['method'] in ('lagrangian', 'shieldaware'):
        if 'penalty_coef' in value and value['penalty_coef'] != 0:
            raise ValueError('Constrained/shield-aware arms do not use shaped reward penalties')
        c['penalty_coef'] = 0.
    if c['domain'] not in ('warehouse', 'rware') or c['run_role'] not in ('engineering', 'development', 'confirmatory'):
        raise ValueError('Explicit domain and scientific role required')
    if c['run_role'] != 'engineering' and not re.fullmatch(r'[0-9a-f]{64}', c.get('protocol_sha256', '')):
        raise ValueError('Scientific collection requires a source-bound frozen protocol')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', c['run_id']): raise ValueError('Unsafe run_id')
    nenv = len(c['env_profile_ids'])
    if 'n_envs' in value and value['n_envs'] != nenv: raise ValueError('n_envs and profile slots differ')
    c['n_envs'] = nenv
    if nenv < 1 or len(c['env_seeds']) != nenv or len(set(c['env_seeds'])) != nenv:
        raise ValueError('Distinct explicit environment seeds required')
    if any(type(s) is not int or s < 0 for s in c['env_seeds']): raise ValueError('Invalid environment seed')
    if any(p not in c['profiles'] for p in c['env_profile_ids']): raise ValueError('Unknown profile')
    for k in ('seed', 'init_seed', 'policy_seed', 'minibatch_seed', 'auxiliary_seed', 'entropy_seed', 'probe_seed'):
        if type(c[k]) is not int or c[k] < 0: raise ValueError('Invalid explicit seed: ' + k)
    if c['seed'] != c['init_seed']: raise ValueError('Seed labels the initialization cluster')
    for k in ('n_agents', 'hidden', 'rollout_steps', 'total_steps', 'update_epochs', 'minibatch_groups', 'risk_samples'):
        if type(c[k]) is not int or c[k] < 1: raise ValueError('Invalid positive integer: ' + k)
    if c['n_agents'] > 32: raise ValueError('Operational environments support at most 32 agents')
    if c['total_steps'] % (nenv * c['rollout_steps']): raise ValueError('Complete rollout blocks required')
    for p in c['profiles'].values():
        if p.get('n_agents', c['n_agents']) != c['n_agents']: raise ValueError('Mixed agent dimensions')
        if c['domain'] == 'rware' and p.get('reward_mode', 'native') == 'potential' and not math.isclose(
                c['gamma'], p.get('shaping_gamma', .99), rel_tol=0, abs_tol=1e-15):
            raise ValueError('Potential-shaping discount must match the reward GAE discount')
        if c['run_role'] != 'engineering' and (c['total_steps'] // nenv) % p['horizon']:
            raise ValueError('Scientific run must finish complete episodes per environment')
    if c['checkpoint_every_updates'] != 1 or c['keep_checkpoint_generations'] < 2:
        raise ValueError('Commit every rollout and retain at least two checkpoint generations')
    if c['torch_threads'] != 1: raise ValueError('One native math thread required')
    for k in ('penalty_coef', 'auxiliary_coef', 'vf_coef', 'cost_vf_coef', 'entropy_coef'):
        if not math.isfinite(c[k]) or c[k] < 0: raise ValueError('Invalid coefficient: ' + k)
    if c['auxiliary_reference_n4'] is not None:
        ref = c['auxiliary_reference_n4']
        if not isinstance(ref, (int, float)) or not math.isfinite(ref) or ref < 0:
            raise ValueError('Invalid reference-N4 auxiliary coefficient')
        if c['method'] == 'direct' and not math.isclose(c['auxiliary_coef'], ref * 4 / c['n_agents'], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError('Actual auxiliary coefficient does not match declared N4 dimension transfer')
        if c['method'] != 'direct' and ref != 0:
            raise ValueError('Non-Direct arms cannot declare a nonzero Direct-risk reference coefficient')
    for k in ('learning_rate', 'critic_learning_rate', 'cost_critic_learning_rate', 'max_grad_norm', 'target_kl'):
        if not math.isfinite(c[k]) or c[k] <= 0: raise ValueError('Invalid positive parameter: ' + k)
    if c['vf_coef'] <= 0 or c['cost_vf_coef'] <= 0:
        raise ValueError('Both critics must be fitted in every comparison arm')
    for key in ('gamma', 'cost_gamma', 'gae_lambda', 'cost_gae_lambda'):
        if not math.isfinite(c[key]) or not 0 <= c[key] <= 1: raise ValueError('Invalid discount/GAE parameter: ' + key)
    if not 0 < c['clip_coef'] < 1: raise ValueError('Invalid PPO clip coefficient')
    if c['domain'] == 'rware' and c['method'] in ('compatibility', 'consensus'):
        raise ValueError('Context compatibility/effect bounds are not specified for native RWARE')
    if c['risk_estimator'] not in ('exact', 'score', 'conditional') or c['shield_policy'] not in ('exact', 'sequential'):
        raise ValueError('Unknown risk estimator or shield policy')
    if c['risk_minibatch_groups'] is not None:
        if type(c['risk_minibatch_groups']) is not int or c['risk_minibatch_groups'] < 1:
            raise ValueError('Risk state budget must be None (all) or a positive integer')
        if c['method'] == 'direct' and c['auxiliary_coef'] > 0 and c['risk_estimator'] == 'exact' and c['risk_minibatch_groups'] < c['minibatch_groups']:
            raise ValueError('State subsampling is restricted to sampled risk estimators')
    if c['shield_entropy_groups'] is not None:
        if type(c['shield_entropy_groups']) is not int or c['shield_entropy_groups'] < 1:
            raise ValueError('Shield entropy budget must be None (all) or a positive integer')
        if c['method'] == 'shieldaware' and c['shield_policy'] == 'exact' and c['shield_entropy_groups'] < c['minibatch_groups']:
            raise ValueError('Entropy state subsampling is restricted to the sequential shield')
    if type(c['shield_prefix_cache']) is not bool:
        raise ValueError('Explicit boolean prefix-cache control required')
    needs_joint_enumeration = ((c['method'] == 'direct' and c['auxiliary_coef'] > 0 and c['risk_estimator'] == 'exact') or
                               (c['method'] == 'shieldaware' and c['shield_policy'] == 'exact'))
    if needs_joint_enumeration and 5 ** c['n_agents'] > c['max_joint_actions']:
        raise ValueError('Declared exact-joint enumeration cap would be exceeded')
    projected_dual_update(c['dual_initial'], 0., c['dual_target'], c['dual_lr'], c['dual_max'])
    return c


def runtime_sources():
    root = Path(__file__).resolve().parents[1]
    # Bind the actual training import closure, not unrelated evaluator/controller
    # modules that are independently source-bound by their own receipts.
    names = ['pci_risk_train/' + name for name in (
        '__init__.py', 'baseline_controls.py', 'constrained_models.py', 'environment.py',
        'environment_adapters.py', 'fair_trainer.py', 'io.py', 'model.py',
        'publication_io.py', 'risk_estimators.py', 'rware_adapter.py', 'shield_policy.py')]
    return {name: sha256(root / name) for name in sorted(names)}


def fair_system_meta(domain):
    """Bind imported native-domain dependencies without changing frozen M2 io."""
    meta = system_meta(); meta['dependency_source_sha256'] = {}
    if domain == 'rware':
        for name in ('rware', 'gymnasium', 'networkx'):
            module = importlib.import_module(name); root = Path(module.__file__).resolve().parent
            distribution = importlib.metadata.distribution(name)
            if root != Path(distribution.locate_file(name)).resolve():
                raise RuntimeError('Imported native dependency does not match owning distribution: ' + name)
            declared = getattr(module, '__version__', None)
            version = str(declared if declared is not None else distribution.version)
            bodies = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*.py'))}
            meta['versions'][name] = version
            meta['imported_modules'][name] = {'version': version, 'file': str(module.__file__),
                'version_source': ('module.__version__' if declared is not None else
                                   'owning distribution metadata with matching imported package path'),
                'python_source_files': len(bodies), 'source_tree_sha256': canonical_hash(bodies)}
            meta['dependency_source_sha256'][name] = canonical_hash(bodies)
    return meta


def training_signature(c, sources, system):
    return canonical_hash({'configuration': c, 'sources': sources, 'versions': system['versions'],
                           'dependency_source_sha256': system.get('dependency_source_sha256', {})})


def _vector(model, *, actor=False):
    ps = model.heads.parameters() if actor else model.parameters()
    return torch.cat([p.detach().flatten().cpu() for p in ps])


def _closed_json(path, value):
    return publication.immutable_json(path, value, verify_existing=True)


def _write_retention(out, signature, keep, entries):
    atomic_json(out / 'CHECKPOINT_RETENTION.json', {
        'schema': 'PCI-FAIR-CHECKPOINT-RETENTION-v1', 'signature': signature,
        'keep_latest': keep, 'pruned': entries})


def _restore_retention(out, signature, keep, known_prefix, committed_update):
    """Reconcile postcommit prune intents without forgetting later journal rows."""
    path = out / 'CHECKPOINT_RETENTION.json'
    if not path.exists():
        if known_prefix: raise RuntimeError('Missing committed checkpoint-retention journal')
        return []
    journal, _ = publication.load_json_verified(path)
    if journal.get('schema') != 'PCI-FAIR-CHECKPOINT-RETENTION-v1' or journal.get('signature') != signature or journal.get('keep_latest') != keep:
        raise RuntimeError('Checkpoint-retention journal identity changed')
    entries = journal['pruned']
    stable = lambda x: {k: v for k, v in x.items() if k != 'status'}
    if len(entries) < len(known_prefix) or any(stable(x) != stable(y) for x, y in zip(entries, known_prefix)):
        raise RuntimeError('Checkpoint-retention journal lost its checkpoint-known prefix')
    if any(y.get('status') == 'DELETED' and x.get('status') != 'DELETED' for x, y in zip(entries, known_prefix)):
        raise RuntimeError('Checkpoint-retention journal regressed a checkpoint-known deletion')
    seen = set()
    for entry in entries:
        name = entry.get('path', '')
        if not re.fullmatch(r'checkpoints/update_[0-9]{6}\.pt', name) or name in seen:
            raise RuntimeError('Invalid/duplicate prune member')
        seen.add(name); generation = int(Path(name).stem.split('_')[1])
        requested = entry.get('requested_after_update')
        if (type(requested) is not int or not generation <= requested - keep or
                requested > committed_update or not re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', ''))):
            raise RuntimeError('Prune intent is outside its committed retention boundary')
        member = out / name
        if entry.get('status') == 'DELETE_INTENT':
            if member.exists():
                publication.verified_digest(member, expected=entry['sha256'])
                member.unlink(); fsync_directory(member.parent)
            # Absence after a durable intent also covers a kill after unlink but
            # before the completion receipt. No new scientific state is inferred.
            entry['status'] = 'DELETED'; _write_retention(out, signature, keep, entries)
        elif entry.get('status') == 'DELETED':
            if member.exists(): raise RuntimeError('A journaled pruned checkpoint reappeared')
        else: raise RuntimeError('Unknown checkpoint-retention state')
    return entries


def _shield(dist, oracle, c):
    from .shield_policy import ExactJointShield, SequentialSafeShield
    cls = ExactJointShield if c['shield_policy'] == 'exact' else SequentialSafeShield
    return cls(dist.logits, oracle)


def _native_evaluation(oracle, actions):
    result = oracle.evaluate(actions[:, None, :])
    aliases = {'physical_replacements': ('physical_replacements', 'physical_counts'),
               'institutional_replacements': ('institutional_replacements', 'institutional_counts'),
               'total_replacements': ('total_replacements', 'total_counts'),
               'admitted_actions': ('admitted_actions', 'admitted')}
    selected = {}
    for key, alternatives in aliases.items():
        actual = next((name for name in alternatives if name in result), None)
        if actual is None: raise RuntimeError('Oracle evaluation missing ' + key)
        array = result[actual]
        if hasattr(array, 'detach'): array = array.detach().cpu().numpy()
        selected[key] = np.asarray(array)[:, 0]
    return selected


def train(value, out, *, stop_after_updates=None):
    from .environment_adapters import make_environment, make_oracle, snapshot_oracle
    c = configuration(value); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if stop_after_updates is not None and (c['run_role'] != 'engineering' or stop_after_updates < 1):
        raise ValueError('Early stopping is restricted to engineering recovery fixtures')
    lock = (out / 'RUN.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = runtime_sources(); system = fair_system_meta(c['domain'])
    signature = training_signature(c, sources, system)
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != c:
        raise RuntimeError('Refusing mixed configurations')
    if (out / 'TRAIN_COMPLETE.json').exists():
        done, _ = publication.load_json_verified(out / 'TRAIN_COMPLETE.json')
        if done['signature'] != signature: raise RuntimeError('Completed run identity changed')
        manifest, _ = publication.load_json_verified(out / 'CLOSED_FILES.json', expected=done['closed_manifest_sha256'])
        for name, item in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise RuntimeError('Retained artifact changed: ' + name)
            try: publication.verified_digest(out / name, expected=item['sha256'], expected_bytes=item['bytes'])
            except RuntimeError as error: raise RuntimeError('Retained artifact changed: ' + name) from error
        optional = manifest.get('optional_publication_aliases', {})
        publication.verify_optional_aliases(out, optional)
        actual = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
        allowed = set(manifest['files']) | set(optional) | {'RUN.lock', 'TRAIN_COMPLETE.json', 'CLOSED_FILES.json'}
        if actual - allowed: raise RuntimeError('Unclosed training artifact: ' + str(sorted(actual - allowed)))
        lock.close(); return done
    atomic_json(out / 'config.json', c)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    random.seed(c['init_seed']); np.random.seed(c['init_seed']); torch.manual_seed(c['init_seed'])
    device = torch.device(c['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise RuntimeError('Deterministic CUBLAS workspace required before CUDA use')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    envs = [make_environment(c['domain'], c['profiles'][p], int(seed))
            for p, seed in zip(c['env_profile_ids'], c['env_seeds'])]
    if any(e.n != c['n_agents'] for e in envs): raise RuntimeError('Environment dimension mismatch')
    obs = np.stack([e.features() for e in envs])
    feature_spec = getattr(envs[0], 'feature_spec', {})
    if callable(feature_spec): feature_spec = feature_spec()
    view_indices = feature_spec.get('view_indices', feature_spec.get('context_view_indices'))
    if c['domain'] == 'warehouse' and view_indices is None: view_indices = VIEW_INDICES
    model = ConstrainedContextPolicy(c['hidden'], feature_dim=obs.shape[-1],
        action_count=obs.shape[-2], view_indices=view_indices).to(device)
    optimizer = torch.optim.Adam([
        {'params': list(model.heads.parameters()), 'lr': c['learning_rate']},
        {'params': list(model.reward_critic.parameters()), 'lr': c['critic_learning_rate']},
        {'params': list(model.cost_critic.parameters()), 'lr': c['cost_critic_learning_rate']}], eps=1e-5)
    generators = {name: torch.Generator(device=device).manual_seed(c[name + '_seed'])
                  for name in ('policy', 'minibatch', 'auxiliary', 'entropy', 'probe')}
    initial = _vector(model); initial_actor = _vector(model, actor=True)
    dual = float(c['dual_initial']); step = 0; start_update = 0; elapsed_before = 0.
    updates = []; episodes = []; env_sums = [dict() for _ in envs]; closed = {}; pruned = []
    phase_counts = {p: [0] * int(c['profiles'][p].get('quota_window', 64)) for p in c['profiles']}
    counters = dict(optimizer_steps=0, proposed_agent_actions=0, native_oracle_matches=0,
        risk_requested_joint_queries=0, risk_callback_calls=0, deployment_shield_uses=0,
        collection_oracle_requested_rows=0, collection_oracle_callback_calls=0,
        optimization_oracle_requested_rows=0, optimization_oracle_callback_calls=0,
        collection_oracle_evaluated_rows=0, optimization_oracle_evaluated_rows=0,
        optimization_cache_hit_rows=0, optimization_precompute_rows=0,
        collection_certified_zero_rows=0, optimization_certified_zero_rows=0)
    digest = '0' * 64; commit_path = out / 'LAST_COMMIT.json'
    if commit_path.exists():
        commit, _ = publication.load_json_verified(commit_path)
        if not re.fullmatch(r'checkpoints/update_[0-9]{6}\.pt', commit.get('checkpoint', '')):
            raise RuntimeError('Nonlocal checkpoint generation path')
        checkpoint = out / commit['checkpoint']
        if commit['signature'] != signature: raise RuntimeError('Committed checkpoint identity changed')
        ck, _ = publication.load_torch_verified(checkpoint, expected=commit['checkpoint_sha256'])
        if ck['signature'] != signature or ck['update'] != commit['update']:
            raise RuntimeError('Checkpoint and commit differ')
        for name, digest0 in ck['closed_files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise RuntimeError('Committed evidence changed: ' + name)
            publication.verified_digest(out / name, expected=digest0)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        for e, state in zip(envs, ck['envs']): e.load_state_dict(state)
        obs = ck['obs']; step = ck['step']; start_update = ck['update']; dual = ck['dual']
        updates = ck['updates']; episodes = ck['episodes']; env_sums = ck['env_sums']
        closed = ck['closed_files']; counters = ck['counters']; digest = ck['action_digest']
        phase_counts = ck['phase_counts']
        initial = ck['initial_vector'].cpu(); initial_actor = ck['initial_actor_vector'].cpu()
        elapsed_before = ck['elapsed']
        pruned = _restore_retention(out, signature, c['keep_checkpoint_generations'],
                                    ck.get('pruned_checkpoints', []), ck['update'])
        for name, generator in generators.items(): generator.set_state(ck['generators'][name].cpu())
        torch.set_rng_state(ck['torch_rng'].cpu()); np.random.set_state(ck['numpy_rng']); random.setstate(ck['python_rng'])
        if device.type == 'cuda': torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
    else:
        if (out / 'initial_policy.pt').exists():
            old, initial_hash = publication.load_torch_verified(out / 'initial_policy.pt')
            if any(not torch.equal(old[k], v.cpu()) for k, v in model.state_dict().items()):
                raise RuntimeError('Uncommitted initial policy differs')
        else: initial_hash = atomic_torch(out / 'initial_policy.pt', model.state_dict())
        closed['initial_policy.pt'] = initial_hash
    start = time.perf_counter()
    atomic_json(out / 'provenance.json', {'version': VERSION, 'signature': signature,
        'sources': sources, 'system': system, 'model': model.specification(),
        'cost_definition': 'institutional stage Hamming replacements divided once by N',
        'common_monitor': True, 'two_critics_instantiated_and_fitted_in_every_arm': True,
        'shield_likelihood': 'joint deployed policy, not corrected-action product-policy probability',
        'shield_loss_units': 'Joint reward/cost surrogates and joint entropy divided by N; approximate joint KL divided by N; clipping remains on joint q ratio',
        'sampled_risk_state_mean': 'Uniform subset mean without B/M multiplication; independent auxiliary RNG',
        'auxiliary_units': 'lambda times joint-state mean Jinst; no additional per-agent division',
        'actual_auxiliary_coefficient': c['auxiliary_coef'],
        'auxiliary_reference_n4': c['auxiliary_reference_n4'],
        'auxiliary_reference_ratio': 4 / c['n_agents'] if c['auxiliary_reference_n4'] is not None else None,
        'sampled_entropy_state_mean': 'Uniform subset mean without B/M multiplication; independent entropy RNG',
        'shield_support': ('full valid joint-action support' if c['shield_policy'] == 'exact'
                           else 'sequential valid future-stay completion, restricted support'),
        'published_algorithm_replication_claim': False,
        'prior_M2_difference': 'Extra cost critic fitted in all arms; scientific identity and budgets separately versioned'})
    total_updates = c['total_steps'] // (c['n_envs'] * c['rollout_steps'])
    for update in range(start_update, total_updates):
        before = _vector(model); actor_before = _vector(model, actor=True)
        data = {k: [] for k in ('obs', 'actions', 'old_logp', 'reward_values', 'cost_values',
            'rewards', 'costs', 'dones', 'admitted_actions', 'physical_replacements',
            'institutional_replacements', 'total_replacements', 'task_rewards', 'product_probe_costs',
            'admitted_native_actions')}
        snapshots = []; costs = []; tasks = []; optimization = []
        penalty = c['penalty_coef'] if c['method'] in ('fixed', 'direct', 'compatibility', 'consensus') else 0.
        for _ in range(c['rollout_steps']):
            ob = torch.as_tensor(obs, device=device)
            snapshots.extend([snapshot_oracle(e) for e in envs])
            oracle = make_oracle(envs, domain=c['domain'])
            with torch.no_grad():
                dist, rv, cv, _ = model(ob)
                if c['method'] == 'shieldaware':
                    policy = _shield(dist, oracle, c); actions = policy.sample(generator=generators['policy'])
                    old_logp = policy.log_prob(actions); counters['deployment_shield_uses'] += c['n_envs']
                else:
                    actions = torch.multinomial(dist.probs.reshape(-1, model.action_count), 1,
                        generator=generators['policy']).reshape(c['n_envs'], c['n_agents'])
                    old_logp = dist.log_prob(actions)
                probe = torch.multinomial(dist.probs.reshape(-1, model.action_count), 1,
                    generator=generators['probe']).reshape(c['n_envs'], c['n_agents'])
                probe_cost = oracle(probe[:, None, :])[:, 0]
            batch_actions = actions.cpu().numpy(); evaluated = _native_evaluation(oracle, batch_actions)
            rewards = []; dones = []; next_obs = []; native_costs = []; native_tasks = []; native_commands = []
            for i, env in enumerate(envs):
                phase_counts[c['env_profile_ids'][i]][int(env.t) % int(env.config.quota_window)] += 1
                no, task, _legacy_cost, done, metrics = env.transition(batch_actions[i], enforce=True)
                admitted = np.asarray(env.last.get('canonical_admitted_actions', env.last['actions']))
                native_commands.append(np.asarray(env.last.get('admitted_actions', env.last['actions'])))
                if not np.array_equal(admitted, evaluated['admitted_actions'][i]):
                    raise RuntimeError('Static oracle/native admitted action mismatch')
                inst = int(evaluated['institutional_replacements'][i]); cost = inst / env.n
                if not 0 <= cost <= 1: raise RuntimeError('Institutional cost outside normalized range')
                if c['method'] == 'shieldaware' and int(evaluated['total_replacements'][i]) != 0:
                    raise RuntimeError('Shielded proposal was corrected by authoritative native monitor')
                shared = float(np.mean(task)); rewards.append(np.full(env.n, shared - penalty * cost, dtype=np.float32))
                native_costs.append(cost); native_tasks.append(shared); dones.append(float(done))
                sums = env_sums[i]
                for key, val in metrics.items():
                    if key not in ('physical_replacements', 'institutional_replacements', 'total_replacements') and np.isscalar(val) and isinstance(val, (int, float, np.number)):
                        sums[key] = sums.get(key, 0.) + float(val)
                for key in ('physical_replacements', 'institutional_replacements', 'total_replacements'):
                    sums[key] = sums.get(key, 0.) + int(evaluated[key][i])
                if done:
                    episodes.append({'profile': c['env_profile_ids'][i], 'environment_index': i,
                        'episode': int(getattr(env, 'episode', sums.get('episode', 0))),
                        'joint_steps': int(env.config.horizon), 'training_step_at_end': step + i + 1,
                        'deliveries': sums.get('delivered', 0.), 'pickups': sums.get('pickups', 0.),
                        'task_return': sums.get('task_reward', 0.),
                        'physical_replacements': sums['physical_replacements'],
                        'institutional_replacements': sums['institutional_replacements'],
                        'total_replacements': sums['total_replacements']})
                    env_sums[i] = {}; no = env.reset()
                next_obs.append(no)
            for key, val in [('obs', ob), ('actions', actions), ('old_logp', old_logp),
                             ('reward_values', rv), ('cost_values', cv)]: data[key].append(val)
            data['rewards'].append(torch.as_tensor(np.asarray(rewards), device=device))
            data['costs'].append(torch.as_tensor(np.repeat(np.asarray(native_costs)[:, None], c['n_agents'], axis=1),
                                                device=device, dtype=torch.float32))
            data['dones'].append(torch.as_tensor(dones, device=device))
            data['admitted_native_actions'].append(np.asarray(native_commands))
            data['product_probe_costs'].append(np.asarray(probe_cost.detach().cpu() if hasattr(probe_cost, 'detach') else probe_cost))
            for key in ('admitted_actions', 'physical_replacements', 'institutional_replacements', 'total_replacements'):
                data[key].append(evaluated[key])
            data['task_rewards'].append(np.asarray(native_tasks))
            digest = hashlib.sha256(bytes.fromhex(digest) + batch_actions.astype('<i8').tobytes() +
                evaluated['admitted_actions'].astype('<i8').tobytes() +
                np.asarray(native_commands, dtype='<i8').tobytes() + np.asarray(native_costs, dtype='<f8').tobytes()).hexdigest()
            counters['collection_oracle_requested_rows'] += oracle.counters['requested_rows']
            counters['collection_oracle_callback_calls'] += oracle.counters['callback_calls']
            counters['collection_oracle_evaluated_rows'] += oracle.counters['evaluated_rows']
            counters['collection_certified_zero_rows'] += oracle.counters.get('certified_zero_rows', 0)
            counters['proposed_agent_actions'] += c['n_envs'] * c['n_agents']
            counters['native_oracle_matches'] += c['n_envs']; step += c['n_envs']
            obs = np.stack(next_obs); costs.extend(native_costs); tasks.extend(native_tasks)
        batch = {k: torch.stack(data[k]) for k in ('obs', 'actions', 'old_logp', 'reward_values',
                                                  'cost_values', 'rewards', 'costs', 'dones')}
        with torch.no_grad():
            _, last_reward, last_cost, _ = model(torch.as_tensor(obs, device=device))
        reward_adv, reward_ret = gae(batch['rewards'], batch['reward_values'], batch['dones'],
            last_reward, c['gamma'], c['gae_lambda'])
        cost_adv, cost_ret = gae(batch['costs'], batch['cost_values'], batch['dones'],
            last_cost, c['cost_gamma'], c['cost_gae_lambda'])
        norm_adv, scaled_cost_adv, reward_scale = normalize_reward_and_scale_cost(reward_adv, cost_adv)
        flat = {key: tensor.flatten(0, 1) for key, tensor in batch.items()}
        advantages = norm_adv.flatten(0, 1); cost_advantages = scaled_cost_adv.flatten(0, 1)
        reward_returns = reward_ret.flatten(0, 1); cost_returns = cost_ret.flatten(0, 1)
        all_oracle = make_oracle(snapshots, domain=c['domain']); n_groups = len(snapshots)
        if c['n_agents'] == 4 and ((c['method'] == 'direct' and c['auxiliary_coef'] > 0 and c['risk_estimator'] == 'exact') or
                                   (c['method'] == 'shieldaware' and c['shield_policy'] == 'exact')):
            from .environment_adapters import ExactTableOracleCache
            all_oracle = ExactTableOracleCache(all_oracle, include_admissibility=(c['method'] == 'shieldaware'))
        prepared_likelihood = None
        if c['method'] == 'shieldaware' and c['shield_policy'] == 'sequential' and c['shield_prefix_cache']:
            from .shield_policy import PreparedSequentialLikelihood
            prepared_likelihood = PreparedSequentialLikelihood(all_oracle, flat['actions'])
        for epoch in range(c['update_epochs']):
            order = torch.randperm(n_groups, device=device, generator=generators['minibatch'])
            stop = False
            for begin in range(0, n_groups, c['minibatch_groups']):
                idx = order[begin:begin + c['minibatch_groups']]; indices = idx.detach().cpu().tolist()
                oracle = all_oracle.subset(indices)
                oracle_before = copy.deepcopy(oracle.counters)
                dist, reward_value, cost_value, heads = model(flat['obs'][idx])
                entropy_indices = indices
                if c['method'] == 'shieldaware':
                    policy = _shield(dist, oracle, c)
                    new_logp = (prepared_likelihood.subset(indices).log_prob(dist.logits)
                                if prepared_likelihood is not None else policy.log_prob(flat['actions'][idx]))
                    logratio = new_logp - flat['old_logp'][idx]
                    ra, ca = advantages[idx].mean(-1), cost_advantages[idx].mean(-1)
                    if c['entropy_coef'] == 0:
                        entropy = dist.logits.new_zeros(()); entropy_indices = []
                    elif c['shield_policy'] == 'sequential':
                        budget = c['shield_entropy_groups']
                        if budget is not None and budget < len(indices):
                            chosen = torch.randperm(len(indices), device=device, generator=generators['entropy'])[:budget]
                            chosen_cpu = chosen.detach().cpu().tolist()
                            entropy_policy = _shield(torch.distributions.Categorical(logits=dist.logits[chosen]),
                                                     oracle.subset(chosen_cpu), c)
                            entropy_indices = [indices[j] for j in chosen_cpu]
                        else: entropy_policy = policy
                        entropy = entropy_policy.entropy(samples=c['shield_entropy_samples'],
                            generator=generators['entropy']).mean() / c['n_agents']
                    else: entropy = policy.entropy().mean() / c['n_agents']
                else:
                    logratio = dist.log_prob(flat['actions'][idx]) - flat['old_logp'][idx]
                    ra, ca = advantages[idx], cost_advantages[idx]; entropy = dist.entropy().mean()
                reward_policy, cost_policy = clipped_policy_surrogates(logratio, ra, ca, c['clip_coef'])
                if c['method'] == 'shieldaware':
                    reward_policy = reward_policy / c['n_agents']; cost_policy = cost_policy / c['n_agents']
                if c['method'] == 'lagrangian':
                    reward_policy = reward_policy / (1 + dual); cost_policy = dual * cost_policy / (1 + dual)
                else: cost_policy = cost_policy * 0.
                regularizer = reward_policy.new_zeros(()); estimate = None; risk_indices = []
                if c['method'] == 'direct' and c['auxiliary_coef'] > 0:
                    budget = c['risk_minibatch_groups']
                    if budget is not None and budget < len(indices):
                        selected = torch.randperm(len(indices), device=device, generator=generators['auxiliary'])[:budget]
                        selected_cpu = selected.detach().cpu().tolist()
                        risk_oracle = oracle.subset(selected_cpu); risk_probs = dist.probs[selected]
                        risk_indices = [indices[j] for j in selected_cpu]
                    else:
                        risk_oracle = oracle; risk_probs = dist.probs; risk_indices = indices
                    estimate = estimate_product_risk(risk_probs, risk_oracle, estimator=c['risk_estimator'],
                        samples=c['risk_samples'], generator=generators['auxiliary'],
                        baseline=c['risk_baseline'], max_joint_actions=c['max_joint_actions'],
                        query_chunk_size=c['query_chunk_size'])
                    regularizer = estimate.loss.mean()
                    counters['risk_requested_joint_queries'] += estimate.queries
                    counters['risk_callback_calls'] += estimate.callback_calls
                elif c['method'] in ('compatibility', 'consensus') and c['auxiliary_coef'] > 0:
                    if c['domain'] != 'warehouse': raise ValueError('Compatibility masks/effect bounds not specified for this domain')
                    mode = 'pci' if c['method'] == 'compatibility' else 'consensus'
                    regularizer = compatibility_energy(heads, flat['obs'][idx], mode=mode).mean()
                    regularizer = regularizer * (6 / 13 if mode == 'pci' else 5 / 2)
                components = {'reward_policy': reward_policy, 'cost_policy': cost_policy,
                    'auxiliary': c['auxiliary_coef'] * regularizer,
                    'entropy': -c['entropy_coef'] * entropy,
                    'reward_value': c['vf_coef'] * .5 * (reward_value - reward_returns[idx]).square().mean(),
                    'cost_value': c['cost_vf_coef'] * .5 * (cost_value - cost_returns[idx]).square().mean()}
                loss = sum(components.values())
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                diagnostics = component_gradient_diagnostics(components, model.parameters())
                optimizer.zero_grad(set_to_none=True); loss.backward()
                preclip = gradient_norm(model.parameters())
                if not math.isclose(preclip, diagnostics['summed_component_grad_norm'], rel_tol=2e-4, abs_tol=2e-6):
                    raise RuntimeError('Component gradients do not sum to actual full gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(), c['max_grad_norm'])
                clipped = gradient_norm(model.parameters())
                if not math.isfinite(preclip) or clipped > c['max_grad_norm'] * 1.0001:
                    raise RuntimeError('Nonfinite gradient or incorrect clipping')
                optimizer.step(); counters['optimizer_steps'] += 1
                kl_raw = float(((logratio.exp() - 1) - logratio).mean().detach().cpu())
                kl_denominator = c['n_agents'] if c['method'] == 'shieldaware' else 1
                kl = kl_raw / kl_denominator
                optimization.append({'epoch': epoch, 'minibatch_start': begin, 'indices': indices,
                    'loss': float(loss.detach().cpu()), 'reward_policy_loss': float(reward_policy.detach().cpu()),
                    'cost_policy_loss': float(cost_policy.detach().cpu()),
                    'auxiliary_value': float(regularizer.detach().cpu()), 'entropy_per_agent': float(entropy.detach().cpu()),
                    'preclip_grad_norm': preclip, 'clipped_grad_norm': clipped, 'approx_kl': kl,
                    'approx_kl_raw': kl_raw, 'approx_kl_denominator': kl_denominator,
                    'entropy_state_indices': entropy_indices, 'entropy_batch_size': len(entropy_indices),
                    'shield_entropy_configured_state_budget': c['shield_entropy_groups'],
                    'risk_state_indices': risk_indices, 'risk_batch_size': len(risk_indices),
                    'risk_configured_state_budget': c['risk_minibatch_groups'],
                    'risk_samples_per_state': c['risk_samples'] if estimate is not None else 0,
                    'risk_requested_joint_queries': estimate.queries if estimate is not None else 0,
                    'oracle_counters': {k: oracle.counters[k] - oracle_before.get(k, 0) for k in oracle.counters}, **diagnostics})
                if kl > c['target_kl']: stop = True; break
            if stop: break
        counters['optimization_oracle_requested_rows'] += all_oracle.counters['requested_rows']
        counters['optimization_oracle_callback_calls'] += all_oracle.counters['callback_calls']
        counters['optimization_oracle_evaluated_rows'] += all_oracle.counters['evaluated_rows']
        counters['optimization_cache_hit_rows'] += all_oracle.counters.get('cache_hit_rows', 0)
        counters['optimization_precompute_rows'] += all_oracle.counters.get('precompute_rows', 0)
        counters['optimization_certified_zero_rows'] += all_oracle.counters.get('certified_zero_rows', 0)
        dual_used = dual if c['method'] == 'lagrangian' else 0.
        if c['method'] == 'lagrangian':
            dual = projected_dual_update(dual, float(np.mean(costs)), c['dual_target'], c['dual_lr'], c['dual_max'])
        if not optimization: raise RuntimeError('No optimizer step')
        row = {'update': update + 1, 'joint_steps': step, 'agent_decisions': step * c['n_agents'],
            'method': c['method'], 'mean_task_reward': float(np.mean(tasks)),
            'mean_institutional_cost': float(np.mean(costs)),
            'mean_product_probe_cost': float(np.mean(np.stack(data['product_probe_costs']))),
            'sampled_penalty_used': penalty, 'dual_used': dual_used, 'dual_next': dual,
            'actual_auxiliary_coefficient': c['auxiliary_coef'],
            'auxiliary_reference_n4': c['auxiliary_reference_n4'],
            'auxiliary_reference_ratio': 4 / c['n_agents'] if c['auxiliary_reference_n4'] is not None else None,
            'reward_advantage_sd': float(reward_scale.detach().cpu()),
            'optimizer_steps': len(optimization), 'mean_approx_kl': float(np.mean([x['approx_kl'] for x in optimization])),
            'actor_change_l2': float(torch.linalg.vector_norm(_vector(model, actor=True) - actor_before)),
            'parameter_change_l2': float(torch.linalg.vector_norm(_vector(model) - before)),
            'optimization_requested_rows': all_oracle.counters['requested_rows'],
            'optimization_evaluated_rows': all_oracle.counters['evaluated_rows'],
            'optimization_cache_hit_rows': all_oracle.counters.get('cache_hit_rows', 0),
            'optimization_precompute_rows': all_oracle.counters.get('precompute_rows', 0),
            'optimization_certified_zero_rows': all_oracle.counters.get('certified_zero_rows', 0),
            'elapsed_s': elapsed_before + time.perf_counter() - start}
        for key in ('reward_policy_grad_norm', 'cost_policy_grad_norm', 'auxiliary_grad_norm',
                    'reward_value_grad_norm', 'cost_value_grad_norm', 'entropy_grad_norm',
                    'ppo_policy_grad_norm', 'auxiliary_to_ppo_grad_norm', 'auxiliary_ppo_grad_cosine',
                    'preclip_grad_norm', 'clipped_grad_norm'):
            values = [x[key] for x in optimization if x[key] is not None]
            row['mean_' + key] = float(np.mean(values)) if values else None
        updates.append(row)
        arrays = {key: tensor.detach().cpu().numpy() for key, tensor in batch.items()}
        arrays.update({key: np.stack(data[key]) for key in ('admitted_actions', 'physical_replacements',
            'institutional_replacements', 'total_replacements', 'task_rewards', 'product_probe_costs', 'admitted_native_actions')})
        arrays.update(reward_advantages=reward_adv.detach().cpu().numpy(),
            cost_advantages=cost_adv.detach().cpu().numpy(), reward_returns=reward_ret.detach().cpu().numpy(),
            cost_returns=cost_ret.detach().cpu().numpy(),
            oracle_snapshots_utf8=np.frombuffer(json.dumps(snapshots, sort_keys=True, separators=(',', ':')).encode(), dtype=np.uint8),
            penalty_used=np.asarray(penalty), dual_used=np.asarray(dual_used))
        name = f'rollouts/update_{update + 1:06}.npz'
        closed[name] = atomic_npz(out / name, arrays, verify_existing=True); fsync_directory((out / name).parent)
        name = f'optimization/update_{update + 1:06}.json'; closed[name] = _closed_json(out / name, optimization)
        ckname = f'checkpoints/update_{update + 1:06}.pt'; ckpath = out / ckname
        if ckpath.exists():
            # A generation not named by LAST_COMMIT is an interrupted-write orphan.
            # Preserve its bytes rather than overwrite or mistake it for committed state.
            publication.quarantine_existing(ckpath, out / 'recovery_orphans')
        state = {'signature': signature, 'update': update + 1, 'step': step, 'model': model.state_dict(),
            'model_specification': model.specification(), 'optimizer': optimizer.state_dict(),
            'envs': [e.state_dict() for e in envs], 'obs': obs, 'dual': dual, 'updates': updates,
            'episodes': episodes, 'env_sums': env_sums, 'closed_files': closed, 'counters': counters,
            'phase_counts': phase_counts,
            'action_digest': digest, 'initial_vector': initial, 'initial_actor_vector': initial_actor,
            'elapsed': elapsed_before + time.perf_counter() - start,
            'generators': {k: v.get_state() for k, v in generators.items()}, 'torch_rng': torch.get_rng_state(),
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
            'pruned_checkpoints': pruned}
        checkpoint_hash = atomic_torch(ckpath, state); fsync_directory(ckpath.parent)
        atomic_json(commit_path, {'signature': signature, 'update': update + 1, 'joint_steps': step,
            'checkpoint': ckname, 'checkpoint_sha256': checkpoint_hash, 'action_digest': digest})
        generations = sorted((out / 'checkpoints').glob('update_*.pt'))
        for obsolete in generations[:-c['keep_checkpoint_generations']]:
            pruned.append({'path': obsolete.relative_to(out).as_posix(), 'sha256': publication.verified_digest(obsolete),
                           'requested_after_update': update + 1, 'status': 'DELETE_INTENT',
                           'reason': 'Declared latest-two generational recovery retention after newer commit'})
            _write_retention(out, signature, c['keep_checkpoint_generations'], pruned)
            obsolete.unlink(); fsync_directory(obsolete.parent)
            pruned[-1]['status'] = 'DELETED'
            _write_retention(out, signature, c['keep_checkpoint_generations'], pruned)
        _write_retention(out, signature, c['keep_checkpoint_generations'], pruned)
        if update + 1 in {max(1, total_updates // 4), max(1, total_updates // 2), max(1, 3 * total_updates // 4), total_updates}:
            atomic_torch(out / f'policy_{step}.pt', model.state_dict())
        print(json.dumps({'run_id': c['run_id'], 'joint_steps': step, 'total_steps': c['total_steps'],
                          'update': update + 1, 'method': c['method'], 'dual': dual}), flush=True)
        if stop_after_updates is not None and update + 1 >= stop_after_updates and update + 1 < total_updates:
            paused = {'schema': 'PCI-FAIR-ENGINEERING-PAUSE-v1', 'status': 'PAUSED_ENGINEERING',
                      'signature': signature, 'update': update + 1, 'joint_steps': step,
                      'checkpoint': ckname, 'checkpoint_sha256': sha256(ckpath)}
            atomic_json(out / 'PAUSED_ENGINEERING.json', paused); lock.close(); return paused
    if runtime_sources() != sources: raise RuntimeError('Runtime source changed during collection')
    if fair_system_meta(c['domain'])['dependency_source_sha256'] != system['dependency_source_sha256']:
        raise RuntimeError('Native dependency source changed during collection')
    if step != c['total_steps'] or counters['native_oracle_matches'] != step:
        raise RuntimeError('Incomplete environment/oracle accounting')
    expected_episodes = sum((c['total_steps'] // c['n_envs']) // env.config.horizon for env in envs)
    if len(episodes) != expected_episodes:
        raise RuntimeError('Episode closure count differs from declared budget')
    expected_phases = {p: np.zeros(len(counts), dtype=np.int64) for p, counts in phase_counts.items()}
    for profile, env in zip(c['env_profile_ids'], envs):
        timeline = np.arange(c['total_steps'] // c['n_envs']) % env.config.horizon % env.config.quota_window
        expected_phases[profile] += np.bincount(timeline, minlength=env.config.quota_window)
    if any(not np.array_equal(expected_phases[p], phase_counts[p]) for p in phase_counts):
        raise RuntimeError('Profile/window-phase exposure differs from declared budget')
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError('Nonfinite final model parameters')
    atomic_csv(out / 'updates.csv', updates)
    atomic_json(out / 'train_episodes.json', episodes)
    atomic_npz(out / 'final_export.npz', {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}, verify_existing=True)
    commit, _ = publication.load_json_verified(commit_path)
    optional_aliases, present_aliases = publication.publication_inventory(out)
    files = {p.relative_to(out).as_posix(): {'bytes': p.stat().st_size, 'sha256': publication.verified_digest(p)}
             for p in sorted(out.rglob('*')) if p.is_file() and p.name not in
             ('RUN.lock', 'TRAIN_COMPLETE.json', 'CLOSED_FILES.json') and p.relative_to(out).as_posix() not in optional_aliases}
    manifest_hash = publication.immutable_json(out / 'CLOSED_FILES.json', {
        'schema': 'PCI-FAIR-TRAIN-CLOSED-FILES-v1', 'signature': signature, 'files': files,
        'optional_publication_aliases': optional_aliases}, verify_existing=True)
    done = {'schema': 'PCI-FAIR-TRAIN-COMPLETE-v1', 'status': 'COMPLETE', 'version': VERSION,
        'signature': signature, 'run_role': c['run_role'], 'run_id': c['run_id'], 'method': c['method'],
        'domain': c['domain'], 'seed': c['seed'], 'protocol_sha256': c.get('protocol_sha256'),
        'sources': sources, 'joint_steps': step, 'agent_decisions': step * c['n_agents'],
        'updates': total_updates, 'complete_episodes': len(episodes), 'model': model.specification(),
        'final_dual': dual, 'counters': counters, 'action_digest': digest,
        'phase_counts': phase_counts,
        'checkpoint': commit['checkpoint'], 'checkpoint_sha256': commit['checkpoint_sha256'],
        'portable_export_sha256': publication.verified_digest(out / 'final_export.npz'),
        'closed_manifest_sha256': manifest_hash,
        'actor_change_l2': float(torch.linalg.vector_norm(_vector(model, actor=True) - initial_actor)),
        'parameter_change_l2': float(torch.linalg.vector_norm(_vector(model) - initial)),
        'elapsed_s': elapsed_before + time.perf_counter() - start,
        'scientific_performance_claim': False if c['run_role'] == 'engineering' else None}
    publication.immutable_json(out / 'TRAIN_COMPLETE.json', done, verify_existing=False); lock.close(); return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path); parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--stop-after-updates', type=int)
    args = parser.parse_args(); print(json.dumps(train(json.loads(args.config.read_text()), args.out,
        stop_after_updates=args.stop_after_updates), sort_keys=True))
