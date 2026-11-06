"""Independent NumPy forward plus a CPU Torch reference for portable actors."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from training.pci_risk_train.constrained_models import ConstrainedContextPolicy
from training.pci_risk_train import publication_io as publication
from .common import digest,sha

LOGIT_ATOL=2e-5
PROBABILITY_ATOL=2e-6


def softmax64(x):
    x=np.asarray(x,dtype=np.float64);z=np.exp(x-x.max(axis=-1,keepdims=True))
    return z/z.sum(axis=-1,keepdims=True)


class PortableActor:
    def __init__(self,path,spec):
        self.path=Path(path);self.spec=spec
        with np.load(path,allow_pickle=False) as z:self.arrays={k:z[k].copy() for k in z.files}
        if spec['model']!='ConstrainedContextPolicy-v1' or spec['pooled_actor_information'] is not False:
            raise ValueError('Unsupported actor specification')
        h,d,a=spec['hidden'],spec['feature_dim'],spec['action_count']
        want={'masks':(3,d)}
        for p,input_dim in [('heads.0',2*d),('heads.1',2*d),('heads.2',2*d),
                            ('reward_critic',2*a*d),('cost_critic',2*a*d)]:
            for layer,shape in [(0,(h,input_dim)),(2,(h,h)),(4,(1,h))]:
                want[f'{p}.{layer}.weight']=shape;want[f'{p}.{layer}.bias']=(shape[0],)
        if set(self.arrays)!=set(want):raise RuntimeError('Unexpected portable model array closure')
        if any(v.shape!=want[k] or v.dtype!=np.float32 or not np.isfinite(v).all()
               for k,v in self.arrays.items()):raise RuntimeError('Invalid portable model dimensions/dtypes/values')
        masks=np.zeros((3,d),dtype=np.float32)
        for i,idx in enumerate(spec['context_view_indices']):masks[i,list(idx)]=1
        if not np.array_equal(masks,self.arrays['masks']):raise RuntimeError('Portable context masks disagree')
        self.torch=ConstrainedContextPolicy(h,feature_dim=d,action_count=a,
            view_indices=spec['context_view_indices'])
        if self.torch.specification()!=spec:raise RuntimeError('Model specification/parameter accounting mismatch')
        self.torch.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in self.arrays.items()},strict=True)
        self.torch.eval()
        self.max_logit_error=0.;self.max_probability_error=0.;self.checked_states=0

    def numpy(self,features):
        x=np.asarray(features,dtype=np.float32)
        if x.shape[-2:]!=(self.spec['action_count'],self.spec['feature_dim']):
            raise ValueError('Observation dimensions disagree with portable actor')
        heads=[]
        for i in range(3):
            z=np.concatenate((x*self.arrays['masks'][i],np.zeros_like(x)),axis=-1)
            for layer in (0,2,4):
                z=np.matmul(z,self.arrays[f'heads.{i}.{layer}.weight'].T)+self.arrays[f'heads.{i}.{layer}.bias']
                if layer!=4:z=np.tanh(z)
            heads.append(z[...,0])
        logits=np.stack(heads,axis=-2).mean(axis=-2,dtype=np.float32)
        p=softmax64(logits).astype(np.float32)
        return logits,p

    def authoritative(self,features):
        """Actor-only CPU float32 forward. Critic inference is not charged to deployment."""
        x=np.asarray(features,dtype=np.float32)
        with torch.no_grad():
            f=torch.from_numpy(x.copy())
            heads=[head(torch.cat((f*self.torch.masks[i],torch.zeros_like(f)),dim=-1)).squeeze(-1)
                   for i,head in enumerate(self.torch.heads)]
            logits=torch.stack(heads,dim=-2).mean(dim=-2)
            dist=torch.distributions.Categorical(logits=logits)
            truth_logits=logits.numpy();truth_probs=dist.probs.numpy()
        independent_logits,independent_probs=self.numpy(x)
        if not all(np.isfinite(v).all() for v in (truth_logits,truth_probs,independent_logits,independent_probs)):
            raise RuntimeError('Nonfinite authoritative or independent actor output')
        le=float(np.max(np.abs(truth_logits-independent_logits)))
        pe=float(np.max(np.abs(truth_probs-independent_probs)))
        if le>LOGIT_ATOL or pe>PROBABILITY_ATOL:raise RuntimeError(f'Independent portable inference mismatch {le}/{pe}')
        self.max_logit_error=max(self.max_logit_error,le)
        self.max_probability_error=max(self.max_probability_error,pe);self.checked_states+=1
        # Float64 renormalization defines inverse-CDF sampling, not a claim of
        # CPU/GPU bitwise equivalence.
        probabilities=truth_probs.astype(np.float64)
        probabilities/=probabilities.sum(axis=-1,keepdims=True)
        return truth_logits,probabilities


def verify_training_policy(directory, *, current_sources=None):
    """Require the complete current retained training closure before evaluation."""
    root=Path(directory)
    done=json.loads((root/'TRAIN_COMPLETE.json').read_text())
    cfg=json.loads((root/'config.json').read_text())
    prov=json.loads((root/'provenance.json').read_text())
    closed_path=root/'CLOSED_FILES.json';closed=json.loads(closed_path.read_text())
    if done.get('schema')!='PCI-FAIR-TRAIN-COMPLETE-v1' or done.get('status')!='COMPLETE':
        raise RuntimeError('Final complete training receipt required')
    if done['run_id']!=cfg['run_id'] or done['method']!=cfg['method'] or done['domain']!=cfg['domain'] or done['seed']!=cfg['seed']:
        raise RuntimeError('Training identity mismatch')
    if done['joint_steps']!=cfg['total_steps'] or done['agent_decisions']!=cfg['total_steps']*cfg['n_agents']:
        raise RuntimeError('Training budget mismatch')
    if closed.get('schema')!='PCI-FAIR-TRAIN-CLOSED-FILES-v1' or sha(closed_path)!=done['closed_manifest_sha256'] or closed['signature']!=done['signature']:
        raise RuntimeError('Training closure identity mismatch')
    if any(Path(name).name=='publication_io.py' for name in done['sources']) and 'optional_publication_aliases' not in closed:
        raise RuntimeError('Current training closure requires its exact optional alias inventory')
    optional=closed.get('optional_publication_aliases',{})
    if not isinstance(optional,dict):raise RuntimeError('Invalid optional publication alias map')
    for name,item in optional.items():
        if name in closed['files'] or item.get('publication_intent') not in closed['files']:
            raise RuntimeError('Optional alias must be ancillary and its immutable intent mandatory')
    for name,item in closed['files'].items():
        p=root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink():
            raise RuntimeError('Unsafe training closure path')
        try:publication.verified_digest(p,expected=item['sha256'],expected_bytes=item['bytes'])
        except (RuntimeError,OSError) as error:raise RuntimeError('Training retained artifact mismatch: '+name) from error
    # Re-derive the entire optional map from mandatory immutable intents, then
    # verify every present alias through the training helper's held-FD checks.
    # No alias is a checkpoint source or replaces a required retained file.
    publication.verify_optional_aliases(root,optional)
    allowed={'RUN.lock','TRAIN_COMPLETE.json','CLOSED_FILES.json'}|set(closed['files'])|set(optional)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual-allowed:raise RuntimeError('Unclosed training files: '+str(sorted(actual-allowed)))
    signature=digest({'configuration':cfg,'sources':done['sources'],'versions':prov['system']['versions'],
                      'dependency_source_sha256':prov['system']['dependency_source_sha256']})
    if signature!=done['signature'] or prov['signature']!=signature or prov['sources']!=done['sources']:
        raise RuntimeError('Training signature does not bind current configuration/sources/runtime')
    if current_sources is not None and done['sources']!=current_sources:
        raise RuntimeError('Training source closure differs from executing evaluation environment')
    if done.get('protocol_sha256')!=cfg.get('protocol_sha256'):raise RuntimeError('Training protocol binding mismatch')
    p=root/'final_export.npz'
    if sha(p)!=done['portable_export_sha256']:raise RuntimeError('Portable export identity mismatch')
    checkpoint=root/done['checkpoint']
    if Path(done['checkpoint']).is_absolute() or '..' in Path(done['checkpoint']).parts or sha(checkpoint)!=done['checkpoint_sha256']:
        raise RuntimeError('Final checkpoint identity mismatch')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['signature']!=done['signature'] or ck['step']!=cfg['total_steps']:
        raise RuntimeError('Final checkpoint budget/signature mismatch')
    actor=PortableActor(p,done['model'])
    if set(ck['model'])!=set(actor.arrays) or any(not np.array_equal(v.detach().cpu().numpy(),actor.arrays[k]) for k,v in ck['model'].items()):
        raise RuntimeError('Checkpoint and portable export differ')
    return cfg,done,actor
