#!/usr/bin/env python3
"""Finite CPU component profile only. No checkpoint training or outcome selection."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import benchmark as b
sys.path.insert(0,str(b.FOLLOWUP))
from pci_followup.model import Policy, energy
from pci_followup.envs import make_env

def main():
    torch.set_num_threads(1);torch.manual_seed(919009)
    rows=[]
    for domain,layout in [('custom','twodoors'),('rware','tiny'),('rware','small')]:
        cfg={'domain':domain,'env':dict(layout=layout,n_agents=4,horizon=256,quota_window=64,quota=1,capacity=2,rule_change=False)}
        envs=[make_env(cfg,910000+i) for i in range(8)];model=Policy('pci_norm',64)
        obs=np.stack([e.features() for e in envs]);neural=envtime=0.;saved=[]
        # Same rollout batch shape as existing training, profiled without learning.
        for t in range(128):
            start=time.perf_counter();x=torch.from_numpy(obs)
            with torch.no_grad():dist,_,_,_=model(x);actions=dist.sample().numpy()
            neural+=time.perf_counter()-start;saved.append(x)
            start=time.perf_counter();new=[]
            for e,a in zip(envs,actions):
                no,_,_,done,_=e.transition(a,enforce=False)
                new.append(e.reset() if done else no)
            obs=np.stack(new);envtime+=time.perf_counter()-start
        batch=torch.cat(saved,0)[:128]
        # Diagnostic backward pass is a component microbenchmark, not a PPO update.
        backward=[]
        for repeat in range(8):
            start=time.perf_counter();model.zero_grad(set_to_none=True)
            dist,v,cv,p=model(batch)
            loss=-dist.entropy().mean()+.1*(v.square().mean()+cv.square().mean())+.2*energy(p,batch,'pci_norm',model.random_action_maps).mean()
            loss.backward();backward.append(time.perf_counter()-start)
        rows.append(dict(domain=domain,layout=layout,joint_steps=1024,rollout_inference_s=neural,
             environment_and_feature_s=envtime,rollout_env_fraction=envtime/(envtime+neural),
             backward_microbatch128_median_s=float(np.median(backward)),device='cpu'))
    report={'scope':'CPU component profile at fixed shape; no speedup claim, no training result, no GPU use.',
      'torch':torch.__version__,'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
      'rows':rows,'next':'Measure equivalent CUDA kernels and transfer overhead in an isolated CUDA environment after the other GPU workload completes; do not assume simulator acceleration.'}
    b.atomic_json(b.ROOT/'validation/CPU_COMPONENT_PROFILE.json',report);print('CPU_COMPONENT_PROFILE',json.dumps(report),flush=True)
if __name__=='__main__':main()
