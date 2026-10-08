"""Solver-backed consistency and contradiction tests, separate from RL performance."""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from .io import atomic_csv,atomic_json


def incidence_cycle(n):
    B=np.zeros((n,n))
    for i in range(n):B[i,i]=-1;B[i,(i+1)%n]=1
    return B


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in (3,6,12,24,48):
        B=incidence_cycle(n)
        for gap in (0.,0.25,0.5,1.,2.,4.):
            b=np.zeros(n);b[-1]=gap
            x=np.linalg.lstsq(B,b,rcond=None)[0]
            residual=B@x-b;measured=float(residual@residual);oracle=gap*gap/n
            if not np.isclose(measured,oracle,atol=1e-11,rtol=1e-10):raise AssertionError('Affine-cycle oracle mismatch')
            feasibility=linprog(np.zeros(n),A_eq=B,b_eq=b,bounds=[(None,None)]*n,method='highs')
            expected=(gap==0)
            if feasibility.success!=expected:raise AssertionError('LP feasibility mismatch')
            rows.append({'contexts':n,'injected_cycle_gap':gap,'minimum_energy':measured,
                         'analytic_minimum':oracle,'lp_feasible':int(feasibility.success)})
    atomic_csv(out/'affine_contradiction.csv',rows)
    rng=np.random.default_rng(4501);bounds=[]
    for n in (3,6,12,24):
        B=incidence_cycle(n)
        U,S,Vh=np.linalg.svd(B,full_matrices=True);pos=S>1e-10;gap=float(S[pos].min())
        for seed in range(20):
            x=rng.normal(size=n);distance=float(np.linalg.norm(x-x.mean()))
            upper=float(np.linalg.norm(B@x)/gap)
            if distance>upper+1e-10:raise AssertionError('Spectral bound failed')
            bounds.append({'contexts':n,'sample':seed,'distance_to_kernel':distance,'spectral_upper_bound':upper})
    atomic_csv(out/'spectral_bound.csv',bounds)
    # Direct sparse edge-residual scaling: record actual wall time, not sleeps.
    timing=[]
    for n in (16,64,256,1024,4096):
        u=np.arange(n);v=(u+1)%n
        for d in (2,8,32):
            x=rng.normal(size=(n,d));w=rng.uniform(.1,2,n)
            for repeat in range(20):
                t=time.perf_counter_ns()
                for _ in range(100):energy=np.sum(w[:,None]*(x[v]-x[u])**2)
                ms=(time.perf_counter_ns()-t)/1e6/100
                timing.append({'contexts':n,'edges':n,'dimension':d,'repeat':repeat,'compute_ms':ms,'energy':float(energy)})
    atomic_csv(out/'consistency_scaling.csv',timing)
    atomic_json(out/'DIAGNOSTICS_COMPLETE.json',{'affine_cases':len(rows),'spectral_cases':len(bounds),
        'scaling_measurements':len(timing),'note':'Known affine contradiction experiment; not evidence that a high RL loss proves infeasibility'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/diagnostics');a=p.parse_args();run(a.out)
