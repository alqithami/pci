"""Generate one explicit, static-geometry, all-agent transition circuit.

No host-computed collision flag is trusted. Movement, range, collision, region
membership, aggregate capacity, and quota arithmetic are recomputed in-circuit.
Private-state truth is anchored to a simulator-issued commitment, an explicit trust
assumption, not a claim that SNARKs certify sensors or policy inference.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .environment import geometry
from .io import atomic_json,canonical_hash


def geometry_tag(layout):
    walls,regions,*_=geometry(layout)
    return int(canonical_hash({'walls':walls.astype(int).tolist(),'regions':regions})[:14],16)


def generate(layout,n,out):
    if n not in (2,4,8,12):raise ValueError('Supported agent counts: 2,4,8,12')
    walls,regions,*_=geometry(layout)
    blocked=[(x,y) for y in range(1,12) for x in range(1,12) if walls[y,x]]
    # Bounds below enforce the exterior wall implicitly.
    floor_checks=[]
    for z,(x,y) in enumerate(blocked):
        floor_checks.append(f'''        blockedOld[i][{z}] = IsEqual();
        blockedOld[i][{z}].in[0] <== x[i]+13*y[i];
        blockedOld[i][{z}].in[1] <== {x+13*y};
        blockedOld[i][{z}].out === 0;
        blockedNew[i][{z}] = IsEqual();
        blockedNew[i][{z}].in[0] <== nx[i]+13*ny[i];
        blockedNew[i][{z}].in[1] <== {x+13*y};
        blockedNew[i][{z}].out === 0;''')
    region_checks=[]
    for r,(lx,hx,ly,hy) in enumerate(regions):
        region_checks.append(f'''        oldRegion[i][{r}] = InRegion({lx},{hx},{ly},{hy});
        newRegion[i][{r}] = InRegion({lx},{hx},{ly},{hy});
        oldRegion[i][{r}].x <== x[i]; oldRegion[i][{r}].y <== y[i];
        newRegion[i][{r}].x <== nx[i]; newRegion[i][{r}].y <== ny[i];
        entered[i][{r}] <== newRegion[i][{r}].inside*(1-oldRegion[i][{r}].inside);
        usedBits[i][{r}] = Num2Bits(16); usedBits[i][{r}].in <== used[i][{r}];
        quotaBits[i][{r}] = Num2Bits(16); quotaBits[i][{r}].in <== quotas[i][{r}];
        nextUsed[i][{r}] <== used[i][{r}]+entered[i][{r}];
        quotaLT[i][{r}] = LessThan(17);
        quotaLT[i][{r}].in[0] <== quotas[i][{r}];
        quotaLT[i][{r}].in[1] <== nextUsed[i][{r}];
        quotaLT[i][{r}].out === 0;''')
    source=f'''pragma circom 2.2.0;
include "primitives.circom";
template WarehouseTransition(N) {{
    signal input x[N]; signal input y[N]; signal input action[N];
    signal input used[N][2]; signal input quotas[N][2]; signal input capacities[2];
    signal input salt;
    signal input run_tag; signal input episode; signal input step; signal input nonce;
    signal input policy_version;
    signal input state_root; signal input action_root; signal input rules_root;
    signal output ok; signal output record_hash;
    signal nx[N]; signal ny[N];
    signal entered[N][2]; signal nextUsed[N][2];
    component xBits[N]; component yBits[N]; component nxBits[N]; component nyBits[N];
    component xLo[N]; component xHi[N]; component yLo[N]; component yHi[N];
    component nxLo[N]; component nxHi[N]; component nyLo[N]; component nyHi[N];
    component actionBits[N]; component actionRange[N]; component act[N][5];
    component blockedOld[N][{len(blocked)}]; component blockedNew[N][{len(blocked)}];
    component oldRegion[N][2]; component newRegion[N][2];
    component usedBits[N][2]; component quotaBits[N][2]; component quotaLT[N][2];
    for (var i=0;i<N;i++) {{
        actionBits[i] = Num2Bits(3); actionBits[i].in <== action[i];
        actionRange[i] = LessThan(3); actionRange[i].in[0] <== action[i]; actionRange[i].in[1] <== 5;
        actionRange[i].out === 1;
        for (var a=0;a<5;a++) {{
            act[i][a] = IsEqual(); act[i][a].in[0] <== action[i]; act[i][a].in[1] <== a;
        }}
        nx[i] <== x[i]+act[i][1].out-act[i][2].out;
        ny[i] <== y[i]+act[i][4].out-act[i][3].out;
        xBits[i]=Num2Bits(5); xBits[i].in <== x[i];
        yBits[i]=Num2Bits(5); yBits[i].in <== y[i];
        nxBits[i]=Num2Bits(5); nxBits[i].in <== nx[i];
        nyBits[i]=Num2Bits(5); nyBits[i].in <== ny[i];
        xLo[i]=LessThan(5); xLo[i].in[0] <== x[i]; xLo[i].in[1] <== 1; xLo[i].out === 0;
        yLo[i]=LessThan(5); yLo[i].in[0] <== y[i]; yLo[i].in[1] <== 1; yLo[i].out === 0;
        nxLo[i]=LessThan(5); nxLo[i].in[0] <== nx[i]; nxLo[i].in[1] <== 1; nxLo[i].out === 0;
        nyLo[i]=LessThan(5); nyLo[i].in[0] <== ny[i]; nyLo[i].in[1] <== 1; nyLo[i].out === 0;
        xHi[i]=LessThan(5); xHi[i].in[0] <== x[i]; xHi[i].in[1] <== 12; xHi[i].out === 1;
        yHi[i]=LessThan(5); yHi[i].in[0] <== y[i]; yHi[i].in[1] <== 12; yHi[i].out === 1;
        nxHi[i]=LessThan(5); nxHi[i].in[0] <== nx[i]; nxHi[i].in[1] <== 12; nxHi[i].out === 1;
        nyHi[i]=LessThan(5); nyHi[i].in[0] <== ny[i]; nyHi[i].in[1] <== 12; nyHi[i].out === 1;
{chr(10).join(floor_checks)}
{chr(10).join(region_checks)}
    }}
    // Exact aggregate capacities, not a pairwise approximation.
    signal occupancy[2][N+1]; component capBits[2]; component capLT[2];
    for (var r=0;r<2;r++) {{
        occupancy[r][0] <== 0;
        for(var i=0;i<N;i++) occupancy[r][i+1] <== occupancy[r][i]+newRegion[i][r].inside;
        capBits[r]=Num2Bits(8);capBits[r].in <== capacities[r];
        capLT[r]=LessThan(8);capLT[r].in[0] <== capacities[r];capLT[r].in[1] <== occupancy[r][N];
        capLT[r].out === 0;
    }}
    var PAIRS=N*(N-1)\\2;
    component same[PAIRS]; component oldSame[PAIRS]; component swapA[PAIRS];component swapB[PAIRS];
    signal swapBoth[PAIRS];
    var p=0;
    for(var i=0;i<N;i++) {{
        for(var j=0;j<i;j++) {{
            same[p]=IsEqual();same[p].in[0] <== nx[i]+13*ny[i];same[p].in[1] <== nx[j]+13*ny[j];same[p].out === 0;
            oldSame[p]=IsEqual();oldSame[p].in[0] <== x[i]+13*y[i];oldSame[p].in[1] <== x[j]+13*y[j];oldSame[p].out === 0;
            swapA[p]=IsEqual();swapA[p].in[0] <== nx[i]+13*ny[i];swapA[p].in[1] <== x[j]+13*y[j];
            swapB[p]=IsEqual();swapB[p].in[0] <== nx[j]+13*ny[j];swapB[p].in[1] <== x[i]+13*y[i];
            swapBoth[p] <== swapA[p].out*swapB[p].out;swapBoth[p] === 0;
            p++;
        }}
    }}
    component state=VectorCommit(3+4*N,1001);state.salt <== salt;
    state.values[0] <== run_tag;state.values[1] <== episode;state.values[2] <== step;
    for(var i=0;i<N;i++) {{
        state.values[3+4*i] <== x[i];state.values[4+4*i] <== y[i];
        state.values[5+4*i] <== used[i][0];state.values[6+4*i] <== used[i][1];
    }}
    state.root === state_root;
    component actions=VectorCommit(4+N,1002);actions.salt <== 0;
    actions.values[0] <== run_tag;actions.values[1] <== episode;actions.values[2] <== step;actions.values[3] <== nonce;
    for(var i=0;i<N;i++) actions.values[4+i] <== action[i];
    actions.root === action_root;
    component rules=VectorCommit(5+2*N,1003);rules.salt <== 0;
    rules.values[0] <== policy_version;rules.values[1] <== N;rules.values[2] <== {geometry_tag(layout)};
    rules.values[3] <== capacities[0];rules.values[4] <== capacities[1];
    for(var i=0;i<N;i++) {{rules.values[5+2*i] <== quotas[i][0];rules.values[6+2*i] <== quotas[i][1];}}
    rules.root === rules_root;
    component rec=Poseidon(4);rec.inputs[0] <== state_root;rec.inputs[1] <== action_root;
    rec.inputs[2] <== rules_root;rec.inputs[3] <== nonce;
    record_hash <== rec.out;
    ok <== 1;
}}
component main {{public [run_tag,episode,step,nonce,policy_version,state_root,action_root,rules_root]}} = WarehouseTransition({n});
'''
    # Backslash is Circom's integer division; avoid two-character escaped output.
    source=source.replace('\\\\2','\\2')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    name=f'transition_{layout}_n{n}'
    (out/f'{name}.circom').write_text(source)
    atomic_json(out/f'{name}.spec.json',{'name':name,'layout':layout,'n_agents':n,'geometry_tag':geometry_tag(layout),
        'state_bits':5,'counter_bits':16,'capacity_bits':8,'public_names':['ok','record_hash','run_tag','episode','step','nonce','policy_version','state_root','action_root','rules_root']})
    return out/f'{name}.circom'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--layout',default='twodoors');p.add_argument('--agents',type=int,default=4)
    p.add_argument('--out',default='crypto/generated');a=p.parse_args();print(generate(a.layout,a.agents,a.out))
