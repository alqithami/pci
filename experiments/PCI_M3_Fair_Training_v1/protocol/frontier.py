"""Deterministic finite-family frontier selection using development data only."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from build_study import candidates,DEVELOPMENT_SEEDS,RWARE_DEVELOPMENT_SEEDS,FAMILIES,COST_LIMITS

def hypervolume(points, cost_limit=.05):
    """Area below best utility at each allowed cost, over the declared unit box.

    points are RAW (cost,utility); lower cost and higher utility are preferred.
    Area is normalized by the within-domain bound fixed before new outcomes.
    This is not a noninferiority or equivalence test.
    """
    p=[(float(c),float(u)) for c,u in points]
    if any(not(math.isfinite(c) and math.isfinite(u) and 0<=c<=1 and 0<=u<=1) for c,u in p):
        raise ValueError('Frontier points must be finite normalized costs/utilities in[0,1]')
    if not math.isfinite(cost_limit) or not 0<cost_limit<=1:raise ValueError('Invalid fixed cost rectangle')
    by_cost={}
    for c,u in p:
        if c<=cost_limit:by_cost[c]=max(by_cost.get(c,0.),u)
    level=0.;previous=0.;area=0.
    for c,u in sorted(by_cost.items()):
        area+=(c-previous)*level
        level=max(level,u);previous=c
    return (area+(cost_limit-previous)*level)/cost_limit

def select(records, *, source_sha256, study_domain='warehouse'):
    if study_domain not in ('warehouse','rware'):raise ValueError('STUDY_DOMAIN')
    seeds=DEVELOPMENT_SEEDS if study_domain=='warehouse' else RWARE_DEVELOPMENT_SEEDS
    cost_limit=COST_LIMITS[study_domain]
    want=candidates();index={(r['candidate_id'],int(r['training_seed'])):r for r in records}
    keys={(c['candidate_id'],s) for c in want for s in seeds}
    if len(index)!=len(records) or set(index)!=keys:
        raise ValueError('Selection requires exactly all48 candidates×3 development clusters, no duplicates or holdout rows')
    means={}
    for row in want:
        values=[]
        for seed in seeds:
            r=index[row['candidate_id'],seed]
            if r.get('domain',study_domain)!=study_domain:raise ValueError('MIXED_STUDY_DOMAIN')
            if r.get('cohort_complete') is not True or r.get('evaluation_complete') is not True:
                raise ValueError('Incomplete candidate cannot enter selection')
            n=int(r['n_agents']);steps=int(r['joint_steps'])
            if n!=4 or steps<=0:raise ValueError('Expected qualified N4 development evaluation')
            values.append((r['institutional_replacements']/(n*steps),r['deliveries']/(n*steps)))
        means[row['candidate_id']]=tuple(sum(v[j] for v in values)/len(values) for j in range(2))
    selected=[];details={}
    for f in FAMILIES:
        ids=sorted(r['candidate_id'] for r in want if r['family']==f)
        ranked=[(-hypervolume([means[a],means[b]],cost_limit=cost_limit),(a,b)) for a,b in itertools.combinations(ids,2)]
        neg,chosen=min(ranked)
        selected.extend(chosen);details[f]={'selected':list(chosen),'development_hypervolume':-neg,
                                           'all_candidate_mean_points':{i:{'cost':means[i][0],'utility':means[i][1],'outside_primary_rectangle':means[i][0]>cost_limit} for i in ids},
                                           'descriptive_sensitivity':{str(bound):hypervolume([means[i] for i in chosen],cost_limit=bound) for bound in (.01,.025,.05,.1,1)}}
    active_ids=sorted(r['candidate_id'] for r in want if r['family']=='direct' and r['auxiliary_coef']>0)
    active_neg,active_pair=min((-hypervolume([means[a],means[b]],cost_limit=cost_limit),(a,b)) for a,b in itertools.combinations(active_ids,2))
    return {'schema':'PCI-DEVELOPMENT-FRONTIER-SELECTION-v1','status':'DEVELOPMENT_ONLY_SELECTION',
            'development_seeds':seeds,'source_sha256':source_sha256,'study_domain':study_domain,
            'selected_candidate_ids':selected,'selection_rule':f'Maximum two-point finite-candidate hypervolume cost[0,{cost_limit}],utility[0,1]; lexicographic tie break',
            'cost_limit':cost_limit,'raw_points_preserved':True,'cross_domain_pooling':False,
            'estimator_ablation_candidate_ids':list(active_pair),
            'estimator_ablation_selection':{'eligible_ids':active_ids,'development_hypervolume':-active_neg,
                'rule':'Same development-only two-point hypervolume among NONZERO auxiliary cells; independent of primary selection, deterministic candidate-ID tie break.',
                'primary_frontier_selection_changed':False,'lambda_zero_role':'Identity controls; never active-estimator evidence.'},
            'primary_test_data_opened':False,'families':details}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--development-records',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--study-domain',choices=['warehouse','rware'],default='warehouse')
    a=ap.parse_args();body=a.development_records.read_bytes();records=json.loads(body)
    result=select(records,source_sha256=hashlib.sha256(body).hexdigest(),study_domain=a.study_domain)
    if a.output.exists():raise RuntimeError('Refusing to overwrite development selection receipt')
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(result['status'])
