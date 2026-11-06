#!/usr/bin/env python3
"""Prospectively clarified PCI confirmation inference; never reads live services.

Numerical primitives do not establish external data provenance. The learning CLI
uses separately validated complete-cohort receipts; execution count adapters
must be qualified against the future bridge producer before scientific use.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from functools import reduce

RECIPE_ID = 'PCI-M3-CONFIRMATION-ANALYSIS-20261106-v1'
PINNED_SOURCE = 'e7fbf20f649ce40a21d42d644756cf519dd6fd2717564c5bbcf0eb0793473ed7'
PINNED_PROTOCOL = '9bba352fa2c4f05db70a666e716e99d452981b2406831f237c3be6eef934dff3'
RECIPE_SHA256 = '828f5e668876bd49569d635fff76f73d70630bbfd4e76309cca642ea42542328'
EXPECTED_CONTRASTS = (
    'Warehouse Direct minus Fixed',
    'Warehouse Direct minus Lagrangian',
    'Warehouse Direct minus shield-aware',
    'RWARE Direct minus Fixed',
    'RWARE Direct minus Lagrangian',
    'RWARE Direct minus shield-aware',
    'Warehouse execution Direct minus Fixed below capacity',
    'Warehouse execution Direct minus Lagrangian below capacity',
    'Warehouse execution Direct minus shield-aware below capacity',
    'Warehouse execution Direct minus Fixed overload',
    'Warehouse execution Direct minus Lagrangian overload',
    'Warehouse execution Direct minus shield-aware overload',
)
FAMILIES = ('fixed', 'direct', 'lagrangian', 'shieldaware')
COMPARATORS = ('fixed', 'lagrangian', 'shieldaware')
SEEDS = {'warehouse': tuple(range(62001, 62021)), 'rware': tuple(range(75001, 75021))}
LIMITS = {'warehouse': Fraction(1,20), 'rware': Fraction(1,10)}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def fraction(value):
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, dict):
        n, d = value.get('numerator'), value.get('denominator')
        need(type(n) is int and type(d) is int and d > 0, 'INTEGER_RATIONAL_REQUIRED')
        return Fraction(n,d)
    raise ValueError('EXACT_RATIONAL_REQUIRED_NO_BINARY_FLOAT_INPUT')


def packed(value):
    value = fraction(value)
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'decimal': float(value)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    def pairs(items):
        out={}
        for key,value in items:
            need(key not in out,'DUPLICATE_JSON_KEY:'+key);out[key]=value
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('NONFINITE_JSON:'+s)))


def verify_recipe(path=None):
    path = Path(path) if path else Path(__file__).with_name('ANALYSIS_CLARIFICATION.json')
    need(sha_file(path)==RECIPE_SHA256, 'ANALYSIS_RECIPE_HASH_CHANGED')
    recipe=load_json(path)
    need(recipe['recipe_id']==RECIPE_ID and recipe['canonical_contrast_ids']==list(EXPECTED_CONTRASTS),
         'ANALYSIS_RECIPE_IDENTITY')
    return recipe


def hypervolume_exact(points, cost_limit):
    """Normalized dominated area using exact rational cost/utility coordinates."""
    bound=fraction(cost_limit);need(0 < bound <= 1, 'COST_BOUND')
    by_cost={}
    for cost,utility in points:
        cost,utility=fraction(cost),fraction(utility)
        need(0 <= cost <= 1 and 0 <= utility <= 1, 'NORMALIZED_COORDINATES_REQUIRED')
        if cost <= bound:by_cost[cost]=max(by_cost.get(cost,Fraction(0)),utility)
    area=previous=level=Fraction(0)
    for cost,utility in sorted(by_cost.items()):
        area+=(cost-previous)*level;previous=cost;level=max(level,utility)
    return (area+(bound-previous)*level)/bound


def integer_scale(values):
    values=[fraction(v) for v in values]
    denominator=math.lcm(*(v.denominator for v in values))
    return [v.numerator*(denominator//v.denominator) for v in values],denominator


def exact_sign_flip(differences):
    """All 2^20 signs, absolute mean statistic, inclusive integer-exact tail."""
    differences=[fraction(v) for v in differences]
    need(len(differences)==20, 'EXACTLY_20_TRAINING_CLUSTERS_REQUIRED')
    integers,denominator=integer_scale(differences)
    observed=abs(sum(integers));total=-sum(integers);gray=0
    assignments=1<<20;exceedances=int(abs(total)>=observed)
    for i in range(1,assignments):
        j=(i & -i).bit_length()-1
        gray ^= 1<<j
        total += (2 if gray & (1<<j) else -2)*integers[j]
        exceedances += abs(total)>=observed
    return {'statistic':'absolute_mean_paired_training_cluster_difference',
            'statistic_abs_mean':packed(Fraction(observed,20*denominator)),
            'assignments':assignments, 'exceedances':exceedances,
            'p_value':packed(Fraction(exceedances,assignments)),
            'tail':'inclusive_exact_absolute_statistic', 'monte_carlo_correction':False,
            'null_assumption':'Joint sign-flip invariance under the paired cluster sign-symmetry/exchangeability null; not a distribution-free test of mean zero alone.'}


def contrast_seed(contrast_id):
    need(contrast_id in EXPECTED_CONTRASTS,'UNKNOWN_CONTRAST_ID')
    text=RECIPE_ID+'/'+contrast_id
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8],'big')


def quantile_linear_exact(sorted_values, probability):
    """Hyndman-Fan type 7 / NumPy method='linear', evaluated rationally."""
    p=fraction(probability);need(sorted_values and 0<=p<=1,'QUANTILE_INPUT')
    rank=(len(sorted_values)-1)*p;lo=rank.numerator//rank.denominator
    hi=min(lo+1,len(sorted_values)-1);weight=rank-lo
    return fraction(sorted_values[lo])*(1-weight)+fraction(sorted_values[hi])*weight


def paired_intervals(contrast_id, differences):
    """Frozen pointwise CIs, bootstrap resampling whole paired differences."""
    import numpy as np
    import scipy
    from scipy.stats import t
    differences=[fraction(v) for v in differences]
    need(len(differences)==20, 'EXACTLY_20_TRAINING_CLUSTERS_REQUIRED')
    integers,denominator=integer_scale(differences)
    seed=contrast_seed(contrast_id)
    indices=np.random.Generator(np.random.PCG64(seed)).integers(0,20,size=(10000,20),dtype=np.int64)
    sums=sorted(sum(integers[int(i)] for i in row) for row in indices)
    lower=quantile_linear_exact(sums,Fraction(1,40))/(20*denominator)
    upper=quantile_linear_exact(sums,Fraction(39,40))/(20*denominator)
    mean=sum(differences,Fraction(0))/20
    variance=sum(((d-mean)**2 for d in differences),Fraction(0))/19
    sd=math.sqrt(float(variance));critical=float(t.ppf(.975,19))
    radius=critical*sd/math.sqrt(20)
    bootstrap={'method':'paired_whole_training_cluster_percentile', 'confidence':.95,
        'resamples':10000,'rng':'NumPy Generator(PCG64)','seed':seed,
        'seed_derivation_utf8':RECIPE_ID+'/'+contrast_id,'quantile_method':'linear',
        'endpoint_arithmetic':'exact_rational_linear_quantiles_of_integer_scaled_resample_sums',
        'lower':packed(lower),'upper':packed(upper),'pointwise':True,'simultaneous':False,
        'resample_index_sha256':hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest(),
        'numpy_version':np.__version__}
    sensitivity={'method':'paired_t','confidence':.95,'df':19,'critical_value':critical,
        'lower':float(mean)-radius,'upper':float(mean)+radius,
        'status':'DEGENERATE_ZERO_SAMPLE_VARIANCE' if variance==0 else 'DEFINED',
        'pointwise':True,'simultaneous':False,'descriptive_sensitivity':True,
        'scipy_version':scipy.__version__}
    dz={'status':'UNDEFINED_ZERO_PAIRED_SD','value':None} if variance==0 else {'status':'DEFINED','value':float(mean)/sd}
    return {'bootstrap95':bootstrap,'paired_t95':sensitivity,'paired_sample_variance':packed(variance),
            'paired_sample_sd':sd,'dz':dz}


def expected_seeds(contrast_id):
    need(contrast_id in EXPECTED_CONTRASTS,'UNKNOWN_CONTRAST_ID')
    return SEEDS['rware' if contrast_id.startswith('RWARE ') else 'warehouse']


def contrast_statistics(contrast_id,seeds,differences):
    need(tuple(seeds)==expected_seeds(contrast_id),'WRONG_OR_UNORDERED_CLUSTER_SEEDS')
    differences=[fraction(v) for v in differences]
    need(len(differences)==20 and all(-1<=d<=1 for d in differences),'NORMALIZED_HV_DIFFERENCES')
    test=exact_sign_flip(differences)
    return {'contrast_id':contrast_id,'training_clusters':20,
        'cluster_differences':[{'training_seed':s,'difference':packed(d)} for s,d in zip(seeds,differences)],
        'mean_effect':packed(sum(differences,Fraction(0))/20),'p_value':test['p_value'],
        'test':test,**paired_intervals(contrast_id,differences)}


def has_decision_fields(value):
    if isinstance(value,dict):
        return any(key in ('adjusted_p_value','reject') or has_decision_fields(item) for key,item in value.items())
    if isinstance(value,(list,tuple)):
        return any(has_decision_fields(item) for item in value)
    return False


def holm12(records,require_complete=False):
    """Exact Holm adjustment only when every prescribed contrast is present."""
    by_id={}
    for record in records:
        identifier=record['contrast_id']
        need(identifier in EXPECTED_CONTRASTS,'UNKNOWN_CONTRAST_ID')
        need(identifier not in by_id,'DUPLICATE_CONTRAST_ID')
        p=fraction(record['p_value']);need(0<=p<=1,'P_VALUE_RANGE')
        need(not has_decision_fields(record),'RAW_CONTRAST_RECORD_REQUIRED')
        by_id[identifier]=dict(record,p_value=packed(p))
    missing=[i for i in EXPECTED_CONTRASTS if i not in by_id]
    if missing:
        need(not require_complete,'COMPLETE_12_CONTRAST_FAMILY_REQUIRED')
        return {'status':'DESCRIPTIVE_PARTIAL_FAMILY','family_complete':False,
            'pending_contrasts':missing,'contrasts':[by_id[i] for i in EXPECTED_CONTRASTS if i in by_id],
            'interpretation':'Raw tests and pointwise intervals are provisional/descriptive; no completed familywise inference.'}
    ranked=sorted(EXPECTED_CONTRASTS,key=lambda i:(fraction(by_id[i]['p_value']),EXPECTED_CONTRASTS.index(i)))
    running=Fraction(0);adjusted={}
    for rank,identifier in enumerate(ranked):
        running=max(running,min(Fraction(1),(12-rank)*fraction(by_id[identifier]['p_value'])))
        adjusted[identifier]=running
    return {'status':'COMPLETE_GLOBAL_HOLM12','family_complete':True,'alpha':.05,
        'adjustment':'Holm step-down, fixed twelve-contrast family',
        'contrasts':[dict(by_id[i],adjusted_p_value=packed(adjusted[i]),reject=adjusted[i]<=Fraction(1,20)) for i in EXPECTED_CONTRASTS],
        'interpretation':'Two-sided familywise tests under their stated null assumptions; nonsignificance is not equivalence; intervals remain pointwise.'}


def selected_by_family(selection,domain):
    need(domain in SEEDS,'DOMAIN')
    need(selection.get('schema')=='PCI-DEVELOPMENT-FRONTIER-SELECTION-v1' and
         selection.get('status')=='DEVELOPMENT_ONLY_SELECTION' and
         selection.get('study_domain')==domain and selection.get('primary_test_data_opened') is False,'DEVELOPMENT_SELECTION_IDENTITY')
    ids=selection['selected_candidate_ids'];need(len(ids)==8 and len(set(ids))==8,'EIGHT_DISTINCT_SELECTED_CONFIGURATIONS')
    families={f:[] for f in FAMILIES}
    for identifier in ids:
        possible=[f for f in FAMILIES if identifier in {f'{f}_{j:02d}' for j in range(12)}]
        need(len(possible)==1,'UNKNOWN_CANDIDATE_ID');families[possible[0]].append(identifier)
    need(all(len(v)==2 for v in families.values()),'TWO_SELECTED_CONFIGURATIONS_PER_FAMILY')
    return families


def learning_endpoints(seed_rows,selection,domain):
    chosen=selected_by_family(selection,domain);index={}
    wanted={(c,s) for ids in chosen.values() for c in ids for s in SEEDS[domain]}
    for row in seed_rows:
        key=(row['candidate_id'],row['training_seed'])
        need(key not in index,'DUPLICATE_LEARNING_CELL')
        need(row['domain']==domain and row['n_agents']==4,'LEARNING_DOMAIN_N4')
        need(row['cohort_complete'] is True and row['evaluation_complete'] is True,'INCOMPLETE_LEARNING_CELL')
        need(row['family'] in chosen and row['candidate_id'] in chosen[row['family']],'CANDIDATE_FAMILY')
        for k in ('joint_steps','institutional_replacements','deliveries'):
            need(type(row[k]) is int and row[k]>=0,'INTEGER_NONNEGATIVE_COUNTS')
        need(row['joint_steps']>0,'POSITIVE_EVALUATION_EXPOSURE');index[key]=row
    need(set(index)==wanted,'COMPLETE_160_POLICY_CONFIRMATION_REQUIRED')
    need(len({r['joint_steps'] for r in seed_rows})==1,'MATCHED_EVALUATION_EXPOSURES_REQUIRED')
    endpoints={}
    for seed in SEEDS[domain]:
        for family,ids in chosen.items():
            points=[]
            for candidate in ids:
                row=index[candidate,seed];denominator=4*row['joint_steps']
                points.append((Fraction(row['institutional_replacements'],denominator),Fraction(row['deliveries'],denominator)))
            endpoints[seed,family]=hypervolume_exact(points,LIMITS[domain])
    return endpoints


def execution_endpoints(rows,selection):
    """Numerical adapter contract, pending qualification against bridge producer.

    Does not hardcode the provisional offer count. Requires full two-load,
    two-replicate,160-policy population and averages normalized replicate points
    before computing each within-cluster policy-pair hypervolume.
    """
    chosen=selected_by_family(selection,'warehouse');index={};loads=('below_capacity','overload')
    wanted={(c,s,load,rep) for ids in chosen.values() for c in ids for s in SEEDS['warehouse'] for load in loads for rep in (0,1)}
    for row in rows:
        key=(row['candidate_id'],row['training_seed'],row['load'],row['replicate'])
        need(key not in index,'DUPLICATE_EXECUTION_CELL')
        need(row['complete_drain'] is True and row['n_agents']==4,'COMPLETE_N4_EXECUTION_DRAIN_REQUIRED')
        need(row['family'] in chosen and row['candidate_id'] in chosen[row['family']],'CANDIDATE_FAMILY')
        for k in ('offered_requests','valid_unapplied_proofs','delivered_task_units'):
            need(type(row[k]) is int and row[k]>=0,'INTEGER_NONNEGATIVE_EXECUTION_COUNTS')
        need(row['offered_requests']>0,'POSITIVE_OFFER_EXPOSURE')
        need(row['valid_unapplied_proofs']<=row['offered_requests'] and row['delivered_task_units']<=4*row['offered_requests'],'EXECUTION_REPLICATE_NORMALIZED_BOUNDS')
        index[key]=row
    need(set(index)==wanted,'COMPLETE_640_EXECUTION_CELL_CONTRACT_REQUIRED')
    blocks={}
    for row in rows:
        block=(row['training_seed'],row['load'],row['replicate'])
        blocks.setdefault(block,set()).add(row['offered_requests'])
    need(all(len(exposures)==1 for exposures in blocks.values()),'PAIRED_EXECUTION_OFFER_EXPOSURE')
    endpoints={}
    for seed in SEEDS['warehouse']:
        for load in loads:
            for family,ids in chosen.items():
                points=[]
                for candidate in ids:
                    rr=[index[candidate,seed,load,rep] for rep in (0,1)]
                    cost=sum((Fraction(r['valid_unapplied_proofs'],r['offered_requests']) for r in rr),Fraction(0))/2
                    utility=sum((Fraction(r['delivered_task_units'],4*r['offered_requests']) for r in rr),Fraction(0))/2
                    points.append((cost,utility))
                endpoints[seed,load,family]=hypervolume_exact(points,Fraction(1))
    return endpoints


def learning_contrasts(endpoints,domain):
    offset=0 if domain=='warehouse' else 3
    return [contrast_statistics(EXPECTED_CONTRASTS[offset+j],SEEDS[domain],
            [endpoints[s,'direct']-endpoints[s,f] for s in SEEDS[domain]]) for j,f in enumerate(COMPARATORS)]


def execution_contrasts(endpoints):
    return [contrast_statistics(EXPECTED_CONTRASTS[6+3*i+j],SEEDS['warehouse'],
            [endpoints[s,load,'direct']-endpoints[s,load,f] for s in SEEDS['warehouse']])
            for i,load in enumerate(('below_capacity','overload')) for j,f in enumerate(COMPARATORS)]


def verify_implementation_lock():
    import numpy as np
    import scipy
    root=Path(__file__).resolve().parent
    lock=load_json(root/'ANALYSIS_IMPLEMENTATION_LOCK.json')
    need(lock.get('schema')=='PCI-CONFIRMATION-ANALYSIS-IMPLEMENTATION-LOCK-v1' and
         lock.get('status')=='PASS_SYNTHETIC_QUALIFICATION' and lock.get('recipe_sha256')==RECIPE_SHA256,
         'ANALYSIS_IMPLEMENTATION_LOCK_REQUIRED')
    for name,expected in lock['runtime_sources'].items():
        need(name in ('confirmation_analysis.py','learning_receipts.py') and sha_file(root/name)==expected,
             'ANALYSIS_IMPLEMENTATION_SOURCE_CHANGED:'+name)
    need(set(lock['runtime_sources'])=={'confirmation_analysis.py','learning_receipts.py'},'ANALYSIS_IMPLEMENTATION_SOURCE_CLOSURE')
    versions={'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__}
    need(versions==lock['analysis_runtime_versions'],'ANALYSIS_RUNTIME_REQUALIFICATION_REQUIRED')
    return {'implementation_lock_sha256':sha_file(root/'ANALYSIS_IMPLEMENTATION_LOCK.json'),
            'runtime_sources':lock['runtime_sources'],'analysis_runtime_versions':versions}


def result_packet(records,provenance,require_complete=False):
    verify_recipe()
    implementation=verify_implementation_lock()
    return {'schema':'PCI-CONFIRMATION-INFERENCE-RESULT-v1','recipe_id':RECIPE_ID,
        'recipe_sha256':RECIPE_SHA256,'source_map_sha256':PINNED_SOURCE,
        'protocol_sha256':PINNED_PROTOCOL,'implementation':implementation,'created_utc':datetime.now(timezone.utc).isoformat(),
        'provenance':provenance,**holm12(records,require_complete=require_complete)}


def revalidate_contrast(record):
    identifier=record['contrast_id'];need(identifier in EXPECTED_CONTRASTS,'UNKNOWN_CONTRAST_ID')
    rows=record['cluster_differences'];seeds=[r['training_seed'] for r in rows]
    differences=[fraction(r['difference']) for r in rows]
    raw={k:v for k,v in record.items() if k not in ('adjusted_p_value','reject')}
    recalculated=contrast_statistics(identifier,seeds,differences)
    need(raw==recalculated,'CONTRAST_RECOMPUTATION_DIFFERS:'+identifier)
    return recalculated


def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as handle:handle.write(canonical(value)+b'\n')


def learning_packet(cohort,source_root,selection_path,case_manifest_path):
    from learning_receipts import load_learning_receipts
    inputs={'cohort':str(Path(cohort).resolve()),'source_root':str(Path(source_root).resolve()),
            'selection_path':str(Path(selection_path).resolve()),'case_manifest_path':str(Path(case_manifest_path).resolve())}
    data=load_learning_receipts(**inputs)
    endpoints=learning_endpoints(data['seed_rows'],data['selection'],data['domain'])
    records=learning_contrasts(endpoints,data['domain'])
    provenance={'kind':'SOURCE_BOUND_COMPLETE_LEARNING_COHORT','domain':data['domain'],
                'inputs':inputs,'receipt_adapter':data['provenance']}
    packet=result_packet(records,provenance)
    packet['endpoint_data']={'selected_candidate_ids':data['selection']['selected_candidate_ids'],
        'cost_limit':packed(LIMITS[data['domain']]),
        'family_cluster_hypervolumes':[{'training_seed':seed,'family':family,'hypervolume':packed(endpoints[seed,family])}
            for seed in SEEDS[data['domain']] for family in FAMILIES],
        'policy_points':[{'training_seed':r['training_seed'],'candidate_id':r['candidate_id'],'family':r['family'],
            'institutional_replacements':r['institutional_replacements'],'deliveries':r['deliveries'],'joint_steps':r['joint_steps'],'n_agents':4,
            'cost':packed(Fraction(r['institutional_replacements'],4*r['joint_steps'])),
            'utility':packed(Fraction(r['deliveries'],4*r['joint_steps']))}
            for r in sorted(data['seed_rows'],key=lambda r:(r['training_seed'],r['candidate_id']))]}
    return packet


def verified_learning_packet(path):
    packet=load_json(path)
    need(packet.get('schema')=='PCI-CONFIRMATION-INFERENCE-RESULT-v1' and packet.get('recipe_sha256')==RECIPE_SHA256 and
         packet.get('source_map_sha256')==PINNED_SOURCE and packet.get('protocol_sha256')==PINNED_PROTOCOL,'INFERENCE_PACKET_BINDING')
    provenance=packet.get('provenance',{})
    need(isinstance(provenance,dict) and provenance.get('kind')=='SOURCE_BOUND_COMPLETE_LEARNING_COHORT',
         'SCIENTIFIC_EXECUTION_OR_COMBINED_PACKET_ADAPTER_NOT_QUALIFIED_USE_ORIGINAL_LEARNING_PACKETS')
    # Reopen the closed producer receipts and rebuild exact endpoints, not just
    # recompute tests from unverified user-supplied cluster differences.
    rebuilt=learning_packet(**provenance['inputs'])
    need(packet['provenance']==rebuilt['provenance'],'LEARNING_PACKET_RECEIPT_PROVENANCE_CHANGED')
    need(packet['contrasts']==rebuilt['contrasts'] and packet.get('endpoint_data')==rebuilt['endpoint_data'],'LEARNING_PACKET_RECOMPUTATION_DIFFERS')
    need(packet.get('implementation')==rebuilt['implementation'],'ANALYSIS_IMPLEMENTATION_BINDING_CHANGED')
    need(packet['status']==rebuilt['status'] and packet['family_complete']==rebuilt['family_complete'],'LEARNING_PACKET_SCOPE')
    return rebuilt['contrasts'],{'path':str(Path(path).resolve()),'sha256':sha_file(path),'upstream_provenance':rebuilt['provenance']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    learn=sub.add_parser('learning');learn.add_argument('--cohort',type=Path,required=True)
    learn.add_argument('--source-root',type=Path,required=True);learn.add_argument('--selection',type=Path,required=True)
    learn.add_argument('--case-manifest',type=Path,required=True);learn.add_argument('--out',type=Path,required=True)
    combine=sub.add_parser('combine');combine.add_argument('--inputs',type=Path,nargs='+',required=True)
    combine.add_argument('--require-complete',action='store_true');combine.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();verify_recipe();verify_implementation_lock()
    if args.command=='learning':
        packet=learning_packet(args.cohort,args.source_root,args.selection,args.case_manifest)
    else:
        records=[];provenance=[]
        for path in args.inputs:
            contrasts,origin=verified_learning_packet(path);records.extend(contrasts);provenance.append(origin)
        packet=result_packet(records,{'kind':'REVALIDATED_PARTIAL_LEARNING_PACKETS','inputs':provenance},require_complete=args.require_complete)
    write_once(args.out,packet)
    print(json.dumps({'status':packet['status'],'out':str(args.out),'contrast_count':len(packet['contrasts'])}))


if __name__=='__main__':main()
