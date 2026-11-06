"""Predeclared ten-cluster descriptive and exact paired randomization analysis.

All 1,200 complete cases and their separate retained-record audit are required.
The exact sign-flip family uses integer repair totals, avoiding floating-point
threshold ambiguity. P-values concern a symmetric paired-sign null, not a
universal guarantee or a distribution-free causal assertion.
"""
from __future__ import annotations
import argparse, fractions, itertools, json, math, statistics
from pathlib import Path
from .common import require, sha, read_json, write_new_json
from .engine import METHODS, PROFILES, validate_plan
from .retained_audit import verify_closed_inventory

COMPARATORS = ('semantic','fixed','adaptive','consensus')
RATE_COUNTS = {'institutional_per_1000_agent_decisions':'institutional_stage_agent_replacements',
               'physical_per_1000_agent_decisions':'physical_agent_replacements',
               'total_per_1000_agent_decisions':'total_agent_replacements',
               'deliveries_per_1000_joint_steps':'deliveries'}

def exact_signflip(differences):
    require(type(differences) is list and 1 <= len(differences) <= 20, 'BOUNDED_PAIRED_DIFFERENCE_VECTOR')
    ds = []
    for x in differences:
        require(type(x) in (int,float) and math.isfinite(x), 'FINITE_PAIRED_DIFFERENCE')
        ds.append(fractions.Fraction(str(x)))
    observed = abs(sum(ds))
    extreme = sum(abs(sum(s*d for s,d in zip(signs,ds))) >= observed
                  for signs in itertools.product((-1,1), repeat=len(ds)))
    denominator = 2**len(ds)
    return dict(n_pairs=len(ds), observed_mean=float(sum(ds)/len(ds)),
        extreme_sign_assignments=extreme, sign_assignments=denominator,
        p_numerator=extreme, p_denominator=denominator, p_value=extreme/denominator,
        null='Paired differences admit symmetric sign exchangeability under the tested null.',
        statistic='Two-sided absolute arithmetic mean, exhaustive all 2^n sign assignments including zero differences; no Monte Carlo draws.')

def holm_adjust(p_values):
    require(type(p_values) is list and len(p_values) > 0, 'NONEMPTY_MULTIPLICITY_FAMILY')
    ps = []
    for p in p_values:
        require(type(p) in (int,float) and math.isfinite(p) and 0 <= p <= 1, 'VALID_P_VALUE')
        ps.append(fractions.Fraction(str(p)))
    adjusted = [None]*len(ps)
    ceiling = fractions.Fraction(0)
    for rank, i in enumerate(sorted(range(len(ps)), key=lambda i:(ps[i],i))):
        ceiling = max(ceiling, min(fractions.Fraction(1),(len(ps)-rank)*ps[i]))
        adjusted[i] = float(ceiling)
    return adjusted

def rate(count, joint_steps, n_agents, *, deliveries=False):
    require(type(count) is int and count >= 0 and type(joint_steps) is int and joint_steps > 0 and type(n_agents) is int and n_agents > 0, 'COUNT_AND_UNIT_DENOMINATORS')
    return 1000*count/(joint_steps if deliveries else joint_steps*n_agents)

def summary(values):
    require(len(values) == 10 and all(math.isfinite(x) for x in values), 'TEN_FINITE_CLUSTER_VALUES')
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values), minimum=min(values), maximum=max(values),
                seed_values=values, descriptive=True)

def blank_stats():
    from .engine import COUNT_KEYS
    return dict(worlds=0,joint_steps=0,agent_decisions=0,counts=dict.fromkeys(COUNT_KEYS,0),task_reward_sum=0.0)


def add_stats(target, source, *, world=False):
    target['worlds'] += int(world)
    target['joint_steps'] += source['joint_steps']
    target['agent_decisions'] += source['agent_decisions']
    for k in target['counts']:
        x=source['counts'][k]
        require(type(x) is int and 0 <= x <= source['agent_decisions'],'BOUNDED_RAW_ENDPOINT_COUNT')
        if k.endswith('_joint_steps') or k=='nonadditive_stage_steps':
            require(x <= source['joint_steps'],'JOINT_ENDPOINT_COUNT')
        target['counts'][k] += x
    reward=source['task_reward_sum']
    require(type(reward) in (int,float) and math.isfinite(reward),'FINITE_TASK_RETURN')
    target['task_reward_sum'] += reward


def stratum_endpoints(stats):
    steps, agents=stats['joint_steps'],stats['agent_decisions']
    require(agents==4*steps and steps>0,'STRATUM_UNIT_DENOMINATORS')
    result={e:1000*stats['counts'][c]/(steps if e.startswith('deliveries') else agents) for e,c in RATE_COUNTS.items()}
    for label,count in [('institutional_bad_joint_fraction','institutional_bad_joint_steps'),
                        ('quota_bad_joint_fraction','quota_bad_joint_steps'),('capacity_bad_joint_fraction','capacity_bad_joint_steps'),
                        ('proposed_compliance_fraction','proposed_compliant_joint_steps'),('executed_compliance_fraction','executed_compliant_joint_steps')]:
        result[label]=stats['counts'][count]/steps
    result['raw_quota_flags_per_1000_agent_decisions']=1000*stats['counts']['raw_quota_violating_agent_decisions']/agents
    result['raw_capacity_excess_per_1000_joint_steps']=1000*stats['counts']['raw_capacity_excess']/steps
    result['task_reward_per_joint_step']=stats['task_reward_sum']/steps
    return result


def analyze_summaries(protocol, summaries):
    """Pure summary arithmetic; scientific CLI additionally requires raw-file audit."""
    _, planned, seeds=validate_plan(protocol)
    require(type(summaries) is list and len(summaries)==1200,'ALL_1200_CASE_SUMMARIES_REQUIRED')
    expected={c['case_id']:c for c in planned}
    seen=set();accum={};phase_accum={};half_accum={}
    from .engine import COUNT_KEYS
    for s in summaries:
        c=s['case']
        require(c['case_id'] in expected and c==expected[c['case_id']] and c['case_id'] not in seen,'EXACT_UNIQUE_EVALUATION_SUMMARY')
        require(s['status']=='PASS' and s['scientific'] is True and s['joint_steps']==256 and s['agent_decisions']==1024 and s['config']==PROFILES[c['profile_id']],'COMPLETE_SCIENTIFIC_SUMMARY')
        seen.add(c['case_id']);key=(c['method'],c['training_seed'],c['profile_id'])
        add_stats(accum.setdefault(key,blank_stats()),s,world=True)
        require(len(s['phase_summaries'])==64 and len(s['half_summaries'])==2,'ALL_PHASE_AND_HALF_SUMMARIES_REQUIRED')
        for kind,bins,count,steps,target in [('phase',s['phase_summaries'],64,4,phase_accum),('half',s['half_summaries'],2,128,half_accum)]:
            require([r[kind] for r in bins]==list(range(count)),'CONTIGUOUS_PHASE_OR_HALF_LABELS')
            require(all(r['joint_steps']==steps and r['agent_decisions']==4*steps for r in bins),'PHASE_OR_HALF_UNIT_COUNTS')
            for k in COUNT_KEYS:require(sum(r['counts'][k] for r in bins)==s['counts'][k],'PHASE_OR_HALF_COUNT_CLOSURE')
            require(abs(math.fsum(r['task_reward_sum'] for r in bins)-s['task_reward_sum'])<=1e-9,'PHASE_OR_HALF_RETURN_CLOSURE')
            for r in bins:add_stats(target.setdefault(key+(r[kind],),blank_stats()),r,world=True)
    require(seen==set(expected),'COMPLETE_CASE_LEDGER_CLOSURE')
    profile_rows=[];seed_rows=[];lookup={};integer_totals={}
    for method in METHODS:
        for seed in seeds:
            endpoints=None;rawtotals=dict.fromkeys(COUNT_KEYS,0)
            for profile in PROFILES:
                a=accum[(method,seed,profile)]
                require(a['worlds']==5 and a['joint_steps']==1280 and a['agent_decisions']==5120,'COMPLETE_FIVE_WORLD_PROFILE_CELL')
                require(a['counts']['executed_compliant_joint_steps']==1280,'ALL_EXECUTED_ACTIONS_CHECKED')
                values=stratum_endpoints(a);values['task_return_per_episode']=a['task_reward_sum']/5
                row=dict(method=method,training_seed=seed,profile_id=profile,**a,**values);profile_rows.append(row)
                if endpoints is None:endpoints=dict.fromkeys(values,0.0)
                for e,v in values.items():endpoints[e]+=v/4
                for k in COUNT_KEYS:rawtotals[k]+=a['counts'][k]
            seedrow=dict(method=method,training_seed=seed,profile_weights={p:.25 for p in PROFILES},**endpoints,raw_counts_all_profiles=rawtotals)
            seed_rows.append(seedrow);lookup[(method,seed)]=seedrow
            integer_totals[(method,seed)]=rawtotals['institutional_stage_agent_replacements']
    descriptives=list(stratum_endpoints(next(iter(accum.values()))))+['task_return_per_episode']
    methods={m:{e:summary([lookup[(m,s)][e] for s in seeds]) for e in descriptives} for m in METHODS}
    strata={}
    for label,acc,count,steps in [('phase',phase_accum,64,20),('half',half_accum,2,640)]:
        rows=[]
        for (method,seed,profile,index),a in sorted(acc.items()):
            require(a['worlds']==5 and a['joint_steps']==steps and a['agent_decisions']==4*steps,'COMPLETE_PHASE_OR_HALF_PROFILE_CELL')
            rows.append(dict(method=method,training_seed=seed,profile_id=profile,**{label:index},**a,**stratum_endpoints(a)))
        require(len(rows)==6*10*4*count,'ALL_PHASE_OR_HALF_ROWS_REPORTED')
        strata[label+'_profile_seed_rows']=rows
    comparisons=[]
    for comparator in COMPARATORS:
        diffs=[integer_totals[('directrisk',s)]-integer_totals[(comparator,s)] for s in seeds]
        row=dict(candidate='directrisk',reference=comparator,primary=comparator=='semantic',endpoint='institutional_per_1000_agent_decisions',
            integer_repair_differences=diffs,common_integer_test_denominator_agent_decisions_per_cluster=20480,exact_signflip=exact_signflip(diffs))
        row['paired_endpoint_differences']={e:summary([lookup[('directrisk',s)][e]-lookup[(comparator,s)][e] for s in seeds]) for e in descriptives}
        comparisons.append(row)
    for c,p in zip(comparisons,holm_adjust([c['exact_signflip']['p_value'] for c in comparisons])):
        c.update(holm_adjusted_p_value=p,holm_family_size=4,reject_at_two_sided_familywise_0_05=p<=.05)
    return dict(schema='PCI-M2-FRESH-TRAINING-ANALYSIS-v1',status='PASS',training_seed_clusters=10,evaluation_cases=1200,
        evaluation_joint_steps=307200,evaluation_agent_decisions=1228800,training_seeds=seeds,
        profile_seed_rows=profile_rows,equal_profile_seed_rows=seed_rows,method_descriptives=methods,
        predeclared_institutional_family=comparisons,**strata,
        scope='Matched newly trained policies; four profile weights fixed at one quarter; worlds nested within ten training-seed clusters. Institutional stage Hamming repairs, physical repairs and total repairs remain distinct. All64phase/half and delivery/physical/total/task/compliance/rawflag summaries are descriptive. No task noninferiority, gradient-scale equivalence, new shield theorem, proof assurance or generic superiority follows from a single outcome.')


def analyze(protocol_path, output, audit_path, audit_sha256, receipt):
    protocol_path,output,audit_path,receipt=map(Path,(protocol_path,output,audit_path,receipt))
    require(not receipt.exists() and not receipt.resolve().is_relative_to(output.resolve()), 'NEW_EXTERNAL_ANALYSIS_RECEIPT_REQUIRED')
    require(sha(audit_path)==audit_sha256,'FROZEN_RETAINED_AUDIT_IDENTITY')
    audit=read_json(audit_path)
    manifest=verify_closed_inventory(output)
    require(audit['schema']=='PCI-M2-EVALUATION-RETAINED-AUDIT-v1' and audit['status']=='PASS' and audit['checked_cases']==1200 and audit['checked_joint_steps']==307200,'COMPLETE_RETAINED_AUDIT_REQUIRED')
    require(audit['closed_manifest_sha256']==sha(output/'EVALUATION_COMPLETE.json') and audit['protocol_sha256']==sha(protocol_path)==manifest['protocol_sha256'],'AUDIT_OUTPUT_PROTOCOL_BINDING')
    protocol=read_json(protocol_path)
    summaries=[read_json(output/'cases'/f"{c['case_id']}.json") for c in protocol['evaluation_cases']]
    result=analyze_summaries(protocol,summaries)
    result.update(protocol_sha256=sha(protocol_path),closed_manifest_sha256=sha(output/'EVALUATION_COMPLETE.json'),retained_audit_sha256=audit_sha256)
    provenance=read_json(output/'PROVENANCE.json')
    result['training_work_rows']=[dict(training_case_id=k,**b['training_case'],**b['training_work']) for k,b in sorted(provenance['model_bindings'].items())]
    result['sum_policy_elapsed_seconds']=sum(r['elapsed_seconds'] for r in result['training_work_rows'])
    result['evaluation_elapsed_seconds']=manifest['elapsed_seconds']
    result['work_scope']='All60retainedperpolicycountersandconfigureddevices;sum_policy_elapsed_secondsisnotcohortwalltime. NoCPU/GPUmatchedspeedclaim.'
    write_new_json(receipt,result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','output','audit','receipt'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--audit-sha256',required=True)
    a=p.parse_args()
    r=analyze(a.protocol,a.output,a.audit,a.audit_sha256,a.receipt)
    print(json.dumps({k:r[k] for k in ('schema','status','training_seed_clusters','evaluation_cases')}))

if __name__=='__main__':
    main()
