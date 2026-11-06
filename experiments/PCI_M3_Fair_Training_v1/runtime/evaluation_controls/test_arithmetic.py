"""Literal arithmetic and rejection controls; no worlds, RNG or Torch import."""
from pathlib import Path
import copy, hashlib, json, sys, tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.actor import SHAPES, VIEW_INDICES, portable_forward, inverse_cdf_actions, sampling_probabilities
from evaluation.analysis import exact_signflip, holm_adjust, rate, analyze_summaries
from evaluation.engine import validate_plan, PROFILES, COUNT_KEYS, checked_hot_output
from evaluation.common import EvaluationError, read_json, write_new_json, sha
from evaluation.retained_audit import verify_closed_inventory


def run(output):
    controls=[]
    def check(name,condition):
        if not condition:
            raise AssertionError(name)
        controls.append(dict(name=name,status='PASS'))
    def reject(name,fn):
        try:
            fn()
        except (EvaluationError,ValueError,KeyError,FileNotFoundError):
            controls.append(dict(name=name,status='PASS'))
        else:
            raise AssertionError('DID_NOT_REJECT:'+name)
    def forbidden(*args,**kwargs):
        raise AssertionError('RNG_OR_WORLD_CALL_FORBIDDEN')
    old_default_rng=np.random.default_rng
    old_seed=np.random.seed
    from evaluation.reference import scalar_world
    old_adapter=scalar_world.WorldAdapter
    np.random.default_rng=np.random.seed=forbidden
    scalar_world.WorldAdapter=forbidden
    try:
        r=exact_signflip([1]*10)
        check('two_sided_exact_minimum_two_of_1024',r['extreme_sign_assignments']==2 and r['sign_assignments']==1024 and r['p_value']==2/1024)
        check('all_zero_pairs_p_one',exact_signflip([0]*10)['p_value']==1)
        check('paired_sign_reversal_same_p',exact_signflip([-1]*10)['p_value']==r['p_value'])
        check('zero_observed_mean_p_one',exact_signflip([1,-1]*5)['p_value']==1)
        check('small_literal_exact_count',exact_signflip([1,2,3])['extreme_sign_assignments']==2)
        check('holm_known_distinct_family',holm_adjust([.01,.04,.03,.9])==[.04,.09,.09,.9])
        check('holm_equal_minimum_family',holm_adjust([2/1024]*4)==[8/1024]*4)
        check('per_agent_repair_unit',rate(4,256,4)==3.90625)
        check('per_joint_delivery_unit',rate(4,256,4,deliveries=True)==15.625)
        reject('reject_zero_denominator',lambda:rate(1,0,4))
        reject('reject_nan_difference',lambda:exact_signflip([float('nan')]*10))
        reject('reject_empty_signflip',lambda:exact_signflip([]))
        reject('reject_invalid_p_value',lambda:holm_adjust([1.1]))
        p=np.array([[0,.5,0,.5,0]]*4,dtype=np.float64)
        check('inverse_cdf_zero_mass_and_exact_boundaries',inverse_cdf_actions(p,np.array([0,.5,.499999999, .999999999])).tolist()==[1,3,1,3])
        check('normalization_divides_once',np.array_equal(sampling_probabilities(p*2),p))
        reject('reject_uniform_outside_unit_interval',lambda:inverse_cdf_actions(p,np.ones(4)))
        arrays={k:np.zeros(shape,dtype=np.float32) for k,shape in SHAPES.items()}
        for i,cols in enumerate(VIEW_INDICES):arrays['masks'][i,list(cols)]=1
        logits,prob=portable_forward(arrays,np.zeros((4,5,23),dtype=np.float32))
        check('zero_parameter_forward_uniform',np.array_equal(logits,np.zeros((4,5),dtype=np.float32)) and np.array_equal(prob,np.full((4,5),np.float32(.2))))
        protocol=read_json(ROOT/'design/M2_PROTOCOL.json')
        training,cases,seeds=validate_plan(protocol)
        check('draft_exact_full_cross_60_1200_ten_seeds',len(training)==60 and len(cases)==1200 and len(seeds)==10)
        rows=[]
        arm_offset={'directrisk':0,'semantic':1,'fixed':2,'adaptive':3,'consensus':4,'task':5}
        for c in cases:
            counts=dict.fromkeys(COUNT_KEYS,0)
            counts.update(institutional_stage_agent_replacements=5+arm_offset[c['method']],physical_agent_replacements=8,total_agent_replacements=13+arm_offset[c['method']],
                deliveries=4,institutional_bad_joint_steps=3,quota_bad_joint_steps=3,capacity_bad_joint_steps=1,proposed_compliant_joint_steps=250,executed_compliant_joint_steps=256)
            phases=[dict(phase=j,joint_steps=4,agent_decisions=16,counts={k:v//64+int(j<v%64) for k,v in counts.items()},task_reward_sum=.25) for j in range(64)]
            halves=[dict(half=j,joint_steps=128,agent_decisions=512,counts={k:v//2+int(j<v%2) for k,v in counts.items()},task_reward_sum=8.) for j in range(2)]
            rows.append(dict(status='PASS',scientific=True,case=copy.deepcopy(c),config=PROFILES[c['profile_id']],joint_steps=256,agent_decisions=1024,
                counts=counts,task_reward_sum=16.,phase_summaries=phases,half_summaries=halves))
        a=analyze_summaries(protocol,rows)
        primary=a['predeclared_institutional_family'][0]
        check('all_four_profile_weights_fixed',all(r['profile_weights']=={p:.25 for p in PROFILES} for r in a['equal_profile_seed_rows']))
        check('integer_paired_repair_test_denominator',primary['integer_repair_differences']==[-20]*10 and primary['common_integer_test_denominator_agent_decisions_per_cluster']==20480)
        check('paired_primary_rate_exact',primary['paired_endpoint_differences']['institutional_per_1000_agent_decisions']['mean']==-1000/1024)
        check('complete_four_comparison_family',len(a['predeclared_institutional_family'])==4 and all(r['holm_adjusted_p_value']==8/1024 for r in a['predeclared_institutional_family']))
        check('all_15360_phase_rows_and480_half_rows',len(a['phase_profile_seed_rows'])==15360 and len(a['half_profile_seed_rows'])==480)
        check('task_return_and_compliance_descriptives',a['method_descriptives']['directrisk']['task_return_per_episode']['mean']==16.0 and a['method_descriptives']['directrisk']['proposed_compliance_fraction']['mean']==250/256)
        check('profile_rate_denominators',all(r['joint_steps']==1280 and r['agent_decisions']==5120 for r in a['profile_seed_rows']))
        reject('reject_incomplete_summary_cohort',lambda:analyze_summaries(protocol,rows[:-1]))
        bad=copy.deepcopy(rows);bad[-1]=copy.deepcopy(bad[0])
        reject('reject_duplicated_summary_case',lambda:analyze_summaries(protocol,bad))
        bad=copy.deepcopy(rows);bad[0]['agent_decisions']=256
        reject('reject_joint_vs_agent_unit_confusion',lambda:analyze_summaries(protocol,bad))
        with tempfile.TemporaryDirectory(prefix='pci_m2_owned_hot_controls_') as hd:
            hot=Path(hd);marker=hot/'HOT_ROOT_OWNERSHIP.json'
            body=dict(schema='PCI-M2-OWNED-HOT-ROOT-v1',created_by_controller=True,study_id=protocol['study_id'],
                protocol_sha256=sha(ROOT/'design/M2_PROTOCOL.json'),project_root=str(ROOT),owned_hot_output_root=str(hot))
            marker.write_text(json.dumps(body))
            lock=dict(owned_hot_output_root=str(hot),owned_hot_output_marker_sha256=sha(marker),protocol_sha256=body['protocol_sha256'])
            check('accept_new_owned_marker_bound_hot_subdirectory',checked_hot_output(ROOT,hot/'evaluation',lock,protocol)==hot/'evaluation')
            reject('reject_relative_live_output',lambda:checked_hot_output(ROOT,Path('evaluation'),lock,protocol))
            (hot/'existing').mkdir()
            reject('reject_existing_live_output',lambda:checked_hot_output(ROOT,hot/'existing',lock,protocol))
            reject('reject_source_tree_live_output',lambda:checked_hot_output(ROOT,ROOT/'would_be_live',lock,protocol))
            marker.write_text(json.dumps(body|{'study_id':'wrong-study'}))
            reject('reject_changed_hot_root_marker',lambda:checked_hot_output(ROOT,hot/'evaluation',lock,protocol))
        with tempfile.TemporaryDirectory(prefix='pci_m2_literal_audit_controls_') as td:
            td=Path(td)
            (td/'EVALUATION_COMPLETE.json').write_text(json.dumps(dict(schema='PCI-M2-EVALUATION-CLOSED-MANIFEST-v1',status='PASS',scientific=True,complete_cases=1,joint_steps=256,agent_decisions=1024,files={})))
            reject('reject_incomplete_retained_cohort',lambda:verify_closed_inventory(td))
            (td/'bad.json').write_text('{"x":1,"x":2}')
            reject('reject_duplicate_json_key',lambda:read_json(td/'bad.json'))
            (td/'bad.json').write_text('{"x":NaN}')
            reject('reject_nonfinite_json',lambda:read_json(td/'bad.json'))
    finally:
        np.random.default_rng=old_default_rng
        np.random.seed=old_seed
        scalar_world.WorldAdapter=old_adapter
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'evaluation').rglob('*.py')) if '__pycache__' not in p.parts}
    report=dict(schema='PCI-M2-EVALUATION-ARITHMETIC-CONTROLS-v1',status='PASS',controls=controls,source_sha256=files,
        frozen_protocol_sha256=sha(ROOT/'design/M2_PROTOCOL.json'),new_worlds=0,task_transitions=0,RNG_calls=0,Torch_imports=0,training_GPU_cloud_calls=0,
        scope='Literal arithmetic, sampler/portable-forward controls, complete-family/unit checks and fail-closed malformed/incomplete-cohort rejection. Synthetic summaries are constructed numbers, not observed policy outcomes. No test of actual Torch/native execution or performance.')
    write_new_json(output,report)
    print(json.dumps({'status':'PASS','controls':len(controls),'receipt':str(output),'receipt_sha256':sha(output)}))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
