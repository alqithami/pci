"""Synthetic qualification for prospective estimator statistics; no outcomes."""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import t
import estimator_statistics as s


def rows():
    result = []
    for index, seed in enumerate(s.SEEDS):
        for mode_index, mode in enumerate(s.MODES):
            for base_index, base in enumerate(s.BASE_CANDIDATES):
                result.append({'training_seed': seed, 'candidate_id': base+'_'+mode,
                    'family': 'direct', 'n_agents': 4, 'joint_steps': 1024,
                    'institutional_replacements': 5+13*base_index+index*2+mode_index,
                    'deliveries': 250+30*base_index+index*(mode_index+1)-5*mode_index})
    return result


class StatisticsTests(unittest.TestCase):
    def test_hv_independent_rectangle_inclusion_exclusion(self):
        rng = np.random.default_rng(911121)
        bound = F(1,20)
        for _ in range(100):
            p = [(F(int(rng.integers(0,100)),1000), F(int(rng.integers(0,100)),200)) for _ in range(2)]
            (c1,u1),(c2,u2) = p
            area = (max(F(0),bound-c1)*u1 + max(F(0),bound-c2)*u2
                    - max(F(0),bound-max(c1,c2))*min(u1,u2))/bound
            self.assertEqual(s.two_point_hypervolume(p), area)

    def test_hv_edge_points(self):
        self.assertEqual(s.two_point_hypervolume([(0,F(1,3)),(F(1,20),1)]),F(1,3))
        self.assertEqual(s.two_point_hypervolume([(F(1,10),1),(F(1,5),1)]),0)
        self.assertEqual(s.two_point_hypervolume([(F(1,100),F(1,5)),(F(1,50),F(1,10))]),F(4,25))

    def test_full_endpoint_count_and_points(self):
        data=s.learning_endpoints(rows())
        self.assertEqual(len(data['hypervolumes']),30)
        self.assertEqual(len(data['policy_points']),60)
        point=data['policy_points'][0]
        self.assertEqual(s.fraction(point['cost']),F(point['institutional_replacements'],4096))

    def test_outside_points_retained(self):
        data=rows();data[0]['institutional_replacements']=4096
        out=s.learning_endpoints(data)
        self.assertEqual(len(out['policy_points']),60)
        self.assertEqual(sum(x['outside_primary_rectangle'] for x in out['policy_points']),1)

    def test_missing_and_duplicate_rows_rejected(self):
        for bad in [rows()[:-1],rows()+[rows()[0]]]:
            with self.assertRaises(ValueError):s.learning_endpoints(bad)

    def test_mismatched_mode_evaluation_budgets_rejected(self):
        bad=rows();bad[0]['joint_steps']=2048
        with self.assertRaises(ValueError):s.learning_endpoints(bad)

    def test_wrong_seed_and_twenty_cluster_inputs_rejected(self):
        bad=rows();bad[0]['training_seed']=62001
        with self.assertRaises(ValueError):s.learning_endpoints(bad)
        with self.assertRaises(ValueError):s.exact_sign_flip(range(62001,62021),[F(0)]*20)
        with self.assertRaises(ValueError):s.exact_sign_flip(s.SEEDS[::-1],[F(0)]*10)

    def test_invalid_counts_and_names_rejected(self):
        edits=[('candidate_id','direct_02_other'),('n_agents',8),('family','fixed'),
               ('joint_steps',0),('deliveries',True),('deliveries',4097),
               ('institutional_replacements',-1)]
        for key,value in edits:
            bad=rows();bad[0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):s.learning_endpoints(bad)

    def test_exact_sign_flip_independent_bruteforce(self):
        differences=[F(x,101) for x in [1,-2,0,3,-4,1,1,2,-1,3]]
        cutoff=abs(sum(differences))
        count=sum(abs(sum(sign*x for sign,x in zip(signs,differences)))>=cutoff
                  for signs in product((-1,1),repeat=10))
        actual=s.exact_sign_flip(s.SEEDS,differences)
        self.assertEqual(actual['assignments'],1024)
        self.assertEqual(actual['exceedances'],count)
        self.assertEqual(s.fraction(actual['p_value']),F(count,1024))

    def test_zero_and_constant_differences(self):
        zero=s.contrast_statistics(s.CONTRAST_IDS[0],s.SEEDS,[F(0)]*10)
        self.assertEqual(s.fraction(zero['p_value']),1)
        self.assertEqual(zero['dz']['status'],'UNDEFINED_ZERO_SD')
        self.assertEqual(zero['paired_t95']['status'],'UNDEFINED_ZERO_SD')
        nonzero=s.contrast_statistics(s.CONTRAST_IDS[0],s.SEEDS,[F(2,7)]*10)
        self.assertEqual(s.fraction(nonzero['p_value']),F(2,1024))
        self.assertEqual(s.fraction(nonzero['bootstrap95']['lower']),F(2,7))

    def test_bootstrap_repeat_and_independent_float_percentiles(self):
        differences=[F(x,211) for x in [1,-2,0,3,-4,1,1,2,-1,3]]
        result=s.paired_bootstrap(s.CONTRAST_IDS[1],s.SEEDS,differences)
        self.assertEqual(result,s.paired_bootstrap(s.CONTRAST_IDS[1],s.SEEDS,differences))
        material='PCI-M3-ESTIMATOR-ANALYSIS-20261109-v1/'+s.CONTRAST_IDS[1]
        seed=int.from_bytes(hashlib.sha256(material.encode()).digest()[:8],'big')
        index=np.random.Generator(np.random.PCG64(seed)).integers(0,10,size=(10000,10),dtype=np.int64)
        means=np.asarray([float(x) for x in differences])[index].mean(axis=1)
        expected=np.quantile(means,[.025,.975],method='linear')
        self.assertAlmostEqual(result['lower']['decimal'],expected[0],places=14)
        self.assertAlmostEqual(result['upper']['decimal'],expected[1],places=14)
        self.assertEqual(result['resample_index_sha256'],hashlib.sha256(index.astype('<i8').tobytes()).hexdigest())

    def test_paired_t_and_dz(self):
        differences=[F(x,37) for x in [1,-2,0,3,-4,1,1,2,-1,3]]
        result=s.contrast_statistics(s.CONTRAST_IDS[0],s.SEEDS,differences)
        x=np.asarray([float(v) for v in differences]);mean=x.mean();sd=x.std(ddof=1)
        radius=t.ppf(.975,9)*sd/np.sqrt(10)
        self.assertAlmostEqual(result['dz']['value'],mean/sd,places=14)
        self.assertAlmostEqual(result['paired_t95']['lower'],mean-radius,places=14)
        self.assertAlmostEqual(result['paired_t95']['upper'],mean+radius,places=14)
        self.assertEqual(result['paired_t95']['df'],9)

    def test_holm_ties_and_stepdown(self):
        def record(i,p):return {'contrast_id':s.CONTRAST_IDS[i],'p_value':s.packed(p)}
        out=s.holm2([record(0,F(1,50)),record(1,F(1,50))])
        self.assertEqual([s.fraction(x['adjusted_p_value']) for x in out['contrasts']],[F(1,25)]*2)
        self.assertTrue(all(x['reject'] for x in out['contrasts']))
        out=s.holm2([record(0,F(3,100)),record(1,F(1,25))])
        self.assertEqual([s.fraction(x['adjusted_p_value']) for x in out['contrasts']],[F(3,50)]*2)
        self.assertTrue(all(not x['reject'] for x in out['contrasts']))

    def test_holm_partial_and_refusal(self):
        record={'contrast_id':s.CONTRAST_IDS[0],'p_value':s.packed(F(1,10000))}
        out=s.holm2([record]);self.assertFalse(out['family_complete'])
        self.assertNotIn('adjusted_p_value',out['contrasts'][0]);self.assertNotIn('reject',out['contrasts'][0])
        with self.assertRaises(ValueError):s.holm2([record],require_complete=True)
        with self.assertRaises(ValueError):s.holm2([record,record])
        with self.assertRaises(ValueError):s.holm2([{'contrast_id':'Warehouse Direct minus Fixed','p_value':s.packed(0)}])

    def test_packet_is_json_serializable_and_separate(self):
        out=s.analyze_rows(rows());json.dumps(out,allow_nan=False)
        self.assertEqual(out['family_id'],'SEPARATE_ESTIMATOR_HOLM2')
        self.assertTrue(out['family_complete']);self.assertFalse(out['primary_Holm12_changed'])
        self.assertEqual(len(out['endpoint_data']['policy_points']),60)
        self.assertEqual(len(out['endpoint_data']['estimator_cluster_hypervolumes']),30)
        self.assertEqual([x['contrast_id'] for x in out['contrasts']],list(s.CONTRAST_IDS))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).with_name('ESTIMATOR_STATISTICS_TEST_RESULTS.json'))
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(StatisticsTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'PASS_SYNTHETIC_STATISTICAL_QUALIFICATION' if result.wasSuccessful() else 'FAIL',
            'scope':'Synthetic independent arithmetic and inference controls only; no scientific outcomes or receipt-adapter validation.',
            'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'analysis_id':s.ANALYSIS_ID,'required_runtime':s.RUNTIME,
            'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(s.__file__),Path(__file__)]}}
    with args.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
