"""Synthetic integration of exact endpoints with inference/packet revalidation.

The receipt reader is replaced with an explicit artificial160-row object here;
its actual producer-bound input validation is separately exercised by the full
warehouse/RWARE adapter fixtures. No scientific input path is read.
"""
import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import confirmation_analysis as ca

ROOT=Path(__file__).resolve().parent


def synthetic_data():
    ids=[f'{family}_{i:02d}' for family in ca.FAMILIES for i in (0,1)]
    selection={'schema':'PCI-DEVELOPMENT-FRONTIER-SELECTION-v1','status':'DEVELOPMENT_ONLY_SELECTION',
               'study_domain':'warehouse','primary_test_data_opened':False,'selected_candidate_ids':ids}
    points={'fixed':(F(1,64),F(1,4)),'direct':(F(1,128),F(3,8)),
            'lagrangian':(F(3,128),F(5,16)),'shieldaware':(F(0),F(3,16))}
    rows=[]
    for seed in ca.SEEDS['warehouse']:
        for family in ca.FAMILIES:
            cost,utility=points[family]
            for i in (0,1):
                rows.append({'candidate_id':f'{family}_{i:02d}','family':family,'training_seed':seed,
                    'domain':'warehouse','n_agents':4,'joint_steps':20480,
                    'institutional_replacements':int(cost*81920),'deliveries':int(utility*81920),
                    'cohort_complete':True,'evaluation_complete':True})
    return {'domain':'warehouse','selection':selection,'seed_rows':rows,'profile_rows':[],
            'provenance':{'synthetic_validation_only':True,'scope':'ARTIFICIAL NUMERICAL CONTROL'}}


class PipelineTests(unittest.TestCase):
    def test_exact_learning_endpoints_and_missing_duplicate_cells(self):
        data=synthetic_data();out=ca.learning_endpoints(data['seed_rows'],data['selection'],'warehouse')
        expected={'fixed':F(11,64),'direct':F(81,256),'lagrangian':F(85,512),'shieldaware':F(3,16)}
        self.assertEqual(len(out),80)
        for seed in ca.SEEDS['warehouse']:
            for family,value in expected.items():self.assertEqual(out[seed,family],value)
        for rows in (data['seed_rows'][:-1],data['seed_rows']+[data['seed_rows'][0]]):
            with self.assertRaises(ValueError):ca.learning_endpoints(rows,data['selection'],'warehouse')

    def test_packet_rebuild_reopens_receipt_reader_and_rejects_endpoint_tampering(self):
        data=synthetic_data()
        with tempfile.TemporaryDirectory(prefix='pci_numerical_pipeline_synthetic_') as tmp:
            paths={k:str(Path(tmp)/k) for k in ('cohort','source_root','selection_path','case_manifest_path')}
            with patch('learning_receipts.load_learning_receipts',return_value=data) as reader, \
                 patch.object(ca,'verify_implementation_lock',return_value={'synthetic_control':True}):
                packet=ca.learning_packet(**paths)
                self.assertEqual(packet['status'],'DESCRIPTIVE_PARTIAL_FAMILY')
                self.assertFalse(ca.has_decision_fields(packet))
                self.assertEqual(len(packet['endpoint_data']['family_cluster_hypervolumes']),80)
                self.assertEqual(len(packet['endpoint_data']['policy_points']),160)
                for record,effect in zip(packet['contrasts'],(F(37,256),F(77,512),F(33,256))):
                    self.assertEqual(ca.fraction(record['mean_effect']),effect)
                    self.assertEqual(ca.fraction(record['p_value']),F(2,2**20))
                    self.assertEqual(record['dz']['status'],'UNDEFINED_ZERO_PAIRED_SD')
                path=Path(tmp)/'SYNTHETIC_PACKET.json';path.write_text(json.dumps(packet))
                contrasts,_=ca.verified_learning_packet(path)
                self.assertEqual(contrasts,packet['contrasts'])
                self.assertEqual(reader.call_count,2)
                tampered=copy.deepcopy(packet)
                tampered['endpoint_data']['policy_points'][0]['deliveries']+=1
                path.write_text(json.dumps(tampered))
                with self.assertRaisesRegex(ValueError,'LEARNING_PACKET_RECOMPUTATION_DIFFERS'):
                    ca.verified_learning_packet(path)
                with self.assertRaisesRegex(ValueError,'COMPLETE_12_CONTRAST_FAMILY_REQUIRED'):
                    ca.holm12(contrasts,require_complete=True)

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='pci_write_once_control_') as tmp:
            path=Path(tmp)/'SYNTHETIC.json';ca.write_once(path,{'synthetic':True})
            with self.assertRaises(FileExistsError):ca.write_once(path,{'synthetic':False})
            self.assertEqual(json.loads(path.read_text()),{'synthetic':True})


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PipelineTests))
    receipt={'schema':'PCI-CONFIRMATION-NUMERICAL-PIPELINE-CONTROLS-v1',
        'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
        'synthetic_only':True,'scientific_results_read':False,'fixtures_retained':False,
        'core_source_sha256':hashlib.sha256((ROOT/'confirmation_analysis.py').read_bytes()).hexdigest(),
        'test_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'PIPELINE_TEST_RESULTS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
