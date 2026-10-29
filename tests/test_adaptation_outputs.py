import csv
import json
from pathlib import Path
import unittest
import numpy as np
from scripts.analyze_errors import match_boxes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/adaptation_v1'


class DetectionMatching(unittest.TestCase):
    def test_duplicate_predictions_are_not_two_true_positives(self):
        matched,false,missed=match_boxes([[0,0,10,10,.9],[0,0,10,10,.8]],[[0,0,10,10]],.5)
        self.assertEqual(len(matched),1); self.assertEqual(false,[1]); self.assertEqual(missed,[])

    def test_no_predictions_counts_missed_cells(self):
        self.assertEqual(match_boxes([],[[0,0,10,10]],.5),([],[],[0]))


@unittest.skipUnless((OUT/'comparison.json').exists(),'Adaptation run not present')
class ActualAdaptationOutputs(unittest.TestCase):
    def test_paired_probabilities_and_cache_parity(self):
        data=np.load(OUT/'validation_scores.npz')
        self.assertEqual(data['episodes'].shape,(120,2))
        self.assertEqual(len(set(map(tuple,data['episodes']))),120)
        for method in ['baseline','head_only','protonet','lora']:
            self.assertEqual(data[method].shape,(120,3554))
            self.assertTrue(np.isfinite(data[method]).all())
            self.assertTrue(((data[method]>=0)&(data[method]<=1)).all())
        cache=json.loads((OUT/'cache.json').read_text())
        for split in ['train','val']:
            self.assertLess(cache[split]['parity_max_abs'],1e-5)

    def test_matched_baseline_agrees_with_previous_run(self):
        with (ROOT/'runs/baseline_v1/prototype_validation_trials.csv').open() as stream:
            original={(int(r['trial']),int(r['shots_per_class'])):float(r['macro_f1']) for r in csv.DictReader(stream)}
        with (OUT/'comparison_trials.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['method']=='baseline':
                    key=(int(row['trial']),int(row['shots_per_class']))
                    self.assertAlmostEqual(float(row['macro_f1']),original[key],places=6)

    def test_summary_and_no_test_use(self):
        data=json.loads((OUT/'comparison.json').read_text())
        self.assertFalse(data['test_used']); self.assertEqual(data['query'],'cnmc_val')
        self.assertEqual(data['lora_trainable_adapter_parameters'],6144)
        self.assertLess(data['memory']['peak_reserved_gib'],16)
        self.assertEqual(set(data['summary']),{'baseline','head_only','protonet','lora'})


if __name__=='__main__': unittest.main()
