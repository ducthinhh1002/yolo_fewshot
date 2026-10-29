"""Integration checks against actual server baseline artifacts when available."""
import csv
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/baseline_v1'


@unittest.skipUnless((OUT/'pipeline_demo.json').exists(), 'Baseline not generated in this checkout')
class BaselineOutputs(unittest.TestCase):
    def test_features_preserve_manifest_order_labels_and_unit_norm(self):
        for split in ['train','val']:
            with (ROOT/f'data/splits/cnmc_{split}.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            values = np.load(OUT/f'cnmc_{split}_features.npz')
            self.assertEqual(values['features'].shape, (len(rows),384))
            self.assertEqual(values['sample_ids'].tolist(), [r['sample_id'] for r in rows])
            self.assertEqual(values['labels'].tolist(), [int(r['label']) for r in rows])
            self.assertTrue(np.isfinite(values['features']).all())
            np.testing.assert_allclose(np.linalg.norm(values['features'],axis=1),1,atol=1e-5)

    def test_all_paired_trials_are_present_and_summary_matches(self):
        with (OUT/'prototype_validation_trials.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        summary = json.loads((OUT/'prototype.json').read_text())
        self.assertEqual(len(rows),1200)
        self.assertFalse(summary['test_used'])
        self.assertEqual(summary['query'],'cnmc_val')
        for k in ['1','5','10','20']:
            group = [r for r in rows if r['shots_per_class'] == k]
            self.assertEqual({int(r['trial']) for r in group},set(range(300)))
            self.assertAlmostEqual(np.mean([float(r['macro_f1']) for r in group]),summary['summary'][k]['macro_f1']['mean'])

    def test_demo_and_detector_do_not_claim_test_results(self):
        detector = json.loads((OUT/'detector.json').read_text())
        self.assertFalse(detector['test_used'])
        self.assertTrue(Path(detector['best_checkpoint']).is_file())
        demo = json.loads((OUT/'pipeline_demo.json').read_text())
        self.assertEqual(len(demo['results']),8)
        for image in demo['results']:
            for cell in image['cells']:
                self.assertTrue(0 <= cell['abnormal_score_uncalibrated'] <= 1)
                x1,y1,x2,y2 = cell['box_xyxy']
                self.assertGreater(x2,x1)
                self.assertGreater(y2,y1)


if __name__ == '__main__': unittest.main()
