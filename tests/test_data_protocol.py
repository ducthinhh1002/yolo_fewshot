import csv
import importlib.util
import json
from pathlib import Path
import random
import unittest
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module('prepare_data', 'scripts/prepare_data.py')
evaluate = load_module('evaluate_predictions', 'scripts/evaluate_predictions.py')


class ProtocolRegressionTests(unittest.TestCase):
    def test_patient_sampling_exhausts_patients_before_repeating(self):
        rows = [{'sample_id': f'{p}-{i}', 'patient_id': p} for p in ['a', 'b', 'c'] for i in range(4)]
        selected = prepare.sample_patient_balanced(rows, 7, random.Random(42))
        self.assertEqual(len(set(selected)), 7)
        self.assertEqual(len({s.split('-')[0] for s in selected[:3]}), 3)
        self.assertEqual(selected, prepare.sample_patient_balanced(rows, 7, random.Random(42)))

    def test_exact_duplicate_priority_and_label_conflict(self):
        def row(id_, split, label):
            return {'sample_id': id_, 'dataset': 'cnmc', 'split': split, 'label': label, 'pixel_sha256': 'same', 'dhash': '1234567890abcdef', 'status': 'retained'}
        rows = [row('train', 'train', 0), row('test', 'test', 0)]
        prepare.audit_duplicates(rows, 2)
        self.assertEqual(rows[0]['status'], 'excluded_exact_duplicate')
        self.assertEqual(rows[1]['status'], 'retained')
        rows = [row('train', 'train', 0), row('test', 'test', 1)]
        prepare.audit_duplicates(rows, 2)
        self.assertTrue(all(r['status'] == 'excluded_conflicting_duplicate' for r in rows))

    def test_perceptual_match_is_reported_without_deleting_similar_cells(self):
        rows = [dict(sample_id='target', dataset='taleqani', split='external_test', label=0, pixel_sha256='a', dhash='1234567890abcdef', status='retained'), dict(sample_id='source', dataset='cnmc', split='train', label=0, pixel_sha256='b', dhash='1234567890abcdee', status='retained')]
        records = prepare.audit_duplicates(rows, 2)
        self.assertEqual(rows[0]['status'], 'retained')
        self.assertEqual(rows[1]['status'], 'retained')
        self.assertEqual(records[0]['reason'], 'review_potential_near_duplicate_not_excluded')

    def test_patient_leakage_raises(self):
        rows = [{'sample_id': f'{split}-{label}', 'patient_id': 'shared' if label == 0 else split, 'dataset': 'cnmc', 'split': split, 'label': label, 'status': 'retained'} for split in ['train', 'val', 'test', 'external_test'] for label in [0, 1]]
        with self.assertRaisesRegex(ValueError, 'Patient leakage'):
            prepare.validate(rows)

    def test_metrics_positive_class_and_threshold(self):
        result = evaluate.metrics([0, 0, 1, 1], [.1, .7, .5, .9])
        self.assertEqual(result['tp'], 2)
        self.assertEqual(result['fp'], 1)
        self.assertEqual(result['sensitivity'], 1)
        self.assertEqual(result['specificity'], .5)
        self.assertAlmostEqual(result['positive_f1'], .8)


class GeneratedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / 'data/manifests/summary.json').exists():
            raise RuntimeError('Generate real dataset artifacts before running integration checks')
        with (ROOT / 'data/manifests/classification.csv').open(encoding='utf-8') as stream:
            cls.rows = list(csv.DictReader(stream))
        cls.by_id = {r['sample_id']: r for r in cls.rows}

    def test_no_patient_or_pixel_leakage(self):
        patients, pixels = defaultdict(set), defaultdict(set)
        for row in self.rows:
            if row['status'] != 'retained':
                continue
            self.assertTrue((ROOT / row['path']).is_file())
            if row['patient_id']:
                patients[row['patient_id']].add(row['split'])
            pixels[row['pixel_sha256']].add(row['split'])
        self.assertTrue(all(len(s) == 1 for s in patients.values()))
        self.assertTrue(all(len(s) == 1 for s in pixels.values()))

    def test_classification_episodes_have_exact_budget_and_nested_support(self):
        with (ROOT / 'data/episodes/classification.jsonl').open(encoding='utf-8') as stream:
            episodes = [json.loads(line) for line in stream]
        self.assertEqual(len(episodes), 1200)
        trials = defaultdict(dict)
        for episode in episodes:
            k = episode['shots_per_class']
            ids = episode['support_ids']
            self.assertEqual(len(set(ids)), 2*k)
            support = [self.by_id[i] for i in ids]
            self.assertTrue(all(r['split'] == 'train' and r['status'] == 'retained' for r in support))
            self.assertEqual(sum(r['label'] == '1' for r in support), k)
            trials[episode['trial']][k] = set(ids)
        for support in trials.values():
            self.assertTrue(support[1] < support[5] < support[10] < support[20])

    def test_detection_support_has_exact_box_budget(self):
        with (ROOT / 'data/manifests/bccd_boxes.csv').open() as stream:
            counts = defaultdict(int)
            for row in csv.DictReader(stream):
                counts[row['sample_id']] += 1
        with (ROOT / 'data/splits/bccd_train.csv').open(encoding='utf-8') as stream:
            train_ids = {r['sample_id'] for r in csv.DictReader(stream)}
        with (ROOT / 'data/episodes/detection.jsonl').open() as stream:
            episodes = [json.loads(line) for line in stream]
        self.assertEqual(len(episodes), 1200)
        for e in episodes:
            self.assertEqual(len(set(e['support_ids'])), e['shots_per_class'])
            self.assertTrue(set(e['support_ids']) <= train_ids)
            self.assertEqual(sum(counts[i] for i in e['support_ids']), e['shots_per_class'])

    def test_all_frozen_artifact_hashes_match(self):
        import hashlib
        lock = json.loads((ROOT / 'data/provenance/manifests.lock.json').read_text())
        self.assertGreaterEqual(len(lock), 14)
        for path, expected in lock.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)


if __name__ == '__main__':
    unittest.main()
