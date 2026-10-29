import json
import csv
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT/'runs/devit_smoke/result.json').exists(),'Official DE-ViT smoke output not present')
class DEViTOutputs(unittest.TestCase):
    def test_official_checkpoint_and_valid_forward(self):
        d=json.loads((ROOT/'runs/devit_smoke/result.json').read_text())
        self.assertEqual(d['status'],'passed')
        self.assertEqual(d['upstream_commit'],'9cab390fbe78b3bc8d62c224411dce1c71056cf9')
        self.assertEqual(d['missing_learned_parameters'],[])
        self.assertFalse(d['test_used'])
        self.assertLess(d['memory']['peak_reserved_gib'],16)
        self.assertEqual([r['shots_per_class'] for r in d['records']],[1,5,10,20])

    def test_support_query_budgets_and_nesting(self):
        d=json.loads((ROOT/'runs/devit_smoke/result.json').read_text())
        with (ROOT/'data/episodes/detection.jsonl').open() as stream: canonical={r['shots_per_class']:r for r in map(json.loads,stream) if r['trial']==0}
        old=set()
        query=d['records'][0]['query_ids']
        with (ROOT/'data/splits/bccd_val.csv').open() as stream: val=list(csv.DictReader(stream))
        self.assertEqual(query,[r['sample_id'] for r in val[:8]])
        for record in d['records']:
            k=record['shots_per_class'];support=record['support_ids']
            self.assertEqual(support,canonical[k]['support_ids'])
            self.assertEqual(len(set(support)),k)
            self.assertTrue(old<=set(support));old=set(support)
            self.assertFalse(set(support)&set(query))
            self.assertEqual(record['query_ids'],query)
            self.assertEqual(len(record['results']),8)
            self.assertTrue(np.isfinite(record['prototype']).all())
            for image in record['results']:
                self.assertEqual(len(image['boxes_xyxy']),len(image['scores']))
                self.assertTrue(all(0<=s<=1 for s in image['scores']))
                for box in image['boxes_xyxy']:
                    self.assertEqual(len(box),4)
                    self.assertTrue(np.isfinite(box).all())
                    self.assertTrue(0<=box[0]<box[2]<=640)
                    self.assertTrue(0<=box[1]<box[3]<=480)


if __name__=='__main__':unittest.main()
