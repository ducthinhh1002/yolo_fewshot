import unittest
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from bloodcells.data import CellDataset, WBCDataset, classification_tensor, detection_collate, read_manifest, balanced_subset


class LoaderTests(unittest.TestCase):
    def test_padding_preserves_aspect_and_normalizes(self):
        im = Image.new('RGB', (100, 50), 'white')
        x = classification_tensor(im, size=224, mean=(0,0,0), std=(1,1,1))
        self.assertEqual(tuple(x.shape), (3,224,224))
        self.assertEqual(float(x[:,0,:].max()), 0)
        self.assertEqual(float(x[:,112,:].min()), 1)

    def test_actual_classification_batch(self):
        rows = balanced_subset(read_manifest('data/splits/cnmc_train.csv'), 2, 20261025)
        batch = next(iter(DataLoader(CellDataset(rows=rows), batch_size=4)))
        self.assertEqual(tuple(batch['pixel_values'].shape), (4,3,224,224))
        self.assertTrue(torch.isfinite(batch['pixel_values']).all())
        self.assertEqual(sorted(batch['label'].tolist()), [0,0,1,1])

    def test_detection_boxes_fit_actual_image(self):
        dataset = WBCDataset('data/splits/bccd_train.csv')
        images, targets = next(iter(DataLoader(dataset, batch_size=2, collate_fn=detection_collate)))
        for im, target in zip(images, targets):
            boxes = target['boxes']
            self.assertGreater(len(boxes), 0)
            self.assertTrue((boxes[:,0] >= 0).all() and (boxes[:,1] >= 0).all())
            self.assertTrue((boxes[:,2] <= im.shape[2]).all() and (boxes[:,3] <= im.shape[1]).all())
            self.assertTrue((boxes[:,2:] > boxes[:,:2]).all())


if __name__ == '__main__':
    unittest.main()
