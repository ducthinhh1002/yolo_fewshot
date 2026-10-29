"""Read fixed audited splits without implicit resplitting or label inference."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest empty or duplicate sample IDs')
    if any(r['status'] != 'retained' for r in rows):
        raise ValueError('Only retained rows are allowed in model inputs')
    return rows


def balanced_subset(rows, per_class, seed):
    from random import Random
    from scripts.prepare_data import sample_patient_balanced
    rng = Random(seed)
    selected = []
    for label in [0, 1]:
        pool = [r for r in rows if int(r['label']) == label]
        if len(pool) < per_class:
            raise ValueError('Requested subset exceeds available class images')
        ids = set(sample_patient_balanced(pool, per_class, rng))
        selected.extend(r for r in pool if r['sample_id'] in ids)
    return selected


def classification_tensor(image, size=224, mean=(.485, .456, .406), std=(.229, .224, .225)):
    image = image.convert('RGB')
    thumb = ImageOps.contain(image, (size, size), Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (size, size), (0, 0, 0))
    canvas.paste(thumb, ((size-thumb.width)//2, (size-thumb.height)//2))
    tensor = torch.from_numpy(np.array(canvas, dtype=np.float32).transpose(2, 0, 1).copy()) / 255
    return (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


class CellDataset(Dataset):
    def __init__(self, manifest=None, *, rows=None, mean=(.485, .456, .406), std=(.229, .224, .225)):
        self.rows = read_manifest(manifest) if rows is None else rows
        self.mean, self.std = mean, std
        if any(int(r['label']) not in [0, 1] for r in self.rows):
            raise ValueError('Invalid normalized binary label')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row['path']) as image:
            pixels = classification_tensor(image, mean=self.mean, std=self.std)
        return {'pixel_values': pixels, 'label': int(row['label']), 'sample_id': row['sample_id'], 'patient_id': row['patient_id'], 'path': row['path']}


class WBCDataset(Dataset):
    """Original-resolution tensors and zero-based half-open xyxy WBC boxes."""
    def __init__(self, manifest, annotations='data/manifests/bccd_boxes.csv'):
        self.rows = read_manifest(manifest)
        self.boxes = {}
        with (ROOT / annotations).open(encoding='utf-8') as f:
            for b in csv.DictReader(f):
                self.boxes.setdefault(b['sample_id'], []).append([float(b[k]) for k in ['xmin', 'ymin', 'xmax', 'ymax']])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row['path']) as im:
            pixels = torch.from_numpy(np.array(im.convert('RGB')).transpose(2, 0, 1).copy()).float()/255
        boxes = torch.tensor(self.boxes.get(row['sample_id'], []), dtype=torch.float32).reshape(-1, 4)
        return pixels, {'boxes': boxes, 'labels': torch.zeros(len(boxes), dtype=torch.int64), 'sample_id': row['sample_id']}


def detection_collate(batch):
    return tuple(zip(*batch))
