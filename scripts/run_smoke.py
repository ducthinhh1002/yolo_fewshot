"""Audit -> DINOv2 features -> one-epoch YOLO train/val/predict, no test tuning."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bloodcells.data import read_manifest, balanced_subset


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def gpu_budget(config):
    import torch
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    budget = int(config['max_vram_gib'] * 1024**3)
    if budget > 16 * 1024**3:
        raise ValueError('This runner is limited to 16 GiB to remain below 20 GB with CUDA overhead')
    torch.cuda.set_per_process_memory_fraction(min(budget / total, 1.0), 0)
    torch.cuda.reset_peak_memory_stats()


def gpu_stats():
    import torch
    if not torch.cuda.is_available():
        return {'device': 'cpu'}
    return {'device': torch.cuda.get_device_name(0), 'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3, 'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3, 'allocator_limit_gib': 16}


def audit(out):
    lock = json.loads((ROOT/'data/provenance/manifests.lock.json').read_text())
    for path, expected in lock.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen artifact differs: {path}')
    datasets = {}
    groups, pixels = defaultdict(set), defaultdict(set)
    for name in ['cnmc_train', 'cnmc_val', 'cnmc_test', 'aml_external_test', 'bccd_train', 'bccd_val', 'bccd_test']:
        rows = read_manifest(f'data/splits/{name}.csv')
        datasets[name] = {'images': len(rows), 'labels': dict(Counter(r['label'] for r in rows))}
        for row in rows:
            if not (ROOT/row['path']).is_file():
                raise FileNotFoundError(row['path'])
            pixels[row['pixel_sha256']].add(name)
            if row['patient_id']:
                groups[row['patient_id']].add(name)
    if any(len(s)>1 for s in groups.values()) or any(len(s)>1 for s in pixels.values()):
        raise ValueError('Patient or exact-pixel leakage')
    with (ROOT/'data/manifests/bccd_boxes.csv').open(encoding='utf-8') as stream:
        boxes = list(csv.DictReader(stream))
    detection_counts = {}
    for split in ['train', 'val', 'test']:
        ids = {r['sample_id'] for r in read_manifest(f'data/splits/bccd_{split}.csv')}
        counts = Counter(b['sample_id'] for b in boxes if b['sample_id'] in ids)
        detection_counts[split] = {'boxes': sum(counts.values()), 'single_wbc_images': sum(n == 1 for n in counts.values())}
    if min(int(datasets['cnmc_train']['labels'].get(str(c), 0)) for c in [0,1]) < 20 or detection_counts['train']['single_wbc_images'] < 20:
        raise ValueError('Insufficient images for 20-shot experiments')
    with (ROOT/'data/manifests/classification.csv').open(encoding='utf-8') as stream:
        exclusions = Counter(r['status'] for r in csv.DictReader(stream) if r['status'] != 'retained')
    result = {'status': 'passed', 'datasets': datasets, 'detection': detection_counts, 'classification_exclusions': dict(exclusions), 'sufficient_for_20_shot': True, 'patient_overlap': 0, 'cross_split_exact_pixel_overlap': 0, 'near_duplicates': 'review candidates only; dHash cannot prove duplicate clinical cells'}
    save_json(out/'audit.json', result)
    print('Audit passed', flush=True)


def dino(config, out):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoModel
    from bloodcells.data import CellDataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config['seed'])
    gpu_budget(config)
    processor = AutoImageProcessor.from_pretrained(config['dino_model'], revision=config['dino_revision'])
    model = AutoModel.from_pretrained(config['dino_model'], revision=config['dino_revision']).to(device).eval()
    results = {}
    for split, n in [('train', config['classification_train_per_class']), ('val', config['classification_val_per_class'])]:
        rows = balanced_subset(read_manifest(f'data/splits/cnmc_{split}.csv'), n, config['seed'])
        dataset = CellDataset(rows=rows, mean=processor.image_mean, std=processor.image_std)
        loader = DataLoader(dataset, batch_size=config['feature_batch_size'], shuffle=False, num_workers=0)
        features, labels, ids = [], [], []
        with torch.inference_mode():
            for batch in loader:
                x = model(pixel_values=batch['pixel_values'].to(device)).last_hidden_state[:, 0, :]
                features.append(torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy())
                labels.extend(batch['label'].tolist())
                ids.extend(batch['sample_id'])
        values = np.concatenate(features)
        if values.shape != (2*n, model.config.hidden_size) or not np.isfinite(values).all():
            raise ValueError('Invalid feature output')
        np.savez_compressed(out/f'cnmc_{split}_features.npz', features=values, labels=np.asarray(labels), sample_ids=np.asarray(ids))
        save_json(out/f'cnmc_{split}_feature_rows.json', rows)
        results[split] = {'shape': list(values.shape), 'labels': dict(Counter(labels)), 'finite': True}
    save_json(out/'dinov2.json', {'status': 'passed', 'model': config['dino_model'], 'revision': getattr(model.config, '_commit_hash', None), 'device': device, 'feature': 'L2_normalized_CLS', 'preprocessing': 'RGB_aspect_preserving_pad_224_then_pretrained_mean_std', 'outputs': results})
    save_json(out/'dinov2_memory.json', gpu_stats())
    print('DINOv2 extraction passed', flush=True)


def yolo(config, out):
    import torch
    from ultralytics import YOLO
    from bloodcells.data import WBCDataset
    gpu_budget(config)
    sample, target = WBCDataset('data/splits/bccd_train.csv')[0]
    if sample.shape[0] != 3 or target['boxes'].shape[1] != 4:
        raise ValueError('Invalid detection loader output')
    subset = out/'bccd_subset'
    selected = {}
    for split, n in [('train', config['detection_train_images']), ('val', config['detection_val_images'])]:
        rows = read_manifest(f'data/splits/bccd_{split}.csv')
        random.Random(config['seed']).shuffle(rows)
        rows = rows[:n]
        selected[split] = [r['sample_id'] for r in rows]
        for folder in ['images', 'labels']:
            (subset/folder/split).mkdir(parents=True, exist_ok=True)
        for row in rows:
            source = ROOT/row['path']
            shutil.copy2(source, subset/'images'/split/source.name)
            shutil.copy2(ROOT/'data/processed/bccd/labels'/split/(source.stem+'.txt'), subset/'labels'/split/(source.stem+'.txt'))
    yaml = subset/'dataset.yaml'
    yaml.write_text(f'path: {subset.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: WBC\n', encoding='utf-8')
    weights = ROOT/'models'/config['yolo_model']
    weights.parent.mkdir(exist_ok=True)
    model = YOLO(str(weights))
    device = 0 if torch.cuda.is_available() else 'cpu'
    model.train(data=str(yaml), epochs=config['yolo_epochs'], imgsz=config['yolo_image_size'], batch=config['yolo_batch_size'], workers=0, device=device, seed=config['seed'], amp=False, plots=False, project=str(out/'yolo'), name='train', exist_ok=True)
    metrics = model.val(data=str(yaml), split='val', imgsz=config['yolo_image_size'], batch=config['yolo_batch_size'], workers=0, device=device, plots=False, project=str(out/'yolo'), name='validation', exist_ok=True)
    predictions = model.predict(source=str(subset/'images/val'), imgsz=config['yolo_image_size'], device=device, save=True, save_txt=True, project=str(out/'yolo'), name='predictions', exist_ok=True)
    save_json(out/'yolo.json', {'status': 'passed', 'model': config['yolo_model'], 'pretrained_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(), 'epochs': config['yolo_epochs'], 'selected': selected, 'predicted_images': len(predictions), 'metrics_validation_smoke_only': {k: float(v) for k,v in metrics.results_dict.items()}, 'note': 'Pipeline check, not a trained benchmark or test result.'})
    save_json(out/'yolo_memory.json', gpu_stats())
    print('YOLO train/val/predict passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/smoke.json')
    parser.add_argument('--stage', choices=['all', 'audit', 'dino', 'yolo'], default='all')
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text())
    import setproctitle
    setproctitle.setproctitle(config['process_name'])
    out = ROOT/config['output']
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/'runtime.json', {'pid': os.getpid(), 'process_title': setproctitle.getproctitle(), 'linux_comm': Path('/proc/self/comm').read_text().strip() if Path('/proc/self/comm').exists() else None})
    print(f"Process: {setproctitle.getproctitle()} (PID {os.getpid()})", flush=True)
    if args.stage in ['all', 'audit']:
        audit(out)
    if args.stage in ['all', 'dino']:
        dino(config, out)
    if args.stage in ['all', 'yolo']:
        yolo(config, out)
    save_json(out/'environment.json', {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'transformers', 'ultralytics', 'numpy', 'Pillow']})


if __name__ == '__main__':
    main()
