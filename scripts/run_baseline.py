"""Full training-split baselines, selected/evaluated on validation only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_smoke import audit, gpu_budget, gpu_stats, save_json
from bloodcells.data import CellDataset, read_manifest


def features(config, out):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoModel
    gpu_budget(config)
    processor = AutoImageProcessor.from_pretrained(config['dino_model'], revision=config['dino_revision'])
    model = AutoModel.from_pretrained(config['dino_model'], revision=config['dino_revision']).cuda().eval()
    summary = {}
    for split in ['train', 'val']:
        manifest = ROOT/f'data/splits/cnmc_{split}.csv'
        rows = read_manifest(manifest)
        loader = DataLoader(CellDataset(rows=rows, mean=processor.image_mean, std=processor.image_std), batch_size=config['feature_batch_size'], num_workers=0, shuffle=False)
        values, labels, ids = [], [], []
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                x = model(pixel_values=batch['pixel_values'].cuda()).last_hidden_state[:,0,:]
                values.append(torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy())
                labels.extend(batch['label'].tolist())
                ids.extend(batch['sample_id'])
                if i % 25 == 0:
                    print(f'DINO {split}: {len(ids)}/{len(rows)}', flush=True)
        x = np.concatenate(values)
        if x.shape != (len(rows),384) or not np.isfinite(x).all():
            raise ValueError('Invalid full features')
        np.savez_compressed(out/f'cnmc_{split}_features.npz', features=x, labels=np.asarray(labels), sample_ids=np.asarray(ids))
        summary[split] = {'shape': list(x.shape), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    save_json(out/'features.json', {'model': config['dino_model'], 'revision': config['dino_revision'], 'splits': summary, 'memory': gpu_stats(), 'test_used': False})


def prototype(config, out):
    import numpy as np
    import torch
    from scripts.evaluate_predictions import metrics
    gpu_budget(config)
    train = np.load(out/'cnmc_train_features.npz')
    val = np.load(out/'cnmc_val_features.npz')
    index = {str(sample): i for i, sample in enumerate(train['sample_ids'])}
    query = torch.from_numpy(val['features']).cuda()
    values, demo_proto = [], None
    with (ROOT/'data/episodes/classification.jsonl').open() as stream:
        for line in stream:
            e = json.loads(line)
            if e['trial'] >= config['prototype_trials']:
                continue
            support_idx = [index[s] for s in e['support_ids']]
            x, y = train['features'][support_idx], train['labels'][support_idx]
            centers = np.stack([x[y==label].mean(axis=0) for label in [0,1]])
            centers /= np.linalg.norm(centers, axis=1, keepdims=True)
            with torch.inference_mode():
                scores = (query @ torch.from_numpy(centers).cuda().T) / config['prototype_temperature']
                p = scores.softmax(dim=-1)[:,1].cpu().numpy()
            values.append({'trial': e['trial'], 'shots_per_class': e['shots_per_class'], **metrics(val['labels'], p)})
            if e['trial'] == 0 and e['shots_per_class'] == 20:
                demo_proto = centers
    if len(values) != config['prototype_trials']*4:
        raise ValueError('Incomplete paired prototype evaluation')
    with (out/'prototype_validation_trials.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    summary = {}
    for k in [1,5,10,20]:
        group = [r for r in values if r['shots_per_class'] == k]
        summary[k] = {metric: {'mean': float(np.mean([r[metric] for r in group])), 'sd': float(np.std([r[metric] for r in group],ddof=1))} for metric in ['macro_f1','sensitivity','specificity','auroc','auprc']}
    np.save(out/'demo_prototypes.npy', demo_proto)
    save_json(out/'prototype.json', {'method': 'frozen_DINOv2_cosine_nearest_prototype', 'trials': config['prototype_trials'], 'query': 'cnmc_val', 'test_used': False, 'temperature': config['prototype_temperature'], 'summary': summary})
    print('Prototype validation complete', flush=True)


def detector(config, out):
    from ultralytics import YOLO
    gpu_budget(config)
    model = YOLO(str(ROOT/'models'/config['yolo_model']))
    yaml = ROOT/'data/processed/bccd/dataset.yaml'
    model.train(data=str(yaml), epochs=config['yolo_epochs'], patience=config['yolo_patience'], imgsz=config['yolo_image_size'], batch=config['yolo_batch_size'], workers=0, device=0, seed=config['seed'], amp=False, plots=True, project=str(out/'yolo'), name='train', exist_ok=True)
    # Explicitly reload selected best validation checkpoint, not the final epoch.
    checkpoint = Path(model.trainer.best)
    model = YOLO(str(checkpoint))
    result = model.val(data=str(yaml), split='val', imgsz=config['yolo_image_size'], batch=config['yolo_batch_size'], workers=0, device=0, plots=True, project=str(out/'yolo'), name='validation', exist_ok=True)
    save_json(out/'detector.json', {'method': 'YOLO11n_full_detection_supervision', 'train_images':205, 'validation_images':87, 'max_epochs':config['yolo_epochs'], 'best_checkpoint':str(checkpoint), 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'validation': {k:float(v) for k,v in result.results_dict.items()}, 'test_used':False, 'memory':gpu_stats()})
    print('Full YOLO baseline complete', flush=True)


def demo(config, out):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from ultralytics import YOLO
    from bloodcells.data import classification_tensor
    gpu_budget(config)
    detector_info = json.loads((out/'detector.json').read_text())
    yolo = YOLO(detector_info['best_checkpoint'])
    processor = AutoImageProcessor.from_pretrained(config['dino_model'], revision=config['dino_revision'])
    encoder = AutoModel.from_pretrained(config['dino_model'], revision=config['dino_revision']).cuda().eval()
    centers = torch.from_numpy(np.load(out/'demo_prototypes.npy')).cuda()
    rows = read_manifest('data/splits/bccd_val.csv')[:8]
    results = []
    for row in rows:
        predicted = yolo.predict(str(ROOT/row['path']), conf=.25, imgsz=config['yolo_image_size'], device=0, verbose=False)[0]
        cells = []
        with Image.open(ROOT/row['path']) as image:
            for box, confidence in zip(predicted.boxes.xyxy.cpu().tolist(), predicted.boxes.conf.cpu().tolist()):
                x1,y1,x2,y2 = [int(round(v)) for v in box]
                x1,y1,x2,y2 = max(0,x1),max(0,y1),min(image.width,x2),min(image.height,y2)
                if x2 <= x1 or y2 <= y1: continue
                x = classification_tensor(image.crop((x1,y1,x2,y2)), mean=processor.image_mean, std=processor.image_std).unsqueeze(0).cuda()
                with torch.inference_mode():
                    f = torch.nn.functional.normalize(encoder(pixel_values=x).last_hidden_state[:,0,:].float(),dim=-1)
                    p = float(((f@centers.T)/config['prototype_temperature']).softmax(-1)[0,1])
                cells.append({'box_xyxy':[x1,y1,x2,y2], 'wbc_confidence':confidence, 'abnormal_score_uncalibrated':p, 'label':int(p>=.5)})
        results.append({'sample_id':row['sample_id'], 'cells':cells})
    save_json(out/'pipeline_demo.json', {'results':results, 'warning':'Engineering demo only: BCCD has no cell cancer labels and is outside C-NMC classification domain. These are not validated medical predictions.'})
    print('Image -> boxes -> crops -> features -> prototype demo complete', flush=True)


def main():
    import setproctitle
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all','features','prototype','detector','demo'], default='all')
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/baseline.json').read_text())
    setproctitle.setproctitle(config['process_name'])
    if not torch.cuda.is_available(): raise RuntimeError('Run baseline on the authorized GPU server')
    torch.manual_seed(config['seed'])
    out = ROOT/config['output']
    out.mkdir(parents=True,exist_ok=True)
    audit(out)
    for name, function in [('features',features),('prototype',prototype),('detector',detector),('demo',demo)]:
        if args.stage in ['all',name]:
            started = time.monotonic()
            function(config,out)
            print(f'{name}: {time.monotonic()-started:.1f} seconds',flush=True)


if __name__ == '__main__': main()
