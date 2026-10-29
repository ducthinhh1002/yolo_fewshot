"""Build audited manifests, fixed splits, YOLO labels and paired few-shot trials."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import platform
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['sample_id', 'dataset', 'path', 'source_label', 'label', 'patient_id', 'group_id', 'split', 'width', 'height', 'sha256', 'pixel_sha256', 'dhash', 'status']


def write_csv(path, rows, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_row(path, dataset, source_label, label, patient_id, group_id, split):
    with Image.open(path) as im:
        im.load()
        rgb = im.convert('RGB')
        width, height = rgb.size
        pixels = hashlib.sha256(f'{width}x{height}:'.encode() + rgb.tobytes()).hexdigest()
        small = np.asarray(rgb.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        dhash = sum(int(bit) << i for i, bit in enumerate(bits))
    return dict(zip(FIELDS, [dataset + ':' + path.stem, dataset, path.relative_to(ROOT).as_posix(), source_label, label, patient_id, group_id, split, width, height, hashlib.sha256(path.read_bytes()).hexdigest(), pixels, f'{dhash:016x}', 'retained']))


def classification_rows(config):
    tasks = []
    excluded_counts = Counter()
    for path in sorted((ROOT / 'data/raw/cnmc').rglob('*.bmp')):
        fold = next((part for part in path.parts if part in config['cnmc_folds']), None)
        if fold is None:
            excluded_counts['cnmc_unselected_challenge_partitions'] += 1
            continue
        source_label = path.parent.name.lower()
        if source_label not in config['cnmc_label_map']:
            raise ValueError(f'Unknown CNMC label: {path}')
        match = re.fullmatch(r'UID_([^_]+)_([^_]+)_([^_]+)_(all|hem)', path.stem, re.IGNORECASE)
        if not match:
            raise ValueError(f'Cannot parse patient ID: {path.name}')
        if match.group(4).lower() != source_label:
            raise ValueError(f'Filename label differs from folder label: {path}')
        # Namespace by source class: healthy subject H1 and ALL subject 1 differ.
        patient = f'cnmc:{source_label}:{match.group(1)}'
        tasks.append((path, 'cnmc', source_label, config['cnmc_label_map'][source_label], patient, patient, config['cnmc_folds'][fold]))
    if not tasks:
        raise ValueError('No C-NMC training images found; run download_data.py first')
    for path in sorted((ROOT / 'data/raw/aml').rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        label_name = path.parent.name
        if label_name not in config['aml_label_map']:
            excluded_counts['aml_other_morphological_classes'] += 1
            continue
        if not path.stem.startswith(label_name + '_'):
            raise ValueError(f'AML filename/folder label mismatch: {path}')
        tasks.append((path, 'aml', label_name, config['aml_label_map'][label_name], '', 'aml:image:' + path.stem, 'external_test'))
    if not any(task[1] == 'aml' for task in tasks):
        raise ValueError('No AML single-cell images found')
    counts = Counter(task[1] for task in tasks)
    if counts != {'cnmc': 10661, 'aml': 7205}:
        raise ValueError(f'Incomplete or changed source release: {dict(counts)}')
    rows = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for row in executor.map(lambda args: image_row(*args), tasks):
            rows.append(row)
            if len(rows) % 3000 == 0:
                print(f'Checked {len(rows)}/{len(tasks)} selected images', flush=True)
    return rows, excluded_counts


def audit_duplicates(rows, distance):
    """Keep test priority. Quarantine lower-priority copies, never move patients."""
    rank = {'external_test': 0, 'test': 1, 'val': 2, 'train': 3}
    records = []
    exact = defaultdict(list)
    for row in rows:
        exact[row['pixel_sha256']].append(row)
    for group in exact.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda r: (rank[r['split']], r['sample_id']))
        conflict = len({r['label'] for r in group}) > 1
        for row in ordered if conflict else ordered[1:]:
            row['status'] = 'excluded_conflicting_duplicate' if conflict else 'excluded_exact_duplicate'
            records.append({'a': ordered[0]['sample_id'], 'b': row['sample_id'], 'reason': row['status'], 'distance': 0})
    # Four disjoint 16-bit buckets guarantee a candidate for <=2 changed bits.
    buckets = defaultdict(list)
    retained = sorted((r for r in rows if r['status'] == 'retained'), key=lambda r: (rank[r['split']], r['sample_id']))
    for row in retained:
        value = int(row['dhash'], 16)
        candidates = {}
        for i in range(4):
            for other in buckets[(i, (value >> (16*i)) & 65535)]:
                candidates[other['sample_id']] = other
        for other in candidates.values():
            if other['split'] == row['split'] and other['dataset'] == row['dataset']:
                continue
            d = (value ^ int(other['dhash'], 16)).bit_count()
            if d <= distance:
                # Masked cell images have similar black backgrounds: dHash alone
                # is not reliable enough to delete clinical images.
                records.append({'a': other['sample_id'], 'b': row['sample_id'], 'reason': 'review_potential_near_duplicate_not_excluded', 'distance': d})
                break
        if row['status'] == 'retained':
            for i in range(4):
                buckets[(i, (value >> (16*i)) & 65535)].append(row)
    return records


def sample_patient_balanced(rows, count, rng):
    patients = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['sample_id']):
        patients[row['patient_id']].append(row['sample_id'])
    keys = sorted(patients)
    rng.shuffle(keys)
    for values in patients.values():
        rng.shuffle(values)
    sampled = []
    while len(sampled) < count:
        progress = False
        for key in keys:
            if patients[key]:
                sampled.append(patients[key].pop())
                progress = True
                if len(sampled) == count:
                    return sampled
        if not progress:
            raise ValueError('Not enough unique support images')
    return sampled


def build_episodes(rows, config):
    out = ROOT / 'data/episodes'
    out.mkdir(parents=True, exist_ok=True)
    pools = {label: [r for r in rows if r['dataset'] == 'cnmc' and r['split'] == 'train' and r['label'] == label and r['status'] == 'retained'] for label in [0, 1]}
    # Each trial is reused for all methods and both target domains.
    with (out / 'classification.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for trial in range(config['trials']):
            rng = random.Random(config['seed'] + trial)
            supports = {label: sample_patient_balanced(pool, max(config['shots']), rng) for label, pool in pools.items()}
            for shot in config['shots']:
                record = {'trial': trial, 'seed': config['seed'] + trial, 'shots_per_class': shot, 'support_ids': supports[0][:shot] + supports[1][:shot], 'query_manifests': {'in_domain': 'data/splits/cnmc_test.csv', 'cross_source_and_disease_shift': 'data/splits/aml_external_test.csv'}, 'validation_manifest': 'data/splits/cnmc_val.csv'}
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def build_detection(config):
    bases = list((ROOT / 'data/raw/bccd').glob('*/BCCD'))
    if len(bases) != 1:
        raise ValueError('BCCD archive missing or ambiguous')
    base = bases[0]
    rows, boxes = [], []
    yolo = ROOT / 'data/processed/bccd'
    for split in ['train', 'val', 'test']:
        names = (base / 'ImageSets/Main' / (split + '.txt')).read_text().split()
        for name in sorted(names):
            path = base / 'JPEGImages' / (name + '.jpg')
            row = image_row(path, 'bccd', 'WBC', '', '', 'bccd:image:' + name, split)
            rows.append(row)
            tree = ET.parse(base / 'Annotations' / (name + '.xml'))
            lines = []
            for obj in tree.findall('object'):
                if obj.findtext('name') != 'WBC':
                    continue
                raw = [int(obj.findtext('bndbox/' + k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
                x1, y1, x2, y2 = raw[0]-1, raw[1]-1, raw[2], raw[3]
                if not (0 <= x1 < x2 <= row['width'] and 0 <= y1 < y2 <= row['height']):
                    raise ValueError(f'Invalid box: {name} {raw}')
                boxes.append({'sample_id': row['sample_id'], 'split': split, 'class_id': 0, 'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2})
                lines.append(f"0 {(x1+x2)/2/row['width']:.8f} {(y1+y2)/2/row['height']:.8f} {(x2-x1)/row['width']:.8f} {(y2-y1)/row['height']:.8f}")
            label_path = yolo / 'labels' / split / (name + '.txt')
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('BCCD source split overlap')
    if len(rows) != 364:
        raise ValueError(f'Incomplete BCCD release: {len(rows)} images')
    duplicate_records = audit_duplicates(rows, config['near_duplicate_hamming_distance'])
    for split in ['train', 'val', 'test']:
        selected = [r for r in rows if r['split'] == split and r['status'] == 'retained']
        write_csv(ROOT / f'data/splits/bccd_{split}.csv', selected)
        image_dir = yolo / 'images' / split
        image_dir.mkdir(parents=True, exist_ok=True)
        # Materialize only retained images, retaining source image bytes.
        expected = set()
        for row in selected:
            source = ROOT / row['path']
            target = image_dir / source.name
            expected.add(target.name)
            if not target.exists():
                target.write_bytes(source.read_bytes())
        for stale in image_dir.glob('*.jpg'):
            if stale.name not in expected:
                stale.unlink()
    yaml = 'path: ' + yolo.as_posix() + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: WBC\n'
    (yolo / 'dataset.yaml').write_text(yaml, encoding='utf-8')
    write_csv(ROOT / 'data/manifests/bccd.csv', rows)
    write_csv(ROOT / 'data/manifests/bccd_boxes.csv', boxes, ['sample_id', 'split', 'class_id', 'xmin', 'ymin', 'xmax', 'ymax'])
    box_counts = Counter(b['sample_id'] for b in boxes)
    pool = sorted([r for r in rows if r['split'] == 'train' and r['status'] == 'retained' and box_counts[r['sample_id']] == 1], key=lambda r: r['sample_id'])
    if len(pool) < max(config['shots']):
        raise ValueError('Insufficient single-WBC detection support images')
    with (ROOT / 'data/episodes/detection.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for trial in range(config['trials']):
            ordered = pool.copy()
            random.Random(config['seed'] + trial).shuffle(ordered)
            for shot in config['shots']:
                support = ordered[:shot]
                stream.write(json.dumps({'trial': trial, 'seed': config['seed'] + trial, 'shots_per_class': shot, 'class_id': 0, 'support_ids': [r['sample_id'] for r in support], 'support_boxes': shot, 'query_manifest': 'data/splits/bccd_test.csv', 'validation_manifest': 'data/splits/bccd_val.csv'}) + '\n')
    return rows, duplicate_records


def validate(rows):
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated sample IDs')
    patient_splits = defaultdict(set)
    for row in rows:
        if row['patient_id']:
            patient_splits[row['patient_id']].add(row['split'])
    if any(len(splits) > 1 for splits in patient_splits.values()):
        raise ValueError('Patient leakage across folds')
    for split in ['train', 'val', 'test', 'external_test']:
        labels = {r['label'] for r in rows if r['split'] == split and r['status'] == 'retained'}
        if labels != {0, 1}:
            raise ValueError(f'Split {split} missing a class')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update-lock', action='store_true', help='Explicitly accept changed outputs after a documented protocol version update')
    parser.add_argument('--reuse-verified-index', action='store_true', help='Reuse previously decoded/hash-verified image rows when raw files have not changed')
    args = parser.parse_args()
    lock_path = ROOT / 'data/provenance/manifests.lock.json'
    previous_lock = json.loads(lock_path.read_text(encoding='utf-8')) if lock_path.exists() else None
    config = json.loads((ROOT / 'configs/data_protocol.json').read_text(encoding='utf-8'))
    if args.reuse_verified_index:
        with (ROOT / 'data/manifests/classification.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ['label', 'width', 'height']:
                row[key] = int(row[key])
            row['status'] = 'retained'
        excluded = json.loads((ROOT / 'data/manifests/summary.json').read_text())['excluded_partitions']
    else:
        rows, excluded = classification_rows(config)
    print(f'Indexed {len(rows)} classification images', flush=True)
    duplicates = audit_duplicates(rows, config['near_duplicate_hamming_distance'])
    validate(rows)
    write_csv(ROOT / 'data/manifests/classification.csv', rows)
    for dataset, split in [('cnmc', 'train'), ('cnmc', 'val'), ('cnmc', 'test'), ('aml', 'external_test')]:
        selected = [r for r in rows if r['dataset'] == dataset and r['split'] == split and r['status'] == 'retained']
        write_csv(ROOT / f'data/splits/{dataset}_{split}.csv', selected)
    build_episodes(rows, config)
    detection, detection_duplicates = build_detection(config)
    write_csv(ROOT / 'data/manifests/duplicate_audit.csv', duplicates + detection_duplicates, ['a', 'b', 'reason', 'distance'])
    counts = Counter((r['dataset'], r['split'], str(r['label']), r['status']) for r in rows + detection)
    summary = {'protocol_version': config['protocol_version'], 'counts': [{'dataset': k[0], 'split': k[1], 'label': k[2], 'status': k[3], 'images': v} for k, v in sorted(counts.items())], 'patients_by_cnmc_split': {s: len({r['patient_id'] for r in rows if r['dataset'] == 'cnmc' and r['split'] == s}) for s in ['train', 'val', 'test']}, 'excluded_partitions': dict(excluded), 'duplicate_pairs': len(duplicates) + len(detection_duplicates), 'episodes': len(config['shots']) * config['trials']}
    (ROOT / 'data/manifests/summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report = ['# Kiểm kê dữ liệu thực tế', '', 'Tạo bằng `scripts/prepare_data.py`; chỉ các hàng `retained` được dùng.', '', '| Dataset | Split | Label | Status | Images |', '|---|---|---|---|---:|']
    for item in summary['counts']:
        report.append('| ' + ' | '.join(str(item[k]) for k in ['dataset', 'split', 'label', 'status', 'images']) + ' |')
    report += ['', 'C-NMC patient counts trước audit: ' + json.dumps(summary['patients_by_cnmc_split']), '', f"Số trial/K: {config['trials']} × {len(config['shots'])}; 1.200 cấu hình classification và 1.200 detection.", '', 'AML không có mã bệnh nhân trong tên file; BCCD chỉ có split theo ảnh.', 'AML là kiểm tra đồng thời khác nguồn và khác loại bệnh (B-ALL → AML), không phải source-only shift.', 'Không có kết quả huấn luyện mô hình trong báo cáo này.', '']
    (ROOT / 'data/DATA_SUMMARY.md').write_text('\n'.join(report), encoding='utf-8')
    environment = {'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in ['requests', 'Pillow', 'numpy', 'scikit-learn']}}
    (ROOT / 'data/provenance/preparation_environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    # Hash all tracked outputs to permit exact reproduction checks.
    lock = {}
    for folder in ['manifests', 'splits', 'episodes']:
        for path in sorted((ROOT / 'data' / folder).glob('*')):
            if path.is_file():
                lock[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['configs/data_protocol.json', 'docs/DATASET_CATALOG.json', 'data/DATA_SUMMARY.md']:
        lock[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if previous_lock is not None and lock != previous_lock and not args.update_lock:
        changed = sorted(k for k in set(lock) | set(previous_lock) if lock.get(k) != previous_lock.get(k))
        raise ValueError(f'Frozen outputs changed: {changed}. Existing lock preserved. Investigate source/config changes; update protocol and use --update-lock only intentionally.')
    lock_path.write_text(json.dumps(lock, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
