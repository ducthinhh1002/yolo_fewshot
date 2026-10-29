"""Evaluate a complete paired classification experiment, failing on missing queries.

Input CSV: dataset,trial,shots_per_class,sample_id,prob_abnormal
One invocation per method/configuration; predictions must cover all 300 trials.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = p >= 0.5
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    if tp + fn == 0 or tn + fp == 0:
        raise ValueError('Both ground-truth classes are required')
    f1_pos = 2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else 0.0
    f1_neg = 2*tn / (2*tn + fp + fn) if 2*tn + fp + fn else 0.0
    sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)
    return {'macro_f1': (f1_pos+f1_neg)/2, 'positive_f1': f1_pos, 'sensitivity': sensitivity, 'specificity': specificity, 'balanced_accuracy': (sensitivity+specificity)/2, 'auroc': float(roc_auc_score(y, p)), 'auprc': float(average_precision_score(y, p)), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/data_protocol.json').read_text())
    manifests = {}
    for dataset, filename in [('cnmc', 'cnmc_test.csv'), ('aml', 'aml_external_test.csv')]:
        with (ROOT / 'data/splits' / filename).open(encoding='utf-8', newline='') as stream:
            manifests[dataset] = {r['sample_id']: r for r in csv.DictReader(stream)}
    groups = defaultdict(dict)
    with args.predictions.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            dataset, trial, shot = row['dataset'], int(row['trial']), int(row['shots_per_class'])
            if dataset not in manifests or not 0 <= trial < config['trials'] or shot not in config['shots']:
                raise ValueError('Unknown dataset, trial or shot')
            sample = row['sample_id']
            p = float(row['prob_abnormal'])
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Probability must be finite and in [0,1]')
            group = groups[(dataset, trial, shot)]
            if sample in group:
                raise ValueError('Duplicate prediction')
            group[sample] = p
    expected = {(d, t, k) for d in manifests for t in range(config['trials']) for k in config['shots']}
    if set(groups) != expected:
        raise ValueError(f'Incomplete experiment: got {len(groups)} of {len(expected)} groups')
    per_trial, combined = [], defaultdict(list)
    bootstrap = []
    for (dataset, trial, shot), predictions in sorted(groups.items()):
        if set(predictions) != set(manifests[dataset]):
            raise ValueError(f'Queries do not match frozen manifest for {dataset}/{trial}/{shot}')
        ids = sorted(predictions)
        y = [int(manifests[dataset][i]['label']) for i in ids]
        p = [predictions[i] for i in ids]
        result = metrics(y, p)
        per_trial.append({'dataset': dataset, 'trial': trial, 'shots_per_class': shot, **result})
        combined[(dataset, shot)].append(result)
        # Fixed-support conditional CI: trial 0 only, stratified patient bootstrap.
        if dataset == 'cnmc' and trial == 0:
            clusters = defaultdict(list)
            for index, sample in enumerate(ids):
                clusters[manifests[dataset][sample]['patient_id']].append(index)
            by_label = {c: sorted([patient for patient, indexes in clusters.items() if y[indexes[0]] == c]) for c in [0, 1]}
            rng = np.random.default_rng(config['seed'])
            scores = []
            for _ in range(config['evaluation']['bootstrap_replicates']):
                sampled = [patient for c in [0, 1] for patient in rng.choice(by_label[c], size=len(by_label[c]), replace=True)]
                indexes = [i for patient in sampled for i in clusters[patient]]
                # Bootstrap macro-F1 only; no assumptions of independent cells.
                yy, pp = np.asarray(y)[indexes], np.asarray(p)[indexes]
                pred = pp >= 0.5
                tp, tn = ((yy == 1) & pred).sum(), ((yy == 0) & ~pred).sum()
                errors = (pred != yy).sum()
                scores.append(float((2*tp/(2*tp+errors) + 2*tn/(2*tn+errors))/2))
            bootstrap.append({'dataset': dataset, 'shots_per_class': shot, 'trial': 0, 'metric': 'macro_f1', 'low_95': float(np.quantile(scores, .025)), 'high_95': float(np.quantile(scores, .975)), 'scope': 'conditional_on_trial_0_support_stratified_patient_bootstrap'})
    summary = []
    for (dataset, shot), values in sorted(combined.items()):
        result = {'dataset': dataset, 'shots_per_class': shot, 'trials': len(values)}
        for metric in values[0]:
            vector = [v[metric] for v in values]
            result[metric] = {'mean': float(np.mean(vector)), 'sd': float(np.std(vector, ddof=1))}
        summary.append(result)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'per_trial.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_trial[0]))
        writer.writeheader()
        writer.writerows(per_trial)
    (args.output / 'summary.json').write_text(json.dumps({'summary': summary, 'conditional_patient_bootstrap': bootstrap, 'external_patient_ci': 'not_available_patient_ids_not_published'}, indent=2), encoding='utf-8')
    print(f'Validated and evaluated {len(groups)} prediction groups')


if __name__ == '__main__':
    main()
