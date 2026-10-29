"""Analyze validation errors only; labels are source ground truth, not diagnoses."""
import csv
import json
from pathlib import Path
import sys
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from bloodcells.data import read_manifest
from scripts.run_smoke import gpu_budget,save_json


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def classification(c,out):
    scores=np.load(out/'validation_scores.npz')
    rows=read_manifest('data/splits/cnmc_val.csv')
    if scores['sample_ids'].tolist()!=[r['sample_id'] for r in rows]: raise ValueError('Validation ID mismatch')
    selected=scores['episodes'][:,1]==20
    y=scores['labels']
    findings={}
    all_rows=[]
    for method in ['baseline','head_only','protonet','lora']:
        probabilities=scores[method][selected]
        errors=(probabilities>=.5)!=(y[None,:]==1)
        frequency=errors.mean(axis=0)
        average=probabilities.mean(0)
        patients=defaultdict(list)
        for i,row in enumerate(rows):
            patients[row['patient_id']].append(i)
            all_rows.append({'method':method,'sample_id':row['sample_id'],'patient_id':row['patient_id'],'label':int(y[i]),'mean_abnormal_score':float(average[i]),'score_sd':float(probabilities[:,i].std(ddof=1)),'error_frequency':float(frequency[i]),'path':row['path']})
        per_patient=[{'patient_id':patient,'images':len(ix),'mean_cell_error_frequency':float(frequency[ix].mean())} for patient,ix in patients.items()]
        per_patient.sort(key=lambda r:r['mean_cell_error_frequency'],reverse=True)
        confusions=[]
        for p in probabilities:
            prediction=p>=.5
            confusions.append([int(((y==0)&~prediction).sum()),int(((y==0)&prediction).sum()),int(((y==1)&~prediction).sum()),int(((y==1)&prediction).sum())])
        findings[method]={'shots_per_class':20,'trials':int(selected.sum()),'average_confusion_tn_fp_fn_tp':np.mean(confusions,axis=0).tolist(),'cells_wrong_in_every_trial':int((frequency==1).sum()),'cells_error_frequency_at_least_80pct':int((frequency>=.8).sum()),'patient_error_rates':per_patient}
    write_csv(out/'classification_errors.csv',all_rows)
    # Fixed image-level strata, to describe associations rather than diagnose causes.
    baseline=[r for r in all_rows if r['method']=='baseline']
    properties=[]
    for row in rows:
        with Image.open(ROOT/row['path']) as image:
            array=np.asarray(image.convert('RGB'),dtype=float)
        foreground=array.max(axis=-1)>8
        properties.append((float(foreground.mean()),float(array[foreground].mean()/255) if foreground.any() else 0.))
    strata=[]
    for column,name in [(0,'foreground_occupancy'),(1,'foreground_brightness')]:
        x=np.asarray([p[column] for p in properties])
        edges=np.quantile(x,[0,.25,.5,.75,1])
        bucket=np.searchsorted(edges[1:-1],x,side='right')
        for label in [0,1]:
            for q in range(4):
                ix=np.where((y==label)&(bucket==q))[0]
                strata.append({'property':name,'label':label,'quartile':q+1,'images':len(ix),'mean_baseline_error_frequency':float(np.mean([baseline[i]['error_frequency'] for i in ix])) if len(ix) else None})
    findings['strata_associations_not_causal']=strata
    save_json(out/'classification_error_analysis.json',findings)
    classification_plot(out,baseline)


def classification_plot(out,baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    worst=[]
    for label in [0,1]:
        worst.extend(sorted([r for r in baseline if r['label']==label],key=lambda r:(-r['error_frequency'],-r['score_sd'],r['sample_id']))[:4])
    fig,axes=plt.subplots(2,4,figsize=(12,8.5))
    for ax,row in zip(axes.flat,worst):
        with Image.open(ROOT/row['path']) as image: ax.imshow(image.convert('RGB'))
        ax.set_title(f"Nhãn gốc: {row['label']} | Sai: {row['error_frequency']:.0%}\nĐiểm bất thường TB: {row['mean_abnormal_score']:.2f}",fontsize=9)
        ax.axis('off')
    fig.suptitle('Ví dụ baseline khó phân loại trên validation (20-shot)',fontsize=14)
    fig.subplots_adjust(left=.025,right=.975,bottom=.035,top=.86,hspace=.4,wspace=.08)
    fig.savefig(out/'classification_error_examples.png',dpi=150); plt.close(fig)


def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1])
    x2,y2=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


def match_boxes(predictions,ground_truth,threshold):
    used=set(); matched=[]; false=[]
    for i,p in sorted(enumerate(predictions),key=lambda p:-p[1][4]):
        choices=[(iou(p[:4],g),j) for j,g in enumerate(ground_truth) if j not in used]
        best=max(choices,default=(0,-1))
        if best[0]>=threshold:
            used.add(best[1]); matched.append((i,best[1],best[0]))
        else: false.append(i)
    return matched,false,[j for j in range(len(ground_truth)) if j not in used]


def detection(c,out):
    from ultralytics import YOLO
    gpu_budget(c)
    info=json.loads((ROOT/c['baseline']/'detector.json').read_text())
    model=YOLO(info['best_checkpoint'])
    rows=read_manifest('data/splits/bccd_val.csv')
    gt=defaultdict(list)
    with (ROOT/'data/manifests/bccd_boxes.csv').open() as stream:
        for row in csv.DictReader(stream): gt[row['sample_id']].append([float(row[k]) for k in ['xmin','ymin','xmax','ymax']])
    totals={str(t):{'tp':0,'fp':0,'fn':0} for t in [.5,.75]}
    cases=[]
    for row in rows:
        p=model.predict(str(ROOT/row['path']),conf=.25,imgsz=640,device=0,verbose=False)[0]
        predicted=[box+[confidence] for box,confidence in zip(p.boxes.xyxy.cpu().tolist(),p.boxes.conf.cpu().tolist())]
        counts={}
        for threshold in [.5,.75]:
            matched,false,missed=match_boxes(predicted,gt[row['sample_id']],threshold)
            counts[str(threshold)]={'tp':len(matched),'fp':len(false),'fn':len(missed)}
            for key in ['tp','fp','fn']: totals[str(threshold)][key]+=counts[str(threshold)][key]
        cases.append({'sample_id':row['sample_id'],'path':row['path'],'ground_truth':gt[row['sample_id']],'predictions':predicted,'counts':counts})
    for value in totals.values():
        value['precision']=value['tp']/max(1,value['tp']+value['fp'])
        value['recall']=value['tp']/max(1,value['tp']+value['fn'])
    save_json(out/'detection_error_analysis.json',{'confidence_threshold':.25,'matching':'confidence_sorted_greedy_one_to_one','validation_images':len(rows),'totals':totals,'cases':cases,'note':'Fixed-threshold error counts differ from Ultralytics best-F1 operating point; not mAP.'})
    detection_plot(out,cases)


def detection_plot(out,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    worst=sorted(cases,key=lambda r:-(r['counts']['0.75']['fp']+r['counts']['0.75']['fn']))[:4]
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for ax,row in zip(axes.flat,worst):
        with Image.open(ROOT/row['path']) as image: ax.imshow(image.convert('RGB'))
        for color,boxes in [('lime',row['ground_truth']),('red',row['predictions'])]:
            for b in boxes: ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,linewidth=1.5))
        ax.set_title(row['sample_id'],fontsize=10); ax.axis('off')
    fig.suptitle('Lỗi định vị validation: nhãn gốc xanh, dự đoán đỏ\nChọn các ảnh khó nhất theo IoU 0,75; confidence 0,25',fontsize=12)
    fig.subplots_adjust(left=.025,right=.975,bottom=.03,top=.85,hspace=.22,wspace=.08)
    fig.savefig(out/'detection_error_examples.png',dpi=150); plt.close(fig)


def main():
    import argparse
    import setproctitle
    parser=argparse.ArgumentParser(); parser.add_argument('--plots-only',action='store_true'); args=parser.parse_args()
    c=json.loads((ROOT/'configs/adaptation.json').read_text()); out=ROOT/c['output']
    setproctitle.setproctitle(c['process_name'])
    if args.plots_only:
        with (out/'classification_errors.csv').open() as stream: baseline=[r for r in csv.DictReader(stream) if r['method']=='baseline']
        for row in baseline:
            row['label']=int(row['label'])
            for key in ['error_frequency','score_sd','mean_abnormal_score']: row[key]=float(row[key])
        classification_plot(out,baseline)
        detection_plot(out,json.loads((out/'detection_error_analysis.json').read_text())['cases'])
    else:
        classification(c,out); detection(c,out)
    print('Validation error analysis complete',flush=True)


if __name__=='__main__': main()
