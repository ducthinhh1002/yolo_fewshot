"""COCO box AP and fixed-threshold diagnostics on the tiny validation pilot."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_errors import match_boxes
from scripts.run_smoke import save_json


def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    p=ROOT/'runs/devit_smoke';data=json.loads((p/'result.json').read_text())
    query=data['records'][0]['query_ids']; lookup={s:i+1 for i,s in enumerate(query)}
    with (ROOT/'data/splits/bccd_val.csv').open() as stream: rows={r['sample_id']:r for r in csv.DictReader(stream)}
    with (ROOT/'data/manifests/bccd_boxes.csv').open() as stream: boxes=[r for r in csv.DictReader(stream) if r['sample_id'] in lookup]
    gt={'info':{},'images':[{'id':lookup[s],'width':int(rows[s]['width']),'height':int(rows[s]['height'])} for s in query],'categories':[{'id':1,'name':'WBC'}],'annotations':[]}
    expected={s:[] for s in query}
    for i,b in enumerate(boxes):
        x1,y1,x2,y2=[float(b[k]) for k in ['xmin','ymin','xmax','ymax']]
        gt['annotations'].append({'id':i+1,'image_id':lookup[b['sample_id']],'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0})
        expected[b['sample_id']].append([x1,y1,x2,y2])
    coco=COCO();coco.dataset=gt;coco.createIndex()
    output=[]
    for record in data['records']:
        detections=[];tp=fp=fn=0
        for image in record['results']:
            predictions=[]
            for box,score in zip(image['boxes_xyxy'],image['scores']):
                x1,y1,x2,y2=box
                detections.append({'image_id':lookup[image['sample_id']],'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'score':score})
                if score>=.25: predictions.append(box+[score])
            matched,false,missed=match_boxes(predictions,expected[image['sample_id']],.5)
            tp+=len(matched);fp+=len(false);fn+=len(missed)
        if detections:
            result=coco.loadRes(detections);e=COCOeval(coco,result,'bbox');e.params.imgIds=list(lookup.values());e.evaluate();e.accumulate();e.summarize()
            ap,ap50=float(e.stats[0]),float(e.stats[1])
        else: ap=ap50=0.
        output.append({'shots_per_class':record['shots_per_class'],'ap_50_95':ap,'ap50':ap50,'tp_conf025_iou05':tp,'fp_conf025_iou05':fp,'fn_conf025_iou05':fn,'raw_predictions':len(detections)})
    save_json(p/'evaluation.json',{'query_split':'validation_only','images':len(query),'ground_truth_wbc_boxes':len(boxes),'pilot_not_benchmark':True,'results':output})
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
