"""Run the actual pinned official DE-ViT detector with BCCD support prototypes."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor/devit'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(VENDOR))
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from scripts.run_smoke import gpu_budget,gpu_stats,save_json
from bloodcells.data import read_manifest


def main():
    import setproctitle
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from lib.categories import SEEN_CLS_DICT,ALL_CLS_DICT
    c=json.loads((ROOT/'configs/devit_smoke.json').read_text())
    setproctitle.setproctitle(c['process_name']); gpu_budget(c); torch.manual_seed(c['seed'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip()
    if commit!=c['upstream_commit']: raise ValueError('Wrong upstream revision')
    out=ROOT/c['output']; out.mkdir(parents=True,exist_ok=True)
    weights=ROOT/'models/devit'
    original=[torch.load(weights/name,map_location='cpu',weights_only=False) for name in ['fs_coco14_base_train.vits14.pkl','fs_coco_trainval_novel_10shot.vits14.pkl']]
    names=original[0]['label_names']+original[1]['label_names']+['WBC']
    prototypes=torch.cat([p['prototypes'] for p in original]+[original[0]['prototypes'][:1]],dim=0)
    prototype_file=out/'bccd_with_reference_prototypes.pth'
    torch.save({'prototypes':prototypes,'label_names':names},prototype_file)
    SEEN_CLS_DICT['bccd_devit']=['WBC']
    ALL_CLS_DICT['bccd_devit']=names
    cfg=get_cfg()
    cfg.merge_from_file(str(VENDOR/'configs/few-shot/vits_shot10.yaml'))
    cfg.MODEL.DEVICE='cuda'
    cfg.MODEL.WEIGHTS=str(ROOT/c['checkpoint'])
    cfg.DE.CLASS_PROTOTYPES=str(prototype_file)
    cfg.DE.BG_PROTOTYPES=str(weights/'background_prototypes.vits14.pth')
    cfg.DE.OFFLINE_RPN_CONFIG=str(VENDOR/'configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml')
    cfg.DE.OFFLINE_RPN_POST_NMS_TOPK_TEST=c['proposal_limit']
    cfg.DATASETS.TRAIN=('bccd_devit',)
    cfg.DATASETS.TEST=()
    cfg.TEST.DETECTIONS_PER_IMAGE=c['detections_per_image']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=c['score_threshold']
    model=build_model(cfg).eval()
    state=torch.load(ROOT/c['checkpoint'],map_location='cpu',weights_only=False)['model']
    # Replace class-dependent buffers with BCCD + auxiliary reference prototypes.
    state={k:v for k,v in state.items() if k not in ['train_class_weight','test_class_weight']}
    result=model.load_state_dict(state,strict=False)
    missing=[k for k in result.missing_keys if k not in ['train_class_weight','test_class_weight']]
    if missing or result.unexpected_keys: raise ValueError(f'Unloaded detector weights: {missing}, {result.unexpected_keys}')
    print('Official DE-ViT checkpoint loaded; no missing learned detector parameters',flush=True)
    with (ROOT/'data/manifests/bccd_boxes.csv').open() as stream:
        boxes={}
        for row in csv.DictReader(stream): boxes.setdefault(row['sample_id'],[]).append([float(row[k]) for k in ['xmin','ymin','xmax','ymax']])
    train={r['sample_id']:r for r in read_manifest('data/splits/bccd_train.csv')}
    query=read_manifest(f'data/splits/{c["query_split"]}.csv')[:c['query_images']]
    with (ROOT/'data/episodes/detection.jsonl').open() as stream: episodes=[json.loads(line) for line in stream if json.loads(line)['trial']==c['trial']]
    def image_tensor(row):
        with Image.open(ROOT/row['path']) as image: array=np.asarray(image.convert('RGB')).copy()
        h,w=array.shape[:2]
        scale=min(c['short_side']/min(h,w),c['max_side']/max(h,w))
        nh,nw=round(h*scale),round(w*scale)
        tensor=torch.from_numpy(array).permute(2,0,1).float().cuda()
        return F.interpolate(tensor[None],size=(nh,nw),mode='bilinear',align_corners=False)[0],(h,w)
    # Extract foreground patch means using only the single annotated support WBC.
    support_tokens={}
    largest=max(episodes,key=lambda e:e['shots_per_class'])
    with torch.inference_mode():
        for sample in largest['support_ids']:
            if sample not in train or len(boxes[sample])!=1: raise ValueError('Invalid annotated support budget')
            tensor,(h,w)=image_tensor(train[sample])
            mean=torch.tensor([123.675,116.28,103.53],device='cuda')[:,None,None]
            std=torch.tensor([58.395,57.12,57.375],device='cuda')[:,None,None]
            nh,nw=tensor.shape[1:]
            hh,ww=max(14,round(nh/14)*14),max(14,round(nw/14)*14)
            normalized=F.interpolate(((tensor-mean)/std)[None],size=(hh,ww),mode='bicubic',align_corners=False)
            patch=model.backbone(normalized)[model.vit_feat_name][0]
            x1,y1,x2,y2=boxes[sample][0]
            # Soft area mask at patch resolution, reflecting the author's bbox mode.
            mask=torch.zeros((1,1,h,w),device='cuda')
            mask[:,:,int(y1):int(y2),int(x1):int(x2)]=1
            mask=F.interpolate(mask,size=patch.shape[1:],mode='bilinear',align_corners=False)[0]
            token=(patch.float()*mask).flatten(1).sum(1)/mask.sum().clamp_min(1e-6)
            if not torch.isfinite(token).all(): raise ValueError('Invalid support prototype')
            support_tokens[sample]=token
    records=[]
    for episode in sorted(episodes,key=lambda e:e['shots_per_class']):
        k=episode['shots_per_class']
        if k not in c['shots']: continue
        token=F.normalize(torch.stack([support_tokens[s] for s in episode['support_ids']]).mean(0),dim=0)
        model.test_class_weight[-1].copy_(token)
        results=[]; begin=time.monotonic()
        with torch.inference_mode():
            for row in query:
                tensor,(h,w)=image_tensor(row)
                prediction=model([{'image':tensor,'height':h,'width':w}])[0]['instances'].to('cpu')
                wbc=prediction.pred_classes==names.index('WBC')
                xyxy=prediction.pred_boxes.tensor[wbc]; scores=prediction.scores[wbc]
                if not torch.isfinite(xyxy).all() or not torch.isfinite(scores).all():raise ValueError('Non-finite detector predictions')
                results.append({'sample_id':row['sample_id'],'boxes_xyxy':xyxy.tolist(),'scores':scores.tolist(),'all_classes_detections':len(prediction)})
        records.append({'shots_per_class':k,'support_ids':episode['support_ids'],'annotated_wbc_boxes':k,'prototype':token.cpu().tolist(),'query_ids':[r['sample_id'] for r in query],'results':results,'seconds':time.monotonic()-begin})
        print(f'DE-ViT {k}-shot: {len(results)} validation images, {sum(len(r["scores"]) for r in results)} WBC predictions',flush=True)
    save_json(out/'result.json',{'status':'passed','upstream_commit':commit,'checkpoint_sha256':hashlib.sha256((ROOT/c['checkpoint']).read_bytes()).hexdigest(),'missing_learned_parameters':missing,'config':c,'records':records,'memory':gpu_stats(),'test_used':False,'auxiliary_reference_classes':80,'reference_classes_output_filtered':True,'environment':{p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','timm','Pillow']}})
    print('Actual DE-ViT smoke test passed',flush=True)


if __name__=='__main__':main()
