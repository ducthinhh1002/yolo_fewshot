"""Validation-only matched few-shot adaptation; frozen token-prefix caching."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from bloodcells.data import CellDataset, read_manifest, classification_tensor
from bloodcells.adaptation import LoRALinear, EpisodicProjection, proto_logits
from scripts.run_smoke import audit, gpu_budget, save_json, gpu_stats
from scripts.evaluate_predictions import metrics


def load_model(c):
    from transformers import AutoImageProcessor, AutoModel
    kw = dict(revision=c['dino_revision'],local_files_only=True)
    return AutoImageProcessor.from_pretrained(c['dino_model'],**kw), AutoModel.from_pretrained(c['dino_model'],**kw).cuda().eval()


def cache(c,out):
    from PIL import Image, ImageOps, ImageEnhance
    from torch.utils.data import DataLoader
    processor, model = load_model(c)
    capture = {}
    def hook(module,args): capture['tokens'] = args[0].detach()
    handle = model.encoder.layer[-1].register_forward_pre_hook(hook)
    metadata = {}
    for split in ['train','val']:
        path = ROOT/f'data/splits/cnmc_{split}.csv'
        rows = read_manifest(path)
        loader = DataLoader(CellDataset(rows=rows,mean=processor.image_mean,std=processor.image_std),batch_size=c['cache_batch'],num_workers=0,shuffle=False)
        mmap = None
        offset = 0
        max_error = 0.
        with torch.inference_mode():
            for batch in loader:
                cls = F.normalize(model(pixel_values=batch['pixel_values'].cuda()).last_hidden_state[:,0,:].float(),dim=-1)
                tokens = capture['tokens']
                if mmap is None:
                    mmap = np.lib.format.open_memmap(out/f'{split}_tokens.npy',mode='w+',dtype='float32',shape=(len(rows),*tokens.shape[1:]))
                mmap[offset:offset+len(tokens)] = tokens.cpu().numpy()
                if offset == 0:
                    reconstructed = F.normalize(model.layernorm(model.encoder.layer[-1](tokens))[:,0,:],dim=-1)
                    max_error = float((reconstructed-cls).abs().max())
                    if max_error > 1e-5: raise ValueError('Prefix cache parity failed')
                offset += len(tokens)
                if offset % (c['cache_batch']*25) == 0: print(f'Cache {split}: {offset}/{len(rows)}',flush=True)
        mmap.flush()
        metadata[split] = {'count':len(rows),'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'parity_max_abs':max_error,'sample_ids':[r['sample_id'] for r in rows]}
    handle.remove()
    # A second image view for support-only episodic query training. No extra labels.
    rows = read_manifest('data/splits/cnmc_train.csv')
    augmented = []
    with torch.inference_mode():
        for start in range(0,len(rows),c['cache_batch']):
            batch = []
            for row in rows[start:start+c['cache_batch']]:
                with Image.open(ROOT/row['path']) as image:
                    image = ImageEnhance.Brightness(ImageOps.mirror(image.convert('RGB'))).enhance(1.05)
                    batch.append(classification_tensor(image,mean=processor.image_mean,std=processor.image_std))
            cls = model(pixel_values=torch.stack(batch).cuda()).last_hidden_state[:,0,:]
            augmented.append(F.normalize(cls.float(),dim=-1).cpu().numpy())
    np.save(out/'train_augmented_features.npy',np.concatenate(augmented))
    save_json(out/'cache.json',metadata)
    print('Frozen prefix cache and augmented support views ready',flush=True)


def experiment(c,out):
    torch.manual_seed(c['seed'])
    np.random.seed(c['seed'])
    _, model = load_model(c)
    model.requires_grad_(False)
    block = copy.deepcopy(model.encoder.layer[-1])
    norm = copy.deepcopy(model.layernorm)
    block.attention.q_proj = LoRALinear(block.attention.q_proj,c['lora_rank'],c['lora_alpha'])
    block.attention.v_proj = LoRALinear(block.attention.v_proj,c['lora_rank'],c['lora_alpha'])
    trainable = [p for p in block.parameters() if p.requires_grad]
    initial = {name:p.detach().clone() for name,p in block.named_parameters() if p.requires_grad}
    token = {split:torch.from_numpy(np.load(out/f'{split}_tokens.npy')).cuda() for split in ['train','val']}
    base = {split:np.load(ROOT/c['baseline']/f'cnmc_{split}_features.npz') for split in ['train','val']}
    cache_info = json.loads((out/'cache.json').read_text())
    for split in ['train','val']:
        ids = base[split]['sample_ids'].tolist()
        if ids != cache_info[split]['sample_ids']: raise ValueError('Cached token IDs differ from features')
        digest = hashlib.sha256((ROOT/f'data/splits/cnmc_{split}.csv').read_bytes()).hexdigest()
        if cache_info[split]['manifest_sha256'] != digest: raise ValueError('Stale token cache')
    index = {str(s):i for i,s in enumerate(base['train']['sample_ids'])}
    raw = torch.from_numpy(base['train']['features']).cuda()
    aug = torch.from_numpy(np.load(out/'train_augmented_features.npy')).cuda()
    query = torch.from_numpy(base['val']['features']).cuda()
    labels = base['val']['labels']
    records, all_scores = [], {m:[] for m in ['baseline','head_only','protonet','lora']}
    episode_ids = []
    def encode(x): return F.normalize(norm(block(x))[:,0,:].float(),dim=-1)
    with (ROOT/'data/episodes/classification.jsonl').open() as stream: episodes = [json.loads(line) for line in stream if json.loads(line)['trial'] < c['trials']]
    for number,e in enumerate(episodes):
        k,trial = e['shots_per_class'], e['trial']
        torch.manual_seed(c['seed']+trial)
        ix = torch.tensor([index[s] for s in e['support_ids']],device='cuda')
        y = torch.tensor([0]*k+[1]*k,device='cuda')
        support = raw[ix]
        centers = F.normalize(torch.stack([support[y==v].mean(0) for v in [0,1]]),dim=-1)
        # Reproduce the original NumPy reduction exactly: tiny float differences
        # near the 0.5 decision boundary can otherwise flip individual cells.
        support_np = base['train']['features'][ix.cpu().numpy()]
        labels_np = np.asarray([0]*k+[1]*k)
        original_centers = np.stack([support_np[labels_np==v].mean(axis=0) for v in [0,1]])
        original_centers /= np.linalg.norm(original_centers,axis=1,keepdims=True)
        with torch.no_grad(): p_base = (query@torch.from_numpy(original_centers).cuda().T).softmax(-1)[:,1].cpu().numpy()
        # A classifier-only control separates gains of the trainable head from LoRA.
        head = torch.nn.Parameter(centers.detach().clone())
        opt = torch.optim.Adam([head],lr=c['head_lr'])
        for _ in range(c['lora_steps']):
            opt.zero_grad()
            loss = F.cross_entropy(c['logit_scale']*support@F.normalize(head,dim=-1).T,y)
            loss.backward(); opt.step()
        with torch.no_grad(): p_head = (c['logit_scale']*query@F.normalize(head,dim=-1).T).softmax(-1)[:,1].cpu().numpy()
        # Episodic learned metric: only cells inside this K-shot support can appear.
        torch.manual_seed(c['seed']+trial)
        projection = EpisodicProjection(hidden=c['projection_hidden']).cuda()
        opt = torch.optim.Adam(projection.parameters(),lr=c['protonet_lr'])
        for _ in range(c['protonet_steps']):
            sidx,qidx = [],[]
            for v in [0,1]:
                ordered = torch.randperm(k,device='cuda')+v*k
                cut = max(1,k//2)
                sidx.extend(ordered[:cut].tolist())
                qidx.extend((ordered[cut:] if k>1 else ordered).tolist())
            opt.zero_grad()
            logits = proto_logits(projection(support[sidx]),y[sidx],projection(aug[ix[qidx]]),c['logit_scale'])
            loss = F.cross_entropy(logits,y[qidx])
            loss.backward(); opt.step()
        with torch.no_grad(): p_proto = proto_logits(projection(support),y,projection(query),c['logit_scale']).softmax(-1)[:,1].cpu().numpy()
        # Each trial starts from the exact same frozen weights and fresh adapters/head.
        with torch.no_grad():
            for name,p in block.named_parameters():
                if p.requires_grad: p.copy_(initial[name])
        head = torch.nn.Parameter(centers.detach().clone())
        opt = torch.optim.Adam([{'params':trainable,'lr':c['lora_lr']},{'params':[head],'lr':c['head_lr']}])
        block.train()
        for _ in range(c['lora_steps']):
            opt.zero_grad()
            loss = F.cross_entropy(c['logit_scale']*encode(token['train'][ix])@F.normalize(head,dim=-1).T,y)
            loss.backward(); opt.step()
        block.eval()
        probabilities = []
        with torch.inference_mode():
            for start in range(0,len(query),c['query_batch']):
                probabilities.append((c['logit_scale']*encode(token['val'][start:start+c['query_batch']])@F.normalize(head,dim=-1).T).softmax(-1)[:,1].cpu().numpy())
        p_lora = np.concatenate(probabilities)
        for name,p in [('baseline',p_base),('head_only',p_head),('protonet',p_proto),('lora',p_lora)]:
            if not np.isfinite(p).all(): raise ValueError('Non-finite probabilities')
            records.append({'method':name,'trial':trial,'shots_per_class':k,**metrics(labels,p)})
            all_scores[name].append(p)
        episode_ids.append([trial,k])
        if trial == 0:
            torch.save({'adapters':{name:p.detach().cpu() for name,p in block.named_parameters() if p.requires_grad},'head':head.detach().cpu(),'support_ids':e['support_ids']},out/f'lora_trial0_{k}shot.pt')
            torch.save({'projection':projection.state_dict(),'support_ids':e['support_ids']},out/f'protonet_trial0_{k}shot.pt')
        if number % 4 == 3: print(f'Adaptation trial {trial+1}/{c["trials"]} complete',flush=True)
    with (out/'comparison_trials.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    np.savez_compressed(out/'validation_scores.npz',**{name:np.stack(p) for name,p in all_scores.items()},episodes=np.asarray(episode_ids),sample_ids=base['val']['sample_ids'],labels=labels)
    summary = {}
    for name in all_scores:
        summary[name] = {}
        for k in c['shots']:
            group = [r for r in records if r['method']==name and r['shots_per_class']==k]
            baseline = [r for r in records if r['method']=='baseline' and r['shots_per_class']==k]
            result = {metric:{'mean':float(np.mean([r[metric] for r in group])),'sd':float(np.std([r[metric] for r in group],ddof=1))} for metric in ['macro_f1','sensitivity','specificity','auroc','auprc']}
            delta = [a['macro_f1']-b['macro_f1'] for a,b in zip(group,baseline)]
            result['paired_macro_f1_delta_mean'] = float(np.mean(delta))
            result['paired_win_fraction'] = float(np.mean(np.asarray(delta)>0))
            summary[name][k] = result
    save_json(out/'comparison.json',{'summary':summary,'trials':c['trials'],'query':'cnmc_val','test_used':False,'lora_trainable_adapter_parameters':sum(p.numel() for p in trainable),'lora_head_parameters':768,'memory':gpu_stats(),'config':c})


def main():
    import setproctitle
    parser = argparse.ArgumentParser(); parser.add_argument('--stage',choices=['all','cache','experiment'],default='all'); args=parser.parse_args()
    c = json.loads((ROOT/'configs/adaptation.json').read_text())
    setproctitle.setproctitle(c['process_name'])
    gpu_budget(c)
    out=ROOT/c['output']; out.mkdir(parents=True,exist_ok=True)
    audit(out)
    for name,fn in [('cache',cache),('experiment',experiment)]:
        if args.stage in ['all',name]:
            begin=time.monotonic(); fn(c,out); print(f'{name}: {time.monotonic()-begin:.1f}s',flush=True)


if __name__ == '__main__': main()
