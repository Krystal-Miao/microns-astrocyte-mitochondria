"""Small raw 2D U-Net pilot; target-cell masks are incomplete semantic labels."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn
from torch.nn import functional as F


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.down = nn.ModuleList([block(1, width), block(width, width*2),
                                  block(width*2, width*4)])
        self.bottom = block(width*4, width*8)
        self.up = nn.ModuleList([block(width*12, width*4),
                                block(width*6, width*2), block(width*3, width)])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for layer, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = layer(torch.cat([x, skip], dim=1))
        return self.head(x)


def loss_fn(logits, target):
    p = logits.sigmoid()
    dice = (2*(p*target).sum((1,2,3))+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1)
    return F.binary_cross_entropy_with_logits(logits, target) + (1-dice).mean()


def metrics(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'dice': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0,
            'iou': tp/(tp+fp+fn) if tp+fp+fn else 1.0}


def make_split(data, fold):
    """Hold out one complete volume; reserve separated Z blocks for validation."""
    test_name = f'sample_{fold:02d}'
    train = [(n,z) for n,(_,_,zs) in data.items() if n != test_name for z in zs if z < 96]
    val = [(n,z) for n,(_,_,zs) in data.items() if n != test_name for z in zs if z >= 112]
    test = [(test_name,z) for z in data[test_name][2]]
    if not train or not val or not test:
        raise ValueError('Empty training, validation or test split')
    return train, val, test


def load_data(root):
    data, audit = {}, {}
    boxes = {}
    for name in ['sample_01','sample_02','sample_03']:
        folder = root/name
        image = tifffile.imread(folder/'image.tif')
        mask = tifffile.imread(folder/'mask.tif')
        meta = json.loads((folder/'metadata.json').read_text())
        if image.shape != mask.shape or image.shape != (128,512,512):
            raise ValueError(f'{name}: expected matching (128,512,512) arrays')
        if image.dtype != np.uint8 or not set(np.unique(mask)).issubset({0,1}):
            raise ValueError(f'{name}: expected uint8 image and binary mask')
        if meta['saved_axis_order'] != 'ZYX' or meta['em_resolution_nm'] != meta['seg_resolution_nm']:
            raise ValueError(f'{name}: incompatible axes or spacing')
        valid = np.flatnonzero(np.ptp(image, axis=(1,2)) > 0).tolist()
        data[name] = (image, mask, valid)
        boxes[name] = meta['voxel_bbox']
        audit[name] = {'shape_zyx': list(image.shape), 'excluded_constant_z': sorted(set(range(128))-set(valid)),
                       'foreground_fraction': float(mask.mean()), 'root_id': meta['root_id'],
                       'sha256': {f: digest(folder/f) for f in ['image.tif','mask.tif','metadata.json']}}
    names = list(data)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            overlap = all(max(boxes[a]['min'][k],boxes[b]['min'][k]) <
                          min(boxes[a]['max'][k],boxes[b]['max'][k]) for k in range(3))
            if overlap:
                raise ValueError(f'Overlapping volumes: {a}, {b}')
    return data, audit


def batch(data, records, rng, size, count, augment=True, trace=None):
    images, masks = [], []
    for _ in range(count):
        name,z = records[int(rng.integers(len(records)))]
        a,m,_ = data[name]
        y,x = rng.integers(0,513-size,size=2)
        a,m = a[z,y:y+size,x:x+size],m[z,y:y+size,x:x+size]
        k, flip = 0, False
        if augment:
            k = int(rng.integers(4))
            a,m = np.rot90(a,k),np.rot90(m,k)
            flip = bool(rng.random()<0.5)
            if flip:
                a,m = a[:,::-1],m[:,::-1]
        if trace is not None:
            trace.append(dict(sample=name, z=int(z), x=int(x), y=int(y), size=size,
                              rotation_k=k, flip_horizontal=flip, foreground_fraction=float(m.mean())))
        images.append(a.copy())
        masks.append(m.copy())
    return (torch.from_numpy(np.stack(images)[:,None]).float()/255,
            torch.from_numpy(np.stack(masks)[:,None]).float())


@torch.no_grad()
def evaluate(model, data, records, device, out=None):
    model.eval()
    tp=fp=fn=0
    total_pixels = 0
    losses = []
    if not records:
        raise ValueError("Evaluation requires at least one slice")
    predictions={}
    rows=[]
    for name,z in records:
        a,m,_=data[name]
        x=torch.from_numpy(a[z].copy())[None,None].float().to(device)/255
        logits = model(x)
        y = torch.from_numpy(m[z].copy())[None,None].float().to(device)
        losses.append(float(loss_fn(logits, y).cpu()))
        pred=(logits.sigmoid()[0,0].cpu().numpy()>=0.5)
        total_pixels += pred.size
        target=m[z].astype(bool)
        t=int((pred & target).sum()); f=int((pred & ~target).sum()); n=int((~pred & target).sum())
        tp+=t;fp+=f;fn+=n
        rows.append({'sample':name,'z':z,**metrics(t,f,n)})
        if out:
            predictions.setdefault(name,[]).append(pred.astype(np.uint8))
    if out:
        for name,planes in predictions.items():
            tifffile.imwrite(out/f'{name}_prediction_valid_slices.tif',np.stack(planes),metadata={'axes':'ZYX'})
            (out/f'{name}_prediction_z.json').write_text(json.dumps(
                {'source_z_per_output_plane': [z for n,z in records if n == name]}, indent=2))
        with (out/'test_slice_metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {**metrics(tp,fp,fn), "loss": float(np.mean(losses)),
            "predicted_foreground_fraction": (tp+fp)/total_pixels,
            "labeled_foreground_fraction": (tp+fn)/total_pixels,
            "evaluated_slices": len(records)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('samples/astro_864691136057675736_run1'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--width',type=int,default=16)
    p.add_argument('--patch',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--fold',choices=['all','1','2','3'],default='all')
    p.add_argument('--smoke',action='store_true',help='Overfit one training patch; never score test data.')
    p.add_argument('--audit-only', action='store_true', help='Save data audit and splits without training.')
    args=p.parse_args()
    if args.steps<1 or not 8<=args.patch<=512 or args.width<1 or args.batch_size<1:
        p.error('Invalid steps, patch, width or batch size')
    if args.smoke and args.fold == '1':
        p.error('Smoke uses sample_01; choose fold 2 or 3, or omit --fold')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=args.device
    if device=='auto': device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    data,audit=load_data(args.data)
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2))
    config={**vars(args),'data':str(args.data),'out':str(args.out),'device':device,'torch_version':torch.__version__,
            'numpy_version':np.__version__,'script_sha256':digest(Path(__file__)),
            'loss':'BCEWithLogits + soft Dice, equal weights','normalization':'uint8 / 255',
            'test_checkpoint':'last.pt', 'checkpoint_policy':'fixed final step; best retained for reference only',
            'training_monitor':'8 evenly spaced valid training slices per training volume; no augmentation',
            'evaluation_loss':'mean of per-slice BCE + soft Dice losses',
            'threshold':0.5,'learning_rate':0.001,'augmentation':'XY rotations by 90 degrees and reflection',
            'label_limitation':'One target astrocyte only; other astrocytes may be labeled background. No exhaustive semantic-label audit.',
            'validation':'z=112..127 in training volumes; z=96..111 unused guard band; train z=0..95',
            'scope':'Single-cell pilot, not cross-cell generalization; fixed-step preliminary budget, not convergence claim'}
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    if args.audit_only:
        for fold in range(1,4):
            train,val,test = make_split(data,fold)
            (args.out/f'split_{fold}.json').write_text(json.dumps(
                {'train':train,'validation':val,'test':test},indent=2))
        print('Audit and splits saved:',args.out,flush=True)
        return
    if args.smoke:
        model=UNet(args.width).to(device);opt=torch.optim.Adam(model.parameters(),lr=0.001)
        a,m,_=data['sample_01']
        # Development diagnostic only; sample_01 is held out in fold 1 later.
        x=torch.from_numpy(a[47,256:512,128:384].copy())[None,None].float().to(device)/255
        y=torch.from_numpy(m[47,256:512,128:384].copy())[None,None].float().to(device)
        losses=[]
        for step in range(args.steps):
            model.train();opt.zero_grad();loss=loss_fn(model(x),y);loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
            if step%20==0: print('smoke',step,losses[-1],flush=True)
        result={'initial_loss':losses[0],'final_loss':losses[-1],'finite':bool(np.isfinite(losses).all()),'losses':losses}
        (args.out/'smoke.json').write_text(json.dumps(result,indent=2))
        assert result['finite'] and losses[-1]<losses[0]*0.8, result
        print('Smoke passed',result['initial_loss'],result['final_loss'],flush=True)
        return
    results=[]
    folds=range(1,4) if args.fold=='all' else [int(args.fold)]
    for fold in folds:
        torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
        test_name=f'sample_{fold:02d}'
        train,val,test=make_split(data,fold)
        out=args.out/f'fold_{fold}';out.mkdir()
        (out/'split.json').write_text(json.dumps({'train':train,'validation':val,'test':test},indent=2))
        monitor=[]
        for name in sorted({n for n,z in train}):
            records=[r for r in train if r[0]==name]
            monitor.extend(records[i] for i in np.linspace(0,len(records)-1,min(8,len(records)),dtype=int))
        (out/'training_monitor.json').write_text(json.dumps(monitor,indent=2))
        model=UNet(args.width).to(device);opt=torch.optim.Adam(model.parameters(),lr=0.001)
        best=-1;history=[];start=time.time()
        for step in range(1,args.steps+1):
            model.train();x,y=batch(data,train,rng,args.patch,args.batch_size)
            x,y=x.to(device),y.to(device);opt.zero_grad();loss=loss_fn(model(x),y)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward();opt.step()
            history.append({'step':step,'loss':float(loss.detach().cpu())})
            if step%25==0: print(f'fold {fold} step {step}/{args.steps} loss {history[-1]["loss"]:.4f} elapsed {time.time()-start:.0f}s',flush=True)
            if step%100==0 or step==args.steps:
                score=evaluate(model,data,val,device)
                train_score=evaluate(model,data,monitor,device)
                history[-1]['validation_dice']=score['dice']
                history[-1]['validation']=score
                history[-1]['training_monitor']=train_score
                print(f"fold {fold} step {step} train_monitor Dice={train_score['dice']:.4f} IoU={train_score['iou']:.4f} loss={train_score['loss']:.4f} foreground={train_score['predicted_foreground_fraction']:.2%} label={train_score['labeled_foreground_fraction']:.2%} | val Dice={score['dice']:.4f} IoU={score['iou']:.4f} loss={score['loss']:.4f} foreground={score['predicted_foreground_fraction']:.2%} label={score['labeled_foreground_fraction']:.2%}",flush=True)
                if score['dice']>best:
                    best=score['dice']
                    torch.save({'model':model.state_dict(),'width':args.width,'step':step,'validation':score},out/'best.pt')
                (out/'history.json').write_text(json.dumps(history,indent=2))
        torch.save({'model':model.state_dict(),'width':args.width,'step':args.steps,'validation':score},out/'last.pt')
        saved=torch.load(out/'last.pt',map_location=device,weights_only=True);model.load_state_dict(saved['model'])
        score=evaluate(model,data,test,device,out)
        result={'fold':fold,'test_sample':test_name,'seed':args.seed,'selected_step':saved['step'],
                'test_checkpoint':'last.pt', 'best_validation_dice':best,
                'validation_dice':saved['validation']['dice'],'elapsed_seconds':time.time()-start,**score}
        (out/'result.json').write_text(json.dumps(result,indent=2));results.append(result)
        print(result,flush=True)
    (args.out/'results.json').write_text(json.dumps(results,indent=2))
    for n,entry in audit.items():
        for f,h in entry['sha256'].items(): assert digest(args.data/n/f)==h,'Source changed'


if __name__=='__main__': main()
