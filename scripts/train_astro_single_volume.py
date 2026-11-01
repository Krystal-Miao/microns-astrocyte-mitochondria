"""Within-sample development experiment; sample_02/03 are not loaded or scored."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from train_astro_unet import UNet, batch, digest, evaluate, loss_fn
from diagnose_astro_unet import save_overlay


def replay_batch(data, records):
    """Use saved crop coordinates without spatial augmentation."""
    images=[]; masks=[]; trace=[]
    for original in records:
        t=dict(original)
        a,m,_=data[t['sample']]
        z,y,x,size=t['z'],t['y'],t['x'],t['size']
        aa=a[z,y:y+size,x:x+size]; mm=m[z,y:y+size,x:x+size]
        if aa.shape!=(256,256) or float(mm.mean())!=t['foreground_fraction']:
            raise ValueError('Replay crop does not match original sampling trace')
        images.append(aa); masks.append(mm)
        t.update(rotation_k=0,flip_horizontal=False)
        trace.append(t)
    return (torch.from_numpy(np.stack(images)[:,None].copy()).float()/255,
            torch.from_numpy(np.stack(masks)[:,None].copy()).float(),trace)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('samples/astro_864691136057675736_run1/sample_01'))
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--replay-without-augmentation',type=Path,help='Original run directory; replay identical crop order without rotation/flip')
    args=p.parse_args()
    if args.steps<1: p.error('steps must be positive')
    torch.set_num_threads(4);torch.manual_seed(42)
    rng=np.random.default_rng(42)
    hashes={n:digest(args.data/n) for n in ['image.tif','mask.tif','metadata.json']}
    a=tifffile.imread(args.data/'image.tif');m=tifffile.imread(args.data/'mask.tif')
    meta=json.loads((args.data/'metadata.json').read_text())
    if a.shape!=(128,512,512) or m.shape!=a.shape or a.dtype!=np.uint8 or not set(np.unique(m)) <= {0,1}:
        raise ValueError('Invalid image/mask shape, dtype or labels')
    if meta['saved_axis_order']!='ZYX' or meta['em_resolution_nm']!=meta['seg_resolution_nm']:
        raise ValueError('Incompatible axes or spacing')
    zs=np.flatnonzero(np.ptp(a,axis=(1,2))>0).tolist()
    data={'sample_01':(a,m,zs)}
    train=[('sample_01',z) for z in zs if z<96]
    val=[('sample_01',z) for z in zs if z>=112]
    monitor=[train[i] for i in np.linspace(0,len(train)-1,8,dtype=int)]

    replay=None
    if args.replay_without_augmentation:
        previous=args.replay_without_augmentation
        old=json.loads((previous/'config.json').read_text())
        for key,value in dict(seed=42,learning_rate=.0003,steps=args.steps,device=args.device,
                              width=16,batch_size=2,patch=256,source_sha256=hashes,
                              torch_version=torch.__version__,
                              model_script_sha256=digest(Path(__file__).with_name('train_astro_unet.py'))).items():
            if old[key]!=value: raise ValueError(f'Control configuration differs: {key}')
        replay=[json.loads(line) for line in (previous/'sampling.jsonl').read_text().splitlines()]
        if len(replay)!=args.steps*2: raise ValueError('Incorrect replay length')
        for i,t in enumerate(replay):
            if (t['sample'],t['z']) not in train or t['step']!=i//2+1 or t['size']!=256:
                raise ValueError('Invalid replay training record')
            if not (0<=t['x']<=256 and 0<=t['y']<=256): raise ValueError('Invalid crop')
    args.out.mkdir(parents=True,exist_ok=False)
    config=dict(seed=42,learning_rate=.0003,steps=args.steps,device=args.device,width=16,batch_size=2,patch=256,
                normalization='uint8 / 255',loss='BCEWithLogits + soft Dice',threshold=.5,
                sampling='uniform valid training slice then uniform XY top-left; with replacement',
                augmentation='rot90 then horizontal flip, paired image/mask',source_sha256=hashes,
                torch_version=torch.__version__,script_sha256=digest(Path(__file__)),
                model_script_sha256=digest(Path(__file__).with_name('train_astro_unet.py')),
                scope='Within sample_01 development only; no independent biological replication or held-out test',
                source=str(args.data),checkpoint_policy='final step reported; best validation retained separately')
    if replay is not None:
        config.update(augmentation='none',replay_source=str(args.replay_without_augmentation),
                      replay_sha256=digest(args.replay_without_augmentation/'sampling.jsonl'))
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    (args.out/'split.json').write_text(json.dumps(dict(train=train,validation=val,training_monitor=monitor,
        guard_z=list(range(96,112)),excluded_constant_z=sorted(set(range(128))-set(zs))),indent=2))
    model=UNet(16).to(args.device);opt=torch.optim.Adam(model.parameters(),lr=.0003)
    start=time.time();history=[];sampling=[];best=-1
    preview_z=[monitor[0][1],47,val[0][1],val[-1][1]]
    for z in preview_z: save_overlay(args.out/f'z{z:03d}_reference.png',a[z],m[z])
    with (args.out/'sampling.jsonl').open('w') as trace_file:
        for step in range(1,args.steps+1):
            trace=[]
            if replay is None:
                x,y=batch(data,train,rng,256,2,trace=trace)
            else:
                x,y,trace=replay_batch(data,replay[2*(step-1):2*step])
            for item in trace:
                item['step']=step
                trace_file.write(json.dumps(item)+'\n')
                sampling.append(item['foreground_fraction'])
            model.train();opt.zero_grad()
            loss=loss_fn(model(x.to(args.device)),y.to(args.device))
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward();opt.step()
            row=dict(step=step,batch_loss=float(loss.detach().cpu()))
            history.append(row)
            if step%25==0:
                print(f'step {step}/{args.steps} batch_loss={row["batch_loss"]:.4f} elapsed={time.time()-start:.0f}s',flush=True)
            if step%100==0 or step==args.steps:
                trace_file.flush()
                row['training_monitor']=evaluate(model,data,monitor,args.device)
                row['validation']=evaluate(model,data,val,args.device)
                row['elapsed_seconds']=time.time()-start
                for label in ['training_monitor','validation']:
                    r=row[label]
                    print(f"{label}: loss={r['loss']:.4f} Dice={r['dice']:.4f} IoU={r['iou']:.4f} foreground={r['predicted_foreground_fraction']:.2%}",flush=True)
                snap=args.out/f'step_{step:04d}';snap.mkdir()
                with torch.no_grad():
                    for z in preview_z:
                        x=torch.from_numpy(a[z].copy())[None,None].float().to(args.device)/255
                        prob=model(x).sigmoid()[0,0].cpu().numpy()
                        save_overlay(snap/f'z{z:03d}_prediction.png',a[z],prob>=.5)
                if row['validation']['dice']>best:
                    best=row['validation']['dice']
                    torch.save(dict(model=model.state_dict(),width=16,step=step,validation=row['validation']),args.out/'best.pt')
                (args.out/'history.json').write_text(json.dumps(history,indent=2))
                rows=[dict(step=r['step'],**{f'{label}_{k}':v for label in ['training_monitor','validation'] for k,v in r[label].items()}) for r in history if 'validation' in r]
                with (args.out/'metrics.csv').open('w') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    torch.save(dict(model=model.state_dict(),width=16,step=args.steps,validation=row['validation']),args.out/'last.pt')
    result=dict(step=args.steps,training_monitor=row['training_monitor'],validation=row['validation'],
                elapsed_seconds=time.time()-start,sampling=dict(count=len(sampling),
                background_only_fraction=float(np.mean(np.array(sampling)==0)),
                mean_foreground_fraction=float(np.mean(sampling))),test_evaluated=False)
    for name,h in hashes.items():
        if digest(args.data/name)!=h: raise ValueError('Source changed')
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
