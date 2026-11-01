"""Memorize fixed training crops, with optional paired augmentation; no held-out scoring."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import tifffile
import torch
from train_astro_unet import UNet, loss_fn, digest, metrics
from diagnose_astro_unet import measure, save_overlay


def augment_pair(x, y, k, flip):
    x, y = torch.rot90(x, k, (-2,-1)), torch.rot90(y, k, (-2,-1))
    if flip:
        x, y = x.flip(-1), y.flip(-1)
    return x, y


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--augment',action='store_true',help='Paired random rot90 and horizontal flip')
    p.add_argument('--patch-count',type=int,default=4)
    p.add_argument('--extend-if-needed',type=int,help='Continue the same optimizer to this step if any patch Dice < .95')
    args=p.parse_args()
    if args.patch_count<2 or args.patch_count%2:p.error('patch-count must be positive and even')
    if args.extend_if_needed and args.extend_if_needed<=args.steps:p.error('extension must exceed steps')
    if args.steps<1:p.error('steps must be positive')
    run=Path('samples/unet_s01_random256_lr0003_01')
    cfg=json.loads((run/'config.json').read_text());source=Path(cfg['source'])
    trace=[json.loads(s) for s in (run/'sampling.jsonl').read_text().splitlines()]
    # Evenly spaced draw indices, fixed before training; allow repeated Z at different XY.
    indices=np.linspace(0,len(trace)-1,args.patch_count,dtype=int).tolist()
    chosen=[trace[i] for i in indices]
    assert len({(r['z'],r['x'],r['y']) for r in chosen})==args.patch_count
    train=set(map(tuple,json.loads((run/'split.json').read_text())['train']))
    assert all((r['sample'],r['z']) in train for r in chosen)
    for f,h in cfg['source_sha256'].items():assert digest(source/f)==h
    a=tifffile.imread(source/'image.tif');m=tifffile.imread(source/'mask.tif')
    images=[];masks=[]
    for r in chosen:
        z,y,x,n=r['z'],r['y'],r['x'],r['size']
        images.append(a[z,y:y+n,x:x+n].copy());masks.append(m[z,y:y+n,x:x+n].copy())
    assert all(set(np.unique(y))=={0,1} for y in masks)
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    model=UNet(16);opt=torch.optim.Adam(model.parameters(),lr=.0003)
    x=torch.from_numpy(np.stack(images)[:,None]).float()/255
    y=torch.from_numpy(np.stack(masks)[:,None]).float()
    config=dict(seed=42,learning_rate=.0003,steps=args.steps,width=16,batch_size=2,
        device='cpu',augmentation=args.augment,threshold=.5,normalization='uint8 / 255',loss='BCE + soft Dice',
        selection_indices=indices,selection=chosen,selection_policy=f'{args.patch_count} evenly spaced training draw indices; native orientation',
        patch_count=args.patch_count,extend_if_needed=args.extend_if_needed,
        schedule='cycle consecutive pairs in selected order; exposures = steps * 2 / patch_count at complete cycles',
        source_sha256=cfg['source_sha256'],script_sha256=digest(Path(__file__)),
        model_script_sha256=digest(Path(__file__).with_name('train_astro_unet.py')),
        scope='Training memorization diagnostic only; differs from random-crop run in diversity, augmentation and sampling schedule')
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    for i in range(args.patch_count):
        save_overlay(args.out/f'patch_{i}_reference.png',images[i],masks[i])
        tifffile.imwrite(args.out/f'patch_{i}_image.tif',images[i])
        tifffile.imwrite(args.out/f'patch_{i}_mask.tif',masks[i])
    history=[];start=time.time()
    rng=np.random.default_rng(42)
    augmentation_trace=[]
    for step in range((args.extend_if_needed or args.steps)+1):
        if step:
            first=((step-1)%(args.patch_count//2))*2
            bx, by = x[first:first+2], y[first:first+2]
            if args.augment:
                inputs, targets = [], []
                for i in range(first,first+2):
                    k=int(rng.integers(4));flip=bool(rng.random()<.5)
                    ax,ay=augment_pair(x[i],y[i],k,flip)
                    inputs.append(ax);targets.append(ay)
                    augmentation_trace.append(dict(step=step,patch=i,rotation_k=k,flip_horizontal=flip))
                bx,by=torch.stack(inputs),torch.stack(targets)
            model.train();opt.zero_grad();loss=loss_fn(model(bx),by)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();opt.step()
        if step%50==0 or step in (args.steps,args.extend_if_needed):
            scores=[];snap=args.out/f'step_{step:04d}';snap.mkdir()
            for i in range(args.patch_count):
                score,prob=measure(model,x[i:i+1],y[i:i+1]);scores.append(score)
                save_overlay(snap/f'patch_{i}_prediction.png',images[i],prob>=.5)
                tifffile.imwrite(snap/f'patch_{i}_probability.tif',prob)
            aggregate=metrics(*(sum(r[k] for r in scores) for k in ['tp','fp','fn']))
            row=dict(step=step,elapsed_seconds=time.time()-start,patches=scores,pooled=aggregate,
                     mean_loss=float(np.mean([r['loss'] for r in scores])))
            (args.out/'augmentation.json').write_text(json.dumps(augmentation_trace,indent=2))
            history.append(row);(args.out/'history.json').write_text(json.dumps(history,indent=2))
            print(f"step {step}/{args.steps} loss={row['mean_loss']:.4f} pooled Dice={aggregate['dice']:.4f} IoU={aggregate['iou']:.4f} per_patch_Dice={[round(r['dice'],4) for r in scores]} elapsed={row['elapsed_seconds']:.0f}s",flush=True)
            if step==args.steps:
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),width=16,step=step),args.out/f'step_{step:04d}.pt')
                if not args.extend_if_needed or all(r['dice']>=.95 for r in scores):
                    break
                print(f'Continuing same model and optimizer to {args.extend_if_needed} steps',flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),width=16,step=step),args.out/'last.pt')
    result=dict(final=history[-1],all_patches_reach_dice_095=all(r['dice']>=.95 for r in history[-1]['patches']))
    for f,h in cfg['source_sha256'].items():assert digest(source/f)==h
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    print('Finished; all patches meet .95:',result['all_patches_reach_dice_095'],flush=True)


if __name__=='__main__':main()
