"""Frozen-model comparison on 20 evenly spaced recorded training draws."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import tifffile
import torch
from train_astro_unet import UNet, digest, metrics
from diagnose_astro_unet import save_overlay


def transform(a, r):
    a=np.rot90(a,r['rotation_k'])
    return a[:,::-1] if r['flip_horizontal'] else a


def score(p, y):
    y=y.astype(bool);pred=p>=.5
    return dict(**metrics(int((pred&y).sum()),int((pred&~y).sum()),int((~pred&y).sum())),
                probability_min=float(p.min()),probability_max=float(p.max()),
                foreground_probability_mean=float(p[y].mean()) if y.any() else None,
                background_probability_mean=float(p[~y].mean()) if (~y).any() else None,
                predicted_foreground_fraction=float(pred.mean()),label_fraction=float(y.mean()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,default=Path('samples/unet_s01_random256_lr0003_01'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(4)
    config=json.loads((args.run/'config.json').read_text());source=Path(config['source'])
    for f,h in config['source_sha256'].items():
        if digest(source/f)!=h: raise ValueError('Source mismatch')
    trace=[json.loads(s) for s in (args.run/'sampling.jsonl').read_text().splitlines()]
    indices=np.linspace(0,len(trace)-1,20,dtype=int).tolist()
    chosen=[trace[i] for i in indices]
    train=set(tuple(r) for r in json.loads((args.run/'split.json').read_text())['train'])
    assert all((r['sample'],r['z']) in train for r in chosen)
    checkpoint=args.run/'last.pt';before=digest(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=UNet(saved['width']);model.load_state_dict(saved['model']);model.eval()
    a=tifffile.imread(source/'image.tif');m=tifffile.imread(source/'mask.tif')
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'selection.json').write_text(json.dumps(dict(indices=indices,records=chosen,
        policy='20 evenly spaced draw indices, chosen before inference; no label/output selection',
        checkpoint=str(checkpoint),checkpoint_sha256=before,checkpoint_step=saved['step'],
        source_sha256=config['source_sha256'],script_sha256=digest(Path(__file__))),indent=2))
    rows=[];cache={};pooled={k:[] for k in ['patch','full_context','exact_augmented']};labels=[]
    cards=[]
    with torch.no_grad():
        def predict(x):
            return model(torch.from_numpy(x.copy())[None,None].float()/255).sigmoid()[0,0].numpy()
        for i,r in enumerate(chosen):
            z,y,x,n=r['z'],r['y'],r['x'],r['size']
            crop=a[z,y:y+n,x:x+n];target=m[z,y:y+n,x:x+n]
            if z not in cache: cache[z]=predict(a[z])
            small=predict(crop);full=cache[z][y:y+n,x:x+n]
            aug=predict(transform(crop,r))
            # Restore augmented predictions to source coordinates before comparison.
            if r['flip_horizontal']: aug=aug[:,::-1]
            aug=np.rot90(aug,-r['rotation_k']).copy()
            predictions=dict(patch=small,full_context=full,exact_augmented=aug)
            record=dict(index=i,trace_index=indices[i],**r,
                mean_absolute_context_difference=float(np.abs(small-full).mean()),
                center128_context_difference=float(np.abs(small-full)[64:192,64:192].mean()))
            labels.append(target)
            save_overlay(args.out/f'{i:02d}_reference.png',crop,target)
            for mode,prob in predictions.items():
                record[mode]=score(prob,target);pooled[mode].append(prob)
                tifffile.imwrite(args.out/f'{i:02d}_{mode}_probability.tif',prob)
                save_overlay(args.out/f'{i:02d}_{mode}.png',crop,prob>=.5)
            rows.append(record)
            cards.append(f'<h3>#{i+1} · train step {r["step"]} · Z={z}, X={x}, Y={y}</h3><div class="row">'+''.join(f'<figure><img src="{i:02d}_{mode}.png"><figcaption>{title}</figcaption></figure>' for mode,title in [('reference','参考mask'),('patch','256单独输入'),('full_context','512预测后裁剪'),('exact_augmented','回放训练增强')])+'</div><p>'+ '；'.join(f'{mode}: Dice={record[mode]["dice"]:.4f}, P范围={record[mode]["probability_min"]:.3f}–{record[mode]["probability_max"]:.3f}' for mode in predictions)+'</p>')
    summary={k:score(np.stack(v),np.stack(labels)) for k,v in pooled.items()}
    summary['mean_absolute_context_difference']=float(np.mean([r['mean_absolute_context_difference'] for r in rows]))
    summary['scope']='20 recorded training draws, possibly overlapping; pooled counts include repeated pixels; not validation or generalization'
    assert digest(checkpoint)==before
    for f,h in config['source_sha256'].items(): assert digest(source/f)==h
    (args.out/'results.json').write_text(json.dumps(dict(summary=summary,patches=rows),indent=2))
    flat=[dict(index=r['index'],z=r['z'],x=r['x'],y=r['y'],training_step=r['step'],mode=k,**r[k]) for r in rows for k in pooled]
    with (args.out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    html='<!doctype html><meta charset="utf-8"><title>256/512 同位置推理诊断</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;background:#18232b;color:#eee;padding:16px}.row{display:flex;gap:10px}figure{flex:1;margin:0}img{width:100%}p{line-height:1.6}</style><h1>同一个最终模型，同一XYZ位置</h1><p>20个按训练抽样顺序等间距选取的记录；仅推理，无参数更新。另回放训练时的旋转与翻转，再还原坐标。阈值固定0.5。可能重叠的训练区域，不是验证集。</p>'
    html+='<pre>'+json.dumps(summary,indent=2)+'</pre>'+''.join(cards)
    (args.out/'index.html').write_text(html)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
