"""Build an offline comparison of paired sampling runs."""
import json
from pathlib import Path
import html

def main():
    base=Path('samples')
    names=['unet_s01_random256_lr0003_01','unet_s01_random256_noaug_01']
    out=base/'unet_s01_random256_aug_comparison_01'
    histories=[json.loads((base/n/'history.json').read_text()) for n in names]
    traces=[[json.loads(l) for l in (base/n/'sampling.jsonl').read_text().splitlines()] for n in names]
    keys=['sample','z','x','y','size','step','foreground_fraction']
    assert len(traces[0])==len(traces[1])==2000
    assert all(all(a[k]==b[k] for k in keys) for a,b in zip(*traces))
    assert all(t['rotation_k']==0 and not t['flip_horizontal'] for t in traces[1])
    results=[json.loads((base/n/'result.json').read_text()) for n in names]
    out.mkdir(exist_ok=True)
    parts=['<!doctype html><meta charset="utf-8"><title>Random patch augmentation ablation</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;background:#18232b;color:#eee;padding:20px}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #647482}svg{width:100%;background:#24333e}img{width:32%}select,input{margin:12px}a{color:#8dbaff}</style><h1>sample_01: random patch augmentation ablation</h1><p>相同 random initialization、2000 次 crop 的 XYZ 和顺序、1000 steps、batch size 2、learning rate 0.0003。仅关闭 rotation / horizontal flip。Threshold 0.5。Training monitor 和 validation 均为整张 512 × 512 inference；不是 independent test。</p><p>橙色：augmentation ON；蓝色：augmentation OFF。</p><table><tr><th>Run</th><th>Monitor loss / Dice / IoU</th><th>Validation loss / Dice / IoU</th><th>Validation predicted foreground</th><th>Time</th></tr>']
    for label,r in zip(['ON','OFF'],results):
        fields=[' / '.join(f'{r[s][k]:.4f}' for k in ['loss','dice','iou']) for s in ['training_monitor','validation']]
        parts.append(f"<tr><td>{label}</td><td>{fields[0]}</td><td>{fields[1]}</td><td>{r['validation']['predicted_foreground_fraction']:.2%}</td><td>{r['elapsed_seconds']/60:.2f} min</td></tr>")
    parts.append('</table>')
    for split in ['training_monitor','validation']:
        for metric in ['loss','dice','iou']:
            rows=[[r for r in h if split in r] for h in histories]
            top=max(1,max(r[split][metric] for h in rows for r in h))*1.05
            parts.append(f'<h3>{split}: {metric}</h3><svg viewBox="0 0 1000 210"><text x="5" y="20" fill="white">{top:.2f}</text><text x="5" y="185" fill="white">0</text>')
            for h,color in zip(rows,['#ffb454','#8dbaff']):
                pts=' '.join(f"{50+r['step']*.9:.1f},{180-r[split][metric]/top*160:.1f}" for r in h)
                parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
            parts.append('<text x="50" y="205" fill="white">step 0</text><text x="900" y="205" fill="white">1000</text></svg>')
    parts.append('<h2>Prediction overlays</h2><p>左：ON；中：OFF；右：reference mask。Z=0/47 属于 training，Z=112/127 属于 validation。</p><select id="z"><option>0</option><option>47</option><option>112</option><option>127</option></select><input id="step" type="range" min="100" max="1000" step="100" value="1000"><span id="label"></span><div><img id="on"><img id="off"><img id="ref"></div>')
    parts.append('<script>const names='+json.dumps(names)+';function show(){const z=document.getElementById("z").value.padStart(3,"0"),s=document.getElementById("step").value;document.getElementById("label").textContent="Step "+s;["on","off"].forEach((id,i)=>document.getElementById(id).src="../"+names[i]+"/step_"+s.padStart(4,"0")+"/z"+z+"_prediction.png");document.getElementById("ref").src="../"+names[0]+"/z"+z+"_reference.png"}document.getElementById("z").onchange=show;document.getElementById("step").oninput=show;show();</script>')
    (out/'index.html').write_text(''.join(parts))
    (out/'comparison.json').write_text(json.dumps(dict(crop_schedule_identical=True,runs=dict(zip(names,results))),indent=2))
    print(out/'index.html')

if __name__=='__main__': main()
