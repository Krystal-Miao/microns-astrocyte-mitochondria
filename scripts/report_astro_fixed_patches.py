"""Offline training-diagnostic viewer for fixed-patch runs."""
import argparse
import csv
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    cfg=json.loads((args.run/'config.json').read_text())
    history=json.loads((args.run/'history.json').read_text())
    count=len(history[0]['patches']); last=history[-1]
    rows=[dict(step=r['step'],patch=i,**s) for r in history for i,s in enumerate(r['patches'])]
    with (args.run/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    parts=['<!doctype html><meta charset="utf-8"><title>Fixed patches training diagnostic</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:16px;background:#18232b;color:#eee}svg{width:100%;background:#24333e}img{width:48%}input{width:60%}td,th{padding:8px;border:1px solid #647482}table{border-collapse:collapse}a{color:#8dbaff}</style>']
    parts.append(f'<h1>{count} fixed patches: training diagnostic</h1><p>Fresh seed {cfg["seed"]}; learning rate {cfg["learning_rate"]}; batch size 2; augmentation {cfg["augmentation"]}; threshold 0.5. Training scores only, not validation. Crops may overlap.</p>')
    parts.append(f'<p>Latest step {last["step"]}; mean loss {last["mean_loss"]:.4f}; pooled Dice {last["pooled"]["dice"]:.4f}; IoU {last["pooled"]["iou"]:.4f}; elapsed {last["elapsed_seconds"]/60:.2f} min.</p>')
    for metric in ['loss','dice','iou']:
        vals=[r['mean_loss'] if metric=='loss' else r['pooled'][metric] for r in history]
        top=max(1,max(vals))*1.05
        pts=' '.join(f'{50+r["step"]/max(1,last["step"])*900:.1f},{180-v/top*160:.1f}' for r,v in zip(history,vals))
        parts.append(f'<h3>{"Mean" if metric=="loss" else "Pooled"} {metric}</h3><svg viewBox="0 0 1000 210"><text x="5" y="20" fill="white">{top:.2f}</text><text x="5" y="180" fill="white">0</text><polyline points="{pts}" fill="none" stroke="#68d8ac" stroke-width="3"/><text x="50" y="205" fill="white">step 0</text><text x="900" y="205" fill="white">{last["step"]}</text></svg>')
    parts.append('<h2>Per-patch scores</h2><table><tr><th>Patch</th><th>Z / X / Y</th><th>Loss</th><th>Dice</th><th>IoU</th><th>Foreground: predicted / labeled</th></tr>')
    for i,(s,r) in enumerate(zip(last['patches'],cfg['selection'])):
        parts.append(f'<tr><td>{i}</td><td>{r["z"]} / {r["x"]} / {r["y"]}</td><td>{s["loss"]:.4f}</td><td>{s["dice"]:.4f}</td><td>{s["iou"]:.4f}</td><td>{s["predicted_foreground_fraction"]:.2%} / {s["labeled_foreground_fraction"]:.2%}</td></tr>')
    parts.append('</table><h2>Prediction / reference</h2><select id="patch">'+''.join(f'<option>{i}</option>' for i in range(count))+'</select>')
    parts.append(f'<input id="step" type="range" min="0" max="{len(history)-1}" value="{len(history)-1}"><p id="metric"></p><img id="pred"><img id="ref"><p><a href="metrics.csv">Per-patch CSV</a></p>')
    parts.append('<script>const history='+json.dumps(history)+';function show(){const i=Number(document.getElementById("patch").value),r=history[Number(document.getElementById("step").value)],s=r.patches[i];document.getElementById("pred").src="step_"+String(r.step).padStart(4,"0")+"/patch_"+i+"_prediction.png";document.getElementById("ref").src="patch_"+i+"_reference.png";document.getElementById("metric").textContent="Step "+r.step+" | loss "+s.loss.toFixed(4)+" | Dice "+s.dice.toFixed(4)+" | IoU "+s.iou.toFixed(4)}document.getElementById("step").oninput=show;document.getElementById("patch").onchange=show;show();</script>')
    (args.run/'index.html').write_text(''.join(parts))
    print(args.run/'index.html')
if __name__=='__main__':main()
