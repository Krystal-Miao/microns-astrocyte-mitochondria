"""Evaluate 1/4/16-patch checkpoints with and without confirmed other astrocytes."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_astro_unet import save_overlay  # noqa: E402
from evaluate_fixed_patch_checkpoints import RUNS, predict, predict_tiled  # noqa: E402
from train_astro_unet import UNet, digest  # noqa: E402


class Pool:
    def __init__(self):
        self.tp = self.fp = self.fn = self.valid_pixels = 0
        self.predicted = self.target = 0
        self.bce_sum = self.soft_intersection = self.soft_probability = 0.0
        self.other_pixels = self.other_predicted = 0
        self.other_probability_sum = 0.0

    def add(self, probability, target, valid, other):
        probability = probability[valid].astype(np.float64)
        target = target[valid].astype(bool)
        target_float = target.astype(np.float64)
        prediction = probability >= 0.5
        self.tp += int(np.count_nonzero(prediction & target))
        self.fp += int(np.count_nonzero(prediction & ~target))
        self.fn += int(np.count_nonzero(~prediction & target))
        self.valid_pixels += int(valid.sum())
        self.predicted += int(prediction.sum())
        self.target += int(target.sum())
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        self.bce_sum += float((-target_float * np.log(clipped) - (1 - target_float) * np.log(1 - clipped)).sum())
        self.soft_intersection += float((probability * target_float).sum())
        self.soft_probability += float(probability.sum())
        other_probability = probability if np.array_equal(valid, other) else None
        if other_probability is not None:
            self.other_pixels += other_probability.size
            self.other_predicted += int(np.count_nonzero(other_probability >= 0.5))
            self.other_probability_sum += float(other_probability.sum())

    def add_other(self, probability, other):
        values = probability[other]
        self.other_pixels += int(values.size)
        self.other_predicted += int(np.count_nonzero(values >= 0.5))
        self.other_probability_sum += float(values.sum())

    def result(self):
        dice_den = 2 * self.tp + self.fp + self.fn
        iou_den = self.tp + self.fp + self.fn
        precision_den = self.tp + self.fp
        recall_den = self.tp + self.fn
        soft_dice = (2 * self.soft_intersection + 1) / (self.soft_probability + self.target + 1)
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.tp / precision_den if precision_den else 1.0,
            "recall": self.tp / recall_den if recall_den else 1.0,
            "dice": 2 * self.tp / dice_den if dice_den else 1.0,
            "iou": self.tp / iou_den if iou_den else 1.0,
            "loss": self.bce_sum / self.valid_pixels + 1 - soft_dice,
            "predicted_foreground_fraction": self.predicted / self.valid_pixels,
            "labeled_foreground_fraction": self.target / self.valid_pixels,
            "valid_pixels": self.valid_pixels,
            "other_astrocyte_pixels": self.other_pixels,
            "other_astrocyte_mean_probability": self.other_probability_sum / self.other_pixels if self.other_pixels else None,
            "other_astrocyte_predicted_positive_fraction": self.other_predicted / self.other_pixels if self.other_pixels else None,
        }


def slice_metrics(probability, target, valid):
    pool = Pool()
    pool.add(probability, target, valid, np.zeros_like(valid))
    return pool.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("samples/astro_864691136057675736_run1"))
    parser.add_argument("--audit", type=Path, default=Path("samples/astro_864691136057675736_run1/label_audit_01"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)

    sample_data = {}
    source_hashes = {}
    for sample in ("sample_01", "sample_02", "sample_03"):
        folder = args.data / sample
        image = tifffile.memmap(folder / "image.tif", mode="r")
        target = tifffile.memmap(folder / "mask.tif", mode="r").astype(bool)
        labels = tifffile.memmap(args.audit / sample / "audit_labels.tif", mode="r")
        if image.shape != target.shape or image.shape != labels.shape:
            raise ValueError(f"Shape mismatch for {sample}")
        valid_z = np.flatnonzero(np.ptp(image, axis=(1, 2)) > 0).tolist()
        heldout_z = [z for z in valid_z if z >= 112]
        sample_data[sample] = (image, target, labels, valid_z, heldout_z)
        source_hashes[sample] = {
            name: digest(folder / name) for name in ("image.tif", "mask.tif", "metadata.json")
        }

    pools = defaultdict(Pool)
    rows = []
    provenance = {}
    representative_z = {"sample_02": [60, 112]}
    assets = args.out / "assets"
    assets.mkdir()

    for model_label, run in RUNS.items():
        run_config = json.loads((run / "config.json").read_text())
        if run_config["source_sha256"] != source_hashes["sample_01"]:
            raise ValueError(f"{model_label} source hashes do not match sample_01")
        checkpoint = run / "last.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = UNet(saved["width"])
        model.load_state_dict(saved["model"])
        model.eval()
        provenance[model_label] = {
            "checkpoint": str(checkpoint), "sha256": digest(checkpoint),
            "step": saved["step"], "width": saved["width"],
        }
        for sample, (image, target, labels, valid_z, heldout_z) in sample_data.items():
            eval_z = valid_z if sample == "sample_02" else heldout_z
            for z in eval_z:
                other = np.asarray(labels[z]) == 2
                protocols = {
                    "full_512": predict(model, np.asarray(image[z])),
                    "tiled_256": predict_tiled(model, np.asarray(image[z])),
                }
                scopes = ["heldout_z"] if z in heldout_z else []
                if sample == "sample_02":
                    scopes.append("all_valid_z")
                for mode, probability in protocols.items():
                    for scope in scopes:
                        for policy, valid in {
                            "raw": np.ones(target[z].shape, dtype=bool),
                            "ignore_confirmed_other_astrocyte": ~other,
                        }.items():
                            key = (model_label, mode, sample, scope, policy)
                            pools[key].add(probability, np.asarray(target[z]), valid, np.zeros_like(valid))
                            if policy == "raw":
                                pools[key].add_other(probability, other)
                            row = {
                                "model": model_label, "mode": mode, "sample": sample,
                                "scope": scope, "policy": policy, "z": z,
                                **slice_metrics(probability, np.asarray(target[z]), valid),
                            }
                            rows.append(row)
                    if z in representative_z.get(sample, []):
                        stem = f'{model_label.replace(" ", "_")}_{mode}_{sample}_z{z:03d}'
                        save_overlay(assets / f"{stem}_prediction.png", np.asarray(image[z]), probability >= 0.5)

    # Reference/audit overlays already exist; save compact binary references beside predictions.
    image, target, labels, _, _ = sample_data["sample_02"]
    for z in representative_z["sample_02"]:
        save_overlay(assets / f"sample_02_z{z:03d}_target.png", np.asarray(image[z]), target[z])
        save_overlay(assets / f"sample_02_z{z:03d}_other_astrocyte.png", np.asarray(image[z]), np.asarray(labels[z]) == 2)
        shutil.copy2(args.audit / "sample_02" / "assets" / f"overlay_{z:03d}.png",
                     assets / f"sample_02_z{z:03d}_audit.png")

    summary = []
    for key, pool in sorted(pools.items()):
        model_label, mode, sample, scope, policy = key
        summary.append({"model": model_label, "mode": mode, "sample": sample,
                        "scope": scope, "policy": policy, **pool.result()})
    protocol = {
        "threshold": 0.5,
        "parameters_updated": False,
        "heldout_z": "nonconstant Z>=112 for each sample",
        "all_valid_z": "all nonconstant slices; reported only for sample_02 as a diagnostic",
        "raw": "all pixels scored against the single-target mask",
        "ignore_confirmed_other_astrocyte": "audit label code 2 excluded from metric denominators",
        "warning": "sample_02/03 are distinct volumes of the same target astrocyte, not independent biological replicates; all_valid_z is not validation for sample_01",
        "checkpoints": provenance,
        "source_sha256": source_hashes,
    }
    (args.out / "results.json").write_text(json.dumps({"protocol": protocol, "summary": summary}, indent=2))
    with (args.out / "slice_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    html = '''<!doctype html><meta charset="utf-8"><title>Audit-masked checkpoint evaluation</title><style>
body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:16px;background:#18232b;color:#eee}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #647482}img{width:24%;image-rendering:pixelated}select{padding:7px;margin:5px}a{color:#8dbaff}.note{line-height:1.55}</style>
<h1>1 / 4 / 16-patch checkpoints: audit-masked evaluation</h1><p class="note">Frozen inference；threshold 0.5。Raw 使用原 single-target mask；masked 仅忽略 confirmed other astrocyte（不忽略黄色 unknown）。sample_01 的 heldout Z 是未参与这些模型 training 的 block；sample_02/03 是同一 target astrocyte 的其他 volumes，不是独立 biological replicates。</p>
<h2>Summary</h2><table id="table"><tr><th>Model</th><th>Mode</th><th>Sample / scope</th><th>Policy</th><th>Loss</th><th>Dice</th><th>IoU</th><th>Precision</th><th>Recall</th><th>Pred / label fg</th><th>Other astro P / positive</th></tr></table>
<h2>sample_02 representative overlays</h2><select id="model"></select><select id="mode"><option>full_512</option><option>tiled_256</option></select><select id="z"><option>60</option><option>112</option></select><p>预测（红）｜target（红）｜confirmed other astrocyte（红；只是单独展示类别）｜audit 彩色图</p><img id="pred"><img id="target"><img id="other"><img id="audit"><p><a href="slice_metrics.csv">Per-slice CSV</a> · <a href="results.json">Full JSON</a></p><script>'''
    html += "const rows=" + json.dumps(summary).replace("<", "\\u003c") + ";const models=" + json.dumps(list(RUNS)) + ";"
    html += '''const t=document.getElementById('table');function f(x){return x==null?'—':x.toFixed(4)}rows.forEach(r=>{const tr=t.insertRow();[r.model,r.mode,r.sample+' / '+r.scope,r.policy,f(r.loss),f(r.dice),f(r.iou),f(r.precision),f(r.recall),(100*r.predicted_foreground_fraction).toFixed(2)+'% / '+(100*r.labeled_foreground_fraction).toFixed(2)+'%',f(r.other_astrocyte_mean_probability)+' / '+(r.other_astrocyte_predicted_positive_fraction==null?'—':(100*r.other_astrocyte_predicted_positive_fraction).toFixed(2)+'%')].forEach(x=>tr.insertCell().textContent=x)});const m=document.getElementById('model'),mode=document.getElementById('mode'),z=document.getElementById('z');models.forEach(x=>m.add(new Option(x,x.replace(' ','_'))));function show(){const zi=String(z.value).padStart(3,'0');document.getElementById('pred').src='assets/'+m.value+'_'+mode.value+'_sample_02_z'+zi+'_prediction.png';document.getElementById('target').src='assets/sample_02_z'+zi+'_target.png';document.getElementById('other').src='assets/sample_02_z'+zi+'_other_astrocyte.png';document.getElementById('audit').src='assets/sample_02_z'+zi+'_audit.png'}[m,mode,z].forEach(x=>x.onchange=show);show();</script>'''
    (args.out / "index.html").write_text(html)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
