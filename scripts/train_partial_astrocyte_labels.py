"""Train a 2D U-Net using confirmed astrocyte positives and neuron negatives only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_fixed_patch_checkpoints import predict_tiled  # noqa: E402
from train_astro_unet import UNet, metrics  # noqa: E402
from train_foreground_aware_pilot import ranking_metrics, top_left_containing  # noqa: E402


def partial_objective(logits, target, valid):
    probability = logits.sigmoid()
    valid_float = valid.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid_float).sum() / valid_float.sum().clamp_min(1)
    spatial_dims = tuple(range(1, logits.ndim))
    intersection = (probability * target * valid_float).sum(spatial_dims)
    denominator = ((probability + target) * valid_float).sum(spatial_dims)
    soft_dice = (2 * intersection + 1) / (denominator + 1)
    return bce + (1 - soft_dice).mean()


def load(root, names):
    data = {}
    for name in names:
        sample = root / name
        audit = root / "label_audit_01" / name / "audit_labels.tif"
        image = tifffile.memmap(sample / "image.tif", mode="r")
        labels = tifffile.memmap(audit, mode="r")
        if image.shape != labels.shape:
            raise ValueError(f"Shape mismatch for {name}")
        valid_z = [z for z in range(image.shape[0]) if np.ptp(image[z]) > 0]
        data[name] = (image, labels, valid_z)
    return data


def draw_patch(data, rng, patch, foreground_centered):
    name = list(data)[int(rng.integers(len(data)))]
    image, labels, valid_z = data[name]
    if foreground_centered:
        eligible = [z for z in valid_z if np.any((labels[z] == 1) | (labels[z] == 2))]
        z = int(eligible[int(rng.integers(len(eligible)))])
        points = np.argwhere((labels[z] == 1) | (labels[z] == 2))
        ay, ax = points[int(rng.integers(len(points)))]
        y, x = top_left_containing(rng, int(ay), int(ax), patch, image.shape[1], image.shape[2])
    else:
        z = int(valid_z[int(rng.integers(len(valid_z)))])
        y = int(rng.integers(0, image.shape[1] - patch + 1))
        x = int(rng.integers(0, image.shape[2] - patch + 1))
    raw = np.asarray(image[z, y:y + patch, x:x + patch]).copy()
    audit = np.asarray(labels[z, y:y + patch, x:x + patch]).copy()
    target = ((audit == 1) | (audit == 2)).astype(np.float32)
    valid = ((audit == 1) | (audit == 2) | (audit == 3))
    return raw, target, valid, {"sample": name, "z": z, "y": y, "x": x,
                                "valid_fraction": float(valid.mean()),
                                "positive_fraction": float(target[valid].mean())}


@torch.no_grad()
def validate(model, data, centers):
    probabilities, truths = [], []
    tp = fp = fn = 0
    losses = []
    model.eval()
    for name, (image, labels, valid_z) in data.items():
        for z in centers:
            if z not in valid_z:
                continue
            probability = predict_tiled(model, np.asarray(image[z]), 256, 128)
            audit = np.asarray(labels[z])
            truth = (audit == 1) | (audit == 2)
            valid = truth | (audit == 3)
            prediction = probability >= 0.5
            tp += int(np.count_nonzero(prediction & truth & valid))
            fp += int(np.count_nonzero(prediction & ~truth & valid))
            fn += int(np.count_nonzero(~prediction & truth & valid))
            clipped = np.clip(probability, 1e-6, 1 - 1e-6)
            logits = torch.from_numpy(np.log(clipped / (1 - clipped)))[None, None].float()
            target = torch.from_numpy(truth.copy())[None, None].float()
            valid_tensor = torch.from_numpy(valid.copy())[None, None]
            losses.append(float(partial_objective(logits, target, valid_tensor)))
            probabilities.append(probability[valid])
            truths.append(truth[valid])
    score = metrics(tp, fp, fn)
    probability = np.concatenate(probabilities)
    truth = np.concatenate(truths)
    score.update(loss=float(np.mean(losses)), evaluated_pixels=int(len(truth)),
                 labeled_foreground_fraction=float(truth.mean()),
                 predicted_foreground_fraction=float((probability >= 0.5).mean()),
                 foreground_probability_mean=float(probability[truth].mean()),
                 background_probability_mean=float(probability[~truth].mean()))
    score.update(ranking_metrics(probability, truth))
    return score


def write_report(out, result):
    score = result["best_validation"]
    prevalence = score["labeled_foreground_fraction"]
    all_positive_dice = 2 * prevalence / (1 + prevalence)
    html = f'''<!doctype html><meta charset="utf-8"><title>Partial-label pilot</title>
<style>body{{font:16px system-ui;max-width:950px;margin:30px auto;padding:16px;background:#18232b;color:#eee}}table{{border-collapse:collapse}}td,th{{padding:9px;border:1px solid #647482}}a{{color:#8dbaff}}</style>
<h1>Confirmed-label-only pilot</h1>
<p>Training: sample_01/02. Validation: sample_03. Positive = target or confirmed other astrocyte; negative = confirmed neuron; unclassified pixels are ignored. Scope: {result['scope']}.</p>
<table><tr><th>Metric</th><th>Baseline</th><th>Model</th></tr>
<tr><td>Average precision</td><td>{prevalence:.4f} (random ranking)</td><td>{score['average_precision']:.4f}</td></tr>
<tr><td>Best-threshold Dice</td><td>{all_positive_dice:.4f} (predict all positive)</td><td>{score['best_threshold_dice']:.4f} @ {score['best_threshold']:.3f}</td></tr>
<tr><td>Dice at threshold 0.5</td><td>—</td><td>{score['dice']:.4f}</td></tr>
<tr><td>P(foreground | positive / negative)</td><td>—</td><td>{score['foreground_probability_mean']:.4f} / {score['background_probability_mean']:.4f}</td></tr></table>
<p>Ignoring unknown pixels improves ranking modestly, but foreground/background separation remains extremely small. Label incompleteness is one bottleneck, not the only bottleneck.</p>
<p><a href="result.json">Result JSON</a> · <a href="history.json">History</a> · <a href="config.json">Config</a></p>'''
    (out / "index.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("samples/astro_864691136057675736_run1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train = load(args.root, ["sample_01", "sample_02"])
    validation = load(args.root, ["sample_03"])
    train_root_ids = [json.loads((args.root / name / "metadata.json").read_text())["root_id"]
                      for name in ("sample_01", "sample_02")]
    validation_root_ids = [json.loads((args.root / "sample_03" / "metadata.json").read_text())["root_id"]]
    distinct_cells = len(set(train_root_ids + validation_root_ids)) > 1
    validation_z = [18, 47, 76, 105]
    model = UNet(16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    history, trace = [], []
    best_key = (-1.0, float("-inf"))
    start = time.time()
    for step in range(1, args.steps + 1):
        draws = [draw_patch(train, rng, 256, True), draw_patch(train, rng, 256, False)]
        images = np.stack([row[0] for row in draws])[:, None]
        targets = np.stack([row[1] for row in draws])[:, None]
        valid = np.stack([row[2] for row in draws])[:, None]
        trace.extend({"step": step, "batch_index": i, **row[3]} for i, row in enumerate(draws))
        x = torch.from_numpy(images).float() / 255
        y = torch.from_numpy(targets).float()
        v = torch.from_numpy(valid)
        model.train(); optimizer.zero_grad()
        logits = model(x)
        loss = partial_objective(logits, y, v)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward(); optimizer.step()
        if step % 50 == 0 or step == args.steps:
            score = validate(model, validation, validation_z)
            row = {"step": step, "train_loss": float(loss.detach()),
                   "elapsed_seconds": time.time() - start, **score}
            history.append(row)
            key = (score["average_precision"], -score["loss"])
            if key > best_key:
                best_key = key
                torch.save({"model": model.state_dict(), "width": 16, "step": step,
                            "validation": score}, args.out / "best.pt")
            print(f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                  f"val_dice={score['dice']:.4f} AP={score['average_precision']:.4f} "
                  f"best_dice={score['best_threshold_dice']:.4f} "
                  f"Pfg={score['foreground_probability_mean']:.3f} "
                  f"Pbg={score['background_probability_mean']:.3f}", flush=True)
    best = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)
    result = {"best_step": best["step"], "best_validation": best["validation"],
              "last_validation": history[-1], "elapsed_seconds": time.time() - start,
              "scope": ("partial-label cell-level development test; two astrocytes train, one distinct astrocyte validation"
                        if distinct_cells else
                        "partial-label within-cell development test; spatial crops of the same target astrocyte")}
    (args.out / "config.json").write_text(json.dumps({
        "positive_codes": [1, 2], "negative_codes": [3], "ignored_codes": [0, 4, 5, 6],
        "train_samples": ["sample_01", "sample_02"], "validation_samples": ["sample_03"],
        "validation_z": validation_z, "steps": args.steps, "seed": args.seed,
        "train_root_ids": train_root_ids, "validation_root_ids": validation_root_ids,
        "learning_rate": 0.0003, "patch": 256, "batch_size": 2,
    }, indent=2) + "\n")
    (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.out / "sampling.json").write_text(json.dumps(trace, indent=2) + "\n")
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(args.out, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
