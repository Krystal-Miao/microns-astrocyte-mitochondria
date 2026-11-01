"""Threshold and ranking diagnostics on the shared held-out-Z validation set."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
import torch

from evaluate_fixed_patch_checkpoints import RUNS, predict, predict_tiled
from train_astro_unet import UNet, digest


def counts_at_threshold(probability, target, threshold):
    prediction = probability >= threshold
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, dice=dice, iou=iou,
                predicted_foreground_fraction=float(prediction.mean()))


def average_precision(probability, target):
    """Non-interpolated AP with equal-score pixels handled as one threshold."""
    order = np.argsort(-probability, kind="stable")
    scores = probability[order]
    truth = target[order].astype(np.int64)
    tp = np.cumsum(truth)
    fp = np.cumsum(1 - truth)
    distinct_end = np.r_[scores[1:] != scores[:-1], True]
    tp = tp[distinct_end]
    fp = fp[distinct_end]
    precision = tp / (tp + fp)
    recall = tp / target.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def distribution(probability, target):
    result = {}
    for name, values in [("foreground", probability[target]), ("background", probability[~target])]:
        result[name] = dict(
            count=int(values.size), mean=float(values.mean()), std=float(values.std()),
            q05=float(np.quantile(values, .05)), q25=float(np.quantile(values, .25)),
            median=float(np.median(values)), q75=float(np.quantile(values, .75)),
            q95=float(np.quantile(values, .95)),
        )
    return result


def points_svg(rows, x_key, y_key, color):
    return " ".join(
        f'{55 + row[x_key] * 900:.1f},{175 - row[y_key] * 145:.1f}' for row in rows
    ), color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        default=Path("samples/astro_864691136057675736_run1/sample_01"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)

    image = tifffile.imread(args.source / "image.tif")
    mask = tifffile.imread(args.source / "mask.tif")
    validation_z = [z for z in range(112, 128) if np.ptp(image[z]) > 0]
    if validation_z != [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127]:
        raise ValueError("Unexpected validation block")
    target = mask[validation_z].astype(bool).reshape(-1)
    # Dense logarithmic coverage is needed because some models place most
    # probabilities far below 0.01; a purely linear grid would miss them.
    thresholds = np.unique(np.r_[0.0, np.logspace(-8, -1, 71), np.linspace(.1, 1, 91)])
    rows = []
    summary = {}
    probabilities_dir = args.out / "probabilities"
    probabilities_dir.mkdir()

    for label, run in RUNS.items():
        checkpoint = run / "last.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = UNet(saved["width"])
        model.load_state_dict(saved["model"])
        model.eval()
        summary[label] = {}
        for mode in ["full_512", "tiled_256"]:
            volumes = []
            for z in validation_z:
                volumes.append(predict(model, image[z]) if mode == "full_512" else predict_tiled(model, image[z]))
            probability_volume = np.stack(volumes).astype(np.float32)
            probability = probability_volume.reshape(-1)
            tifffile.imwrite(probabilities_dir / f'{label.replace(" ", "_")}_{mode}.tif', probability_volume)
            sweep = []
            for threshold in thresholds:
                row = dict(model=label, checkpoint_step=saved["step"], mode=mode,
                           threshold=float(threshold), **counts_at_threshold(probability, target, threshold))
                rows.append(row)
                sweep.append(row)
            best = max(sweep, key=lambda row: (row["dice"], -abs(row["threshold"] - .5)))
            summary[label][mode] = dict(
                average_precision=average_precision(probability, target),
                foreground_prevalence=float(target.mean()),
                fixed_threshold_05=next(row for row in sweep if row["threshold"] == .5),
                best_grid_threshold=best,
                probability_distribution=distribution(probability, target),
                probability_sha256=digest(probabilities_dir / f'{label.replace(" ", "_")}_{mode}.tif'),
            )

    with (args.out / "threshold_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = dict(
        validation_z=validation_z, excluded_constant_z=[124],
        thresholds="0, logspace 1e-8 through 0.1 (71 points), then 0.1 through 1.0 step 0.01",
        average_precision="non-interpolated precision-recall AP; tied probabilities grouped",
        warning="Best threshold is selected and scored on the same validation set: diagnostic only, not test performance",
        parameters_updated=False,
        checkpoints={label: dict(path=str(run / "last.pt"), sha256=digest(run / "last.pt")) for label, run in RUNS.items()},
    )
    (args.out / "results.json").write_text(json.dumps(dict(protocol=protocol, summary=summary), indent=2))

    colors = {"1 patch": "#ffb454", "4 patches": "#8dbaff", "16 patches": "#68d8ac"}
    html = '''<!doctype html><meta charset="utf-8"><title>Validation threshold diagnostics</title><style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:16px;background:#18232b;color:#eee}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #647482}svg{width:100%;background:#24333e}a{color:#8dbaff}</style><h1>Shared validation: threshold and ranking diagnostics</h1><p>相同 sample_01 held-out Z block。Best threshold 在这组 validation data 上选择，只用于诊断，不能作为 independent test performance。</p><table><tr><th>Model</th><th>Mode</th><th>AP</th><th>Dice @0.5</th><th>Best grid threshold</th><th>Best grid Dice</th><th>Precision / Recall at best</th></tr>'''
    for label in RUNS:
        for mode in ["full_512", "tiled_256"]:
            result = summary[label][mode]
            fixed = result["fixed_threshold_05"]
            best = result["best_grid_threshold"]
            html += f'<tr><td>{label}</td><td>{mode}</td><td>{result["average_precision"]:.4f}</td><td>{fixed["dice"]:.4f}</td><td>{best["threshold"]:.2f}</td><td>{best["dice"]:.4f}</td><td>{best["precision"]:.3f} / {best["recall"]:.3f}</td></tr>'
    html += '</table>'
    for mode in ["full_512", "tiled_256"]:
        html += f'<h2>{mode}: Precision–Recall curve</h2><svg viewBox="0 0 1000 205"><path d="M55 30V175H955" fill="none" stroke="#aaa"/><text x="5" y="35" fill="white">1.0</text><text x="25" y="180" fill="white">0</text><text x="55" y="200" fill="white">Recall 0</text><text x="875" y="200" fill="white">Recall 1</text>'
        for label in RUNS:
            selected = [row for row in rows if row["model"] == label and row["mode"] == mode]
            points, color = points_svg(selected, "recall", "precision", colors[label])
            html += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/><text x="{650}" y="{45 + list(RUNS).index(label)*20}" fill="{color}">{label}</text>'
        html += '</svg>'
        html += f'<h2>{mode}: Dice by threshold</h2><svg viewBox="0 0 1000 205"><path d="M55 30V175H955" fill="none" stroke="#aaa"/><text x="5" y="35" fill="white">1.0</text><text x="25" y="180" fill="white">0</text><text x="55" y="200" fill="white">threshold 0</text><text x="875" y="200" fill="white">threshold 1</text>'
        for label in RUNS:
            selected = [row for row in rows if row["model"] == label and row["mode"] == mode]
            points, color = points_svg(selected, "threshold", "dice", colors[label])
            html += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        html += '</svg>'
    html += '<p><a href="threshold_metrics.csv">All threshold metrics</a></p>'
    (args.out / "index.html").write_text(html)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
