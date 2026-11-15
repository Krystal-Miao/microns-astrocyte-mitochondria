"""Audit frozen 3D predictions by MICrONS label code and save FP examples."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_partial_astrocyte_3d import (  # noqa: E402
    UNet3DNNStyle,
    load,
    normalize_raw,
    uint8_normalization_stats,
)


LABELS = {
    0: "no_segment",
    1: "target_astrocyte",
    2: "confirmed_other_astrocyte",
    3: "confirmed_neuron",
    4: "other_classified_cell",
    5: "unclassified_queried_segment",
    6: "tiny_unqueried_segment",
}


def add(stats, labels, probability, threshold):
    prediction = probability >= threshold
    for code, name in LABELS.items():
        selected = labels == code
        count = int(selected.sum())
        row = stats[name]
        row["voxels"] += count
        row["predicted_positive"] += int(np.count_nonzero(prediction & selected))
        row["probability_sum"] += float(probability[selected].sum())


def finalize(stats):
    result = {}
    for name, row in stats.items():
        voxels = row["voxels"]
        result[name] = {"voxels": voxels,
                        "predicted_positive": row["predicted_positive"],
                        "predicted_positive_fraction": row["predicted_positive"] / voxels if voxels else None,
                        "mean_probability": row["probability_sum"] / voxels if voxels else None}
    return result


def save_png(path, rgb):
    """Write an RGB uint8 PNG using only the standard library."""
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 array")
    scanlines = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                     chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(scanlines, 6)) + chunk(b"IEND", b""))


def panel(raw, labels, probability, threshold):
    raw = np.asarray(raw, dtype=np.uint8)
    base = np.repeat(raw[..., None], 3, axis=2)
    prediction = probability >= threshold
    truth = (labels == 1) | (labels == 2)
    neuron = labels == 3
    fp = prediction & neuron
    pred_overlay = base.astype(np.float32)
    pred_overlay[prediction] = pred_overlay[prediction] * 0.45 + np.array([255, 50, 50]) * 0.55
    audit_overlay = base.astype(np.float32)
    audit_overlay[truth] = audit_overlay[truth] * 0.38 + np.array([0, 235, 205]) * 0.62
    audit_overlay[fp] = audit_overlay[fp] * 0.25 + np.array([255, 0, 220]) * 0.75
    audit_overlay[neuron & ~fp] = audit_overlay[neuron & ~fp] * 0.62 + np.array([80, 145, 255]) * 0.38
    return np.concatenate([base, pred_overlay.astype(np.uint8), audit_overlay.astype(np.uint8)], axis=1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assets = args.out / "assets"; assets.mkdir()
    config = json.loads((args.run / "config.json").read_text())
    protocol = json.loads((args.test / "protocol.json").read_text())
    test_result = json.loads((args.test / "result.json").read_text())
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    threshold = float(test_result["locked_validation_threshold"])
    patch = tuple(protocol["patch_zyx"])
    data = load(args.roots, protocol["coordinates_zyx"].keys())
    normalizers = {name: uint8_normalization_stats(image) for name, (image, _) in data.items()}
    model = UNet3DNNStyle(config["width"]).to(args.device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    total_stats = defaultdict(lambda: {"voxels": 0, "predicted_positive": 0, "probability_sum": 0.0})
    sample_stats = {}; patch_rows = []; candidates = []
    for name, (image, labels) in data.items():
        current = defaultdict(lambda: {"voxels": 0, "predicted_positive": 0, "probability_sum": 0.0})
        for patch_index, (z, y, x) in enumerate(protocol["coordinates_zyx"][name]):
            raw = np.asarray(image[z:z + patch[0], y:y + patch[1], x:x + patch[2]]).copy()
            audit = np.asarray(labels[z:z + patch[0], y:y + patch[1], x:x + patch[2]]).copy()
            probability = model(normalize_raw(raw, normalizers[name])[None, None].to(args.device)).sigmoid()[0, 0].cpu().numpy()
            add(total_stats, audit, probability, threshold); add(current, audit, probability, threshold)
            prediction = probability >= threshold
            fp_volume = prediction & (audit == 3)
            fp_by_z = fp_volume.reshape(patch[0], -1).sum(axis=1)
            center = int(np.argmax(fp_by_z))
            row = {"sample": name, "patch_index": patch_index, "start_zyx": [z, y, x],
                   "representative_source_z": z + center,
                   "neuron_voxels": int(np.count_nonzero(audit == 3)),
                   "neuron_false_positives": int(fp_volume.sum()),
                   "neuron_false_positive_rate": float(fp_volume.sum() / max(np.count_nonzero(audit == 3), 1)),
                   "target_true_positives": int(np.count_nonzero(prediction & ((audit == 1) | (audit == 2))))}
            patch_rows.append(row)
            candidates.append((row["neuron_false_positives"], name, patch_index, center, raw, audit, probability, row))
        sample_stats[name] = finalize(current)
    selected = []
    for name in data:
        sample_candidates = sorted((item for item in candidates if item[1] == name), reverse=True)[:3]
        for _, _, patch_index, center, raw, audit, probability, row in sample_candidates:
            filename = f"{name}_patch{patch_index:02d}_z{row['representative_source_z']:03d}.png"
            save_png(assets / filename, panel(raw[center], audit[center], probability[center], threshold))
            selected.append({**row, "asset": f"assets/{filename}"})
    aggregate = finalize(total_stats)
    scored_fp = aggregate["confirmed_neuron"]["predicted_positive"]
    unscored_names = ["no_segment", "other_classified_cell", "unclassified_queried_segment", "tiny_unqueried_segment"]
    unscored_predicted = sum(aggregate[name]["predicted_positive"] for name in unscored_names)
    result = {"threshold": threshold, "checkpoint_step": int(checkpoint["step"]),
              "counting_unit": "patch observations; overlapping patches can count a source voxel more than once",
              "aggregate_by_label": aggregate, "per_sample_by_label": sample_stats,
              "scored_false_positives": scored_fp,
              "scored_false_positive_label": "confirmed_neuron only",
              "unscored_predicted_positive": unscored_predicted,
              "unscored_labels": unscored_names, "patches": patch_rows,
              "representative_examples": selected,
              "limitations": ["Cell-type categories come from MICrONS annotations, not expert morphology review.",
                              "Ignored-region predictions are unscored and are not false positives.",
                              "This audit does not identify blood vessels because no vessel label code is available."]}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    label_rows = "".join(
        f"<tr><td>{name}</td><td>{row['voxels']:,}</td><td>{row['predicted_positive']:,}</td>"
        f"<td>{(100 * row['predicted_positive_fraction'] if row['predicted_positive_fraction'] is not None else 0):.2f}%</td>"
        f"<td>{(row['mean_probability'] if row['mean_probability'] is not None else 0):.3f}</td></tr>"
        for name, row in aggregate.items())
    example_html = "".join(
        f"<article><h3>{row['sample']} · patch {row['patch_index']} · source Z={row['representative_source_z']}</h3>"
        f"<p>Neuron FP {row['neuron_false_positives']:,}; neuron FPR {100 * row['neuron_false_positive_rate']:.2f}%.</p>"
        f"<img src='{row['asset']}'></article>" for row in selected)
    html = f'''<!doctype html><meta charset="utf-8"><title>3D false-positive audit</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:30px auto;padding:16px;background:#18232b;color:#eee}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #647482}}img{{width:100%;image-rendering:pixelated}}article{{margin:28px 0}}a{{color:#8dbaff}}</style>
<h1>Frozen 3D model: false-positive audit</h1>
<p>Checkpoint step {checkpoint['step']}; validation-locked threshold {threshold:.4f}. Counts are patch observations. No retraining or test-based threshold tuning.</p>
<p><b>Key finding:</b> all {scored_fp:,} scored false positives belong to <code>confirmed_neuron</code>, because neuron is the only scored negative label. Predictions in ignored categories are reported separately as unscored predictions, not false positives.</p>
<table><tr><th>Audit label</th><th>Observed voxels</th><th>Predicted foreground</th><th>Predicted fraction</th><th>Mean probability</th></tr>{label_rows}</table>
<h2>Highest-neuron-FP examples</h2><p>Prediction is red. In the audit panel, truth is cyan, confirmed neuron is blue, and scored neuron FP is magenta.</p>{example_html}
<p><b>Limitations:</b> categories come from automated annotation lookup rather than expert morphology review. No vessel label exists, so this audit cannot determine whether an error is near a blood vessel.</p>
<p><a href="result.json">Full audit JSON</a></p>'''
    (args.out / "index.html").write_text(html)
    print(json.dumps({"scored_false_positives": scored_fp,
                      "unscored_predicted_positive": unscored_predicted,
                      "report": str(args.out / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
