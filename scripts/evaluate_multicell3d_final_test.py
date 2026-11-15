"""Evaluate a frozen 3D checkpoint once on untouched test cells."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_foreground_aware_pilot import ranking_metrics  # noqa: E402
from train_partial_astrocyte_3d import (  # noqa: E402
    UNet3DNNStyle,
    arrays_from_labels,
    fixed_validation_coordinates,
    load,
    normalize_raw,
    uint8_normalization_stats,
)


def threshold_metrics(probability, truth, threshold):
    prediction = probability >= threshold
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "dice": dice, "iou": iou,
            "predicted_foreground_fraction": float(prediction.mean())}


@torch.no_grad()
def predict(model, data, coordinates, patch, normalizers, device):
    model.eval()
    per_sample = {}
    all_probabilities, all_truths = [], []
    for name, (image, labels) in data.items():
        probabilities, truths = [], []
        for z, y, x in coordinates[name]:
            raw = np.asarray(image[z:z + patch[0], y:y + patch[1], x:x + patch[2]]).copy()
            audit = np.asarray(labels[z:z + patch[0], y:y + patch[1], x:x + patch[2]]).copy()
            target, valid = arrays_from_labels(audit)
            tensor = normalize_raw(raw, normalizers[name])[None, None].to(device)
            probability = model(tensor).sigmoid()[0, 0].cpu().numpy()
            probabilities.append(probability[valid])
            truths.append(target.astype(bool)[valid])
        sample_probability = np.concatenate(probabilities)
        sample_truth = np.concatenate(truths)
        per_sample[name] = (sample_probability, sample_truth)
        all_probabilities.append(sample_probability)
        all_truths.append(sample_truth)
    return per_sample, np.concatenate(all_probabilities), np.concatenate(all_truths)


def summarize(probability, truth, locked_threshold):
    result = threshold_metrics(probability, truth, locked_threshold)
    result.update(ranking_metrics(probability, truth))
    result.update(evaluated_pixels=int(len(truth)),
                  labeled_foreground_fraction=float(truth.mean()),
                  foreground_probability_mean=float(probability[truth].mean()),
                  background_probability_mean=float(probability[~truth].mean()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = json.loads((args.run / "config.json").read_text())
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    test_names = config["untouched_test_samples"]
    patch = tuple(config["patch_zyx"])
    locked_threshold = float(checkpoint["validation"]["best_threshold"])
    data = load(args.roots, test_names)
    normalizers = {name: uint8_normalization_stats(image)
                   for name, (image, _) in data.items()}
    coordinates = {name: fixed_validation_coordinates(labels, patch, seed=3026 + i)
                   for i, (name, (_, labels)) in enumerate(data.items())}
    model = UNet3DNNStyle(checkpoint["width"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    start = time.time()
    per_sample_arrays, probability, truth = predict(
        model, data, coordinates, patch, normalizers, args.device)
    per_sample = {name: summarize(p, t, locked_threshold)
                  for name, (p, t) in per_sample_arrays.items()}
    aggregate = summarize(probability, truth, locked_threshold)
    result = {"checkpoint_step": int(checkpoint["step"]),
              "locked_validation_threshold": locked_threshold,
              "test_samples": test_names, "aggregate": aggregate,
              "per_sample": per_sample, "elapsed_seconds": time.time() - start,
              "scope": "One-shot final test; checkpoint and threshold frozen from sample_09/10 validation"}
    protocol = {"run": str(args.run), "checkpoint": str(args.run / "best.pt"),
                "roots": [str(root) for root in args.roots], "patch_zyx": list(patch),
                "coordinates_zyx": coordinates, "coordinate_seeds": [3026, 3027],
                "patches_per_sample": 8,
                "primary_metrics": ["average_precision", "locked-threshold Dice", "locked-threshold IoU"],
                "note": "best_threshold_dice is an oracle diagnostic only and was not used to tune the model"}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
