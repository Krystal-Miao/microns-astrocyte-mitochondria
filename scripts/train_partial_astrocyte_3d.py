"""Small 3D U-Net pilot with cell-level split and partial labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_astro_unet import metrics  # noqa: E402
from train_foreground_aware_pilot import ranking_metrics  # noqa: E402
from train_partial_astrocyte_labels import partial_objective  # noqa: E402


def block3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
    )


def normalized_block3d(in_channels, out_channels):
    groups = min(4, out_channels)
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels), nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels), nn.LeakyReLU(negative_slope=0.01, inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.enc1 = block3d(1, width)
        self.enc2 = block3d(width, width * 2)
        self.bottom = block3d(width * 2, width * 4)
        self.dec2 = block3d(width * 6, width * 2)
        self.dec1 = block3d(width * 3, width)
        self.head = nn.Conv3d(width, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        x = self.bottom(F.max_pool3d(e2, 2))
        x = F.interpolate(x, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        return self.head(self.dec1(torch.cat([x, e1], dim=1)))


class UNet3DNNStyle(UNet3D):
    """Small nnU-Net-style variant for CPU-controlled experiments."""

    def __init__(self, width=4):
        nn.Module.__init__(self)
        self.enc1 = normalized_block3d(1, width)
        self.enc2 = normalized_block3d(width, width * 2)
        self.bottom = normalized_block3d(width * 2, width * 4)
        self.dec2 = normalized_block3d(width * 6, width * 2)
        self.dec1 = normalized_block3d(width * 3, width)
        self.head = nn.Conv3d(width, 1, 1)


def uint8_normalization_stats(image):
    """Exact 0.5/99.5 percentiles and clipped moments from a uint8 volume."""
    histogram = np.zeros(256, dtype=np.int64)
    for z in range(image.shape[0]):
        histogram += np.bincount(np.asarray(image[z]).ravel(), minlength=256)
    cumulative = np.cumsum(histogram)
    total = int(cumulative[-1])
    low = int(np.searchsorted(cumulative, total * 0.005))
    high = int(np.searchsorted(cumulative, total * 0.995))
    values = np.arange(256, dtype=np.float64)
    clipped = np.clip(values, low, high)
    mean = float((histogram * clipped).sum() / total)
    variance = float((histogram * (clipped - mean) ** 2).sum() / total)
    return {"low": low, "high": high, "mean": mean, "std": max(variance ** 0.5, 1e-6)}


def normalize_raw(raw, stats=None):
    if stats is None:
        return torch.from_numpy(raw).float() / 255
    clipped = np.clip(raw, stats["low"], stats["high"]).astype(np.float32)
    return torch.from_numpy((clipped - stats["mean"]) / stats["std"])


def find_sample(roots, name):
    matches = [(root / name, root / "label_audit_01" / name / "audit_labels.tif")
               for root in roots if (root / name / "image.tif").exists()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source for {name}, found {len(matches)}")
    return matches[0]


def load(roots, names):
    result = {}
    for name in names:
        sample, audit = find_sample(roots, name)
        image = tifffile.memmap(sample / "image.tif", mode="r")
        labels = tifffile.memmap(audit, mode="r")
        if image.shape != labels.shape:
            raise ValueError(f"Shape mismatch for {name}")
        result[name] = (image, labels)
    return result


def bounds_containing(rng, point, patch, shape):
    starts = []
    for coordinate, size, maximum in zip(point, patch, shape):
        low = max(0, int(coordinate) - size + 1)
        high = min(int(coordinate), maximum - size)
        starts.append(int(rng.integers(low, high + 1)))
    return tuple(starts)


def arrays_from_labels(audit):
    target = ((audit == 1) | (audit == 2)).astype(np.float32)
    valid = (audit == 1) | (audit == 2) | (audit == 3)
    return target, valid


def draw(data, rng, patch, foreground_centered=True):
    name = list(data)[int(rng.integers(len(data)))]
    image, labels = data[name]
    if foreground_centered:
        points = np.argwhere((labels == 1) | (labels == 2))
        point = points[int(rng.integers(len(points)))]
        z, y, x = bounds_containing(rng, point, patch, image.shape)
    else:
        z = int(rng.integers(0, image.shape[0] - patch[0] + 1))
        y = int(rng.integers(0, image.shape[1] - patch[1] + 1))
        x = int(rng.integers(0, image.shape[2] - patch[2] + 1))
    raw = np.asarray(image[z:z + patch[0], y:y + patch[1], x:x + patch[2]]).copy()
    audit = np.asarray(labels[z:z + patch[0], y:y + patch[1], x:x + patch[2]]).copy()
    target, valid = arrays_from_labels(audit)
    return raw, target, valid, {"sample": name, "z": z, "y": y, "x": x,
                                "positive_fraction_labeled": float(target[valid].mean()),
                                "valid_fraction": float(valid.mean())}


def fixed_validation_coordinates(labels, patch, seed=2026, n_positive=6, n_uniform=2):
    rng = np.random.default_rng(seed)
    points = np.argwhere((labels == 1) | (labels == 2))
    coordinates = []
    for _ in range(n_positive):
        point = points[int(rng.integers(len(points)))]
        coordinates.append(bounds_containing(rng, point, patch, labels.shape))
    for _ in range(n_uniform):
        coordinates.append(tuple(int(rng.integers(0, maximum - size + 1))
                                 for maximum, size in zip(labels.shape, patch)))
    return coordinates


@torch.no_grad()
def validate(model, data, coordinates, patch, device, normalization=None):
    model.eval()
    probabilities, truths = [], []
    tp = fp = fn = 0
    for name, (image, labels) in data.items():
        for z, y, x in coordinates[name]:
            raw = np.asarray(image[z:z + patch[0], y:y + patch[1], x:x + patch[2]]).copy()
            audit = np.asarray(labels[z:z + patch[0], y:y + patch[1], x:x + patch[2]]).copy()
            target, valid = arrays_from_labels(audit)
            stats = None if normalization is None else normalization[name]
            tensor = normalize_raw(raw, stats)[None, None].to(device)
            probability = model(tensor).sigmoid()[0, 0].cpu().numpy()
            prediction = probability >= 0.5
            truth = target.astype(bool)
            tp += int(np.count_nonzero(prediction & truth & valid))
            fp += int(np.count_nonzero(prediction & ~truth & valid))
            fn += int(np.count_nonzero(~prediction & truth & valid))
            probabilities.append(probability[valid]); truths.append(truth[valid])
    probability = np.concatenate(probabilities); truth = np.concatenate(truths)
    score = metrics(tp, fp, fn)
    score.update(evaluated_pixels=int(len(truth)), labeled_foreground_fraction=float(truth.mean()),
                 predicted_foreground_fraction=float((probability >= 0.5).mean()),
                 foreground_probability_mean=float(probability[truth].mean()),
                 background_probability_mean=float(probability[~truth].mean()))
    score.update(ranking_metrics(probability, truth))
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--patch-zyx", type=int, nargs=3, default=[16, 128, 128])
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--train-samples", nargs="+", default=[f"sample_{i:02d}" for i in range(1, 9)])
    parser.add_argument("--validation-samples", nargs="+", default=["sample_09", "sample_10"])
    parser.add_argument("--test-samples", nargs="+", default=["sample_11", "sample_12"])
    parser.add_argument("--smoke", action="store_true", help="Repeatedly fit and evaluate one fixed training patch")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    patch = tuple(args.patch_zyx)
    train = load(args.roots, args.train_samples)
    validation = load(args.roots, args.validation_samples)
    test = load(args.roots, args.test_samples)
    fixed_draw = draw(train, rng, patch, foreground_centered=True) if args.smoke else None
    if args.smoke:
        fixed_name = fixed_draw[3]["sample"]
        validation = {fixed_name: train[fixed_name]}
        coordinates = {fixed_name: [(fixed_draw[3]["z"], fixed_draw[3]["y"], fixed_draw[3]["x"])]}
    else:
        coordinates = {name: fixed_validation_coordinates(labels, patch, seed=2026 + i)
                       for i, (name, (_, labels)) in enumerate(validation.items())}
    model = UNet3D(args.width).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    history, trace = [], []
    best_key = (-1.0, float("-inf")); start = time.time()
    for step in range(1, args.steps + 1):
        raw, target, valid, record = (fixed_draw if args.smoke else
                                      draw(train, rng, patch, foreground_centered=step % 4 != 0))
        trace.append({"step": step, **record})
        x = torch.from_numpy(raw)[None, None].float().to(args.device) / 255
        y = torch.from_numpy(target)[None, None].float().to(args.device)
        v = torch.from_numpy(valid)[None, None].to(args.device)
        model.train(); optimizer.zero_grad()
        logits = model(x); loss = partial_objective(logits, y, v)
        if not torch.isfinite(loss): raise ValueError("Nonfinite loss")
        loss.backward(); optimizer.step()
        if step % 20 == 0 or step == args.steps:
            score = validate(model, validation, coordinates, patch, args.device)
            row = {"step": step, "train_loss": float(loss.detach().cpu()),
                   "elapsed_seconds": time.time() - start, **score}
            history.append(row)
            key = (score["average_precision"], score["best_threshold_dice"])
            if key > best_key:
                best_key = key
                torch.save({"model": model.state_dict(), "width": args.width, "step": step,
                            "validation": score}, args.out / "best.pt")
            print(f"step {step}/{args.steps} loss={row['train_loss']:.4f} AP={score['average_precision']:.4f} "
                  f"best_dice={score['best_threshold_dice']:.4f} dice05={score['dice']:.4f} "
                  f"Pfg={score['foreground_probability_mean']:.3f} Pbg={score['background_probability_mean']:.3f} "
                  f"elapsed={row['elapsed_seconds']:.0f}s", flush=True)
    best = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    test_coordinates = {name: fixed_validation_coordinates(labels, patch, seed=3026 + i)
                        for i, (name, (_, labels)) in enumerate(test.items())}
    test_score = None if args.smoke else validate(model, test, test_coordinates, patch, args.device)
    result = {"best_step": best["step"], "best_validation": best["validation"],
              "last_validation": history[-1], "test": test_score,
              "elapsed_seconds": time.time() - start,
              "scope": ("3D fixed-patch memorization sanity check; not generalization evidence" if args.smoke else
                        "3D partial-label volume split; eight train, two validation, two untouched test astrocytes")}
    config = {"seed": args.seed, "steps": args.steps, "device": args.device, "width": args.width,
              "patch_zyx": list(patch), "learning_rate": 0.0003, "batch_size": 1,
              "positive_codes": [1, 2], "negative_codes": [3], "ignored_codes": [0, 4, 5, 6],
              "roots": [str(root) for root in args.roots],
              "train_samples": args.train_samples,
              "validation_samples": [fixed_draw[3]["sample"]] if args.smoke else args.validation_samples,
              "test_samples": [] if args.smoke else args.test_samples,
              "smoke": args.smoke,
              "validation_coordinates_zyx": coordinates,
              "test_coordinates_zyx": {} if args.smoke else test_coordinates}
    for name, value in (("config.json", config), ("history.json", history),
                        ("sampling.json", trace), ("result.json", result)):
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
