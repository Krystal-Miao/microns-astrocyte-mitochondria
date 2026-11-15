"""Train the lightweight 3D baseline on a reproducible 64-patch pool."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_astro_unet import metrics  # noqa: E402
from train_foreground_aware_pilot import ranking_metrics  # noqa: E402
from train_partial_astrocyte_3d import (  # noqa: E402
    UNet3DNNStyle,
    arrays_from_labels,
    bounds_containing,
    fixed_validation_coordinates,
    load,
    normalize_raw,
    partial_objective,
    uint8_normalization_stats,
    validate,
)


def extract_patch(image, labels, start, patch, sample, kind):
    z, y, x = start
    raw = np.asarray(image[z:z + patch[0], y:y + patch[1], x:x + patch[2]]).copy()
    audit = np.asarray(labels[z:z + patch[0], y:y + patch[1], x:x + patch[2]]).copy()
    target, valid = arrays_from_labels(audit)
    return raw, target, valid, {"sample": sample, "kind": kind, "z": z, "y": y, "x": x,
                                "valid_fraction": float(valid.mean()),
                                "positive_fraction_labeled": float(target[valid].mean())}


def centered_start(rng, labels, patch, codes):
    points = np.argwhere(np.isin(labels, codes))
    if not len(points):
        raise ValueError(f"No voxels with codes {codes}")
    return bounds_containing(rng, points[int(rng.integers(len(points)))], patch, labels.shape)


def build_pool(data, rng, patch, foreground_per_cell, uniform_per_cell,
               neuron_per_cell):
    pool = []
    for name, (image, labels) in data.items():
        for _ in range(foreground_per_cell):
            start = centered_start(rng, labels, patch, (1, 2))
            pool.append(extract_patch(image, labels, start, patch, name, "foreground"))
        for _ in range(uniform_per_cell):
            start = tuple(int(rng.integers(0, maximum - size + 1))
                          for maximum, size in zip(labels.shape, patch))
            pool.append(extract_patch(image, labels, start, patch, name, "uniform"))
        for _ in range(neuron_per_cell):
            start = centered_start(rng, labels, patch, (3,))
            pool.append(extract_patch(image, labels, start, patch, name, "neuron_hard_negative"))
    return pool


def augment(raw, target, valid, rng):
    k = int(rng.integers(4))
    raw, target, valid = (np.rot90(a, k, axes=(1, 2)) for a in (raw, target, valid))
    if rng.random() < 0.5:
        raw, target, valid = (np.flip(a, axis=1) for a in (raw, target, valid))
    if rng.random() < 0.5:
        raw, target, valid = (np.flip(a, axis=2) for a in (raw, target, valid))
    return tuple(np.ascontiguousarray(a) for a in (raw, target, valid))


@torch.no_grad()
def evaluate_pool(model, pool, normalizers, device):
    model.eval()
    probabilities, truths = [], []
    tp = fp = fn = 0
    for raw, target, valid, record in pool:
        probability = model(normalize_raw(raw, normalizers[record["sample"]])
                            [None, None].to(device)).sigmoid()[0, 0].cpu().numpy()
        truth = target.astype(bool); prediction = probability >= 0.5
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
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--patch-zyx", type=int, nargs=3, default=[16, 128, 128])
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--foreground-per-cell", type=int, default=3)
    parser.add_argument("--uniform-per-cell", type=int, default=3)
    parser.add_argument("--neuron-per-cell", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed); patch = tuple(args.patch_zyx)
    train_names = [f"sample_{i:02d}" for i in range(1, 9)]
    validation_names = ["sample_09", "sample_10"]
    train = load(args.roots, train_names); validation = load(args.roots, validation_names)
    normalizers = {name: uint8_normalization_stats(image)
                   for name, (image, _) in {**train, **validation}.items()}
    pool = build_pool(train, rng, patch, args.foreground_per_cell,
                      args.uniform_per_cell, args.neuron_per_cell)
    validation_coordinates = {name: fixed_validation_coordinates(labels, patch, seed=7026 + i)
                              for i, (name, (_, labels)) in enumerate(validation.items())}
    model = UNet3DNNStyle(args.width).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=0.00003)
    order = np.arange(len(pool)); cursor = len(pool)
    history = []; best_key = (-1.0, float("-inf")); start = time.time()
    for step in range(1, args.steps + 1):
        if cursor == len(pool):
            rng.shuffle(order); cursor = 0
        raw, target, valid, record = pool[int(order[cursor])]; cursor += 1
        raw, target, valid = augment(raw, target, valid, rng)
        x = normalize_raw(raw, normalizers[record["sample"]])[None, None].to(args.device)
        y = torch.from_numpy(target)[None, None].float().to(args.device)
        v = torch.from_numpy(valid)[None, None].to(args.device)
        model.train(); optimizer.zero_grad()
        loss = partial_objective(model(x), y, v)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward(); optimizer.step(); scheduler.step()
        if step % args.eval_every == 0 or step == args.steps:
            train_score = evaluate_pool(model, pool, normalizers, args.device)
            val_score = validate(model, validation, validation_coordinates, patch,
                                 args.device, normalizers)
            row = {"step": step, "train_loss": float(loss.detach().cpu()),
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "elapsed_seconds": time.time() - start,
                   "training_pool": train_score, "validation": val_score}
            history.append(row)
            key = (val_score["average_precision"], val_score["best_threshold_dice"])
            if key > best_key:
                best_key = key
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "width": args.width, "step": step, "validation": val_score,
                            "training_pool": train_score}, args.out / "best.pt")
            print(f"step {step}/{args.steps} loss={row['train_loss']:.4f} "
                  f"trainAP={train_score['average_precision']:.4f} "
                  f"trainDice={train_score['best_threshold_dice']:.4f} "
                  f"valAP={val_score['average_precision']:.4f} "
                  f"valDice={val_score['best_threshold_dice']:.4f} "
                  f"valSep={val_score['foreground_probability_mean'] - val_score['background_probability_mean']:+.3f} "
                  f"elapsed={row['elapsed_seconds']:.0f}s", flush=True)
    best = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)
    result = {"best_step": best["step"], "best_validation": best["validation"],
              "best_training_pool": best["training_pool"], "last": history[-1],
              "elapsed_seconds": time.time() - start,
              "scope": (f"{len(pool)}-patch training pool; sample_09/10 validation; "
                        "sample_11/12 untouched")}
    config = {"roots": [str(root) for root in args.roots], "seed": args.seed,
              "steps": args.steps, "patch_zyx": list(patch), "width": args.width,
              "train_samples": train_names, "validation_samples": validation_names,
              "untouched_test_samples": ["sample_11", "sample_12"],
              "pool_composition_per_cell": {"foreground": args.foreground_per_cell,
                                             "uniform": args.uniform_per_cell,
                                             "neuron_hard_negative": args.neuron_per_cell},
              "augmentation": ["xy_rot90", "x_flip", "y_flip"],
              "normalization": "per-volume 0.5/99.5 percentile clipping and z-score",
              "architecture": "GroupNorm + LeakyReLU lightweight 3D U-Net",
              "lr_schedule": "Adam 3e-4 to 3e-5 cosine", "eval_every": args.eval_every,
              "validation_coordinates_zyx": validation_coordinates,
              "pool": [record for *_, record in pool]}
    for name, value in (("config.json", config), ("history.json", history),
                        ("result.json", result)):
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
