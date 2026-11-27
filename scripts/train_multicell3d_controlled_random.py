"""Train the lightweight 3D baseline with balanced on-the-fly random sampling."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_multicell3d_patch_pool import (  # noqa: E402
    augment,
    build_pool,
    centered_start,
    evaluate_pool,
    extract_patch,
)
from train_partial_astrocyte_3d import (  # noqa: E402
    UNet3DNNStyle,
    fixed_validation_coordinates,
    load,
    normalize_raw,
    partial_objective,
    uint8_normalization_stats,
    validate,
)


def draw_random(data, rng, patch, kind):
    name = list(data)[int(rng.integers(len(data)))]
    image, labels = data[name]
    if kind == "foreground":
        start = centered_start(rng, labels, patch, (1, 2))
    elif kind == "neuron_hard_negative":
        start = centered_start(rng, labels, patch, (3,))
    elif kind == "uniform":
        start = tuple(int(rng.integers(0, maximum - size + 1))
                      for maximum, size in zip(labels.shape, patch))
    else:
        raise ValueError(f"Unknown sampling kind: {kind}")
    return extract_patch(image, labels, start, patch, name, kind)


PREPROCESSING = {
    "raw": None,
    "clahe": {"kernel_size": (64, 64), "clip_limit": 0.01, "nbins": 256},
    "nlm": {"h": 2 / 255, "patch_size": 5, "patch_distance": 6, "fast_mode": True,
            "channel_axis": None, "sigma": 0.0, "preserve_range": True},
}


def preprocess_images(data, method):
    """Transform full XY planes before cropping while preserving label alignment."""
    if method == "raw":
        return data
    if method == "clahe":
        from skimage.exposure import equalize_adapthist as transform
    elif method == "nlm":
        from skimage.restoration import denoise_nl_means as transform
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
    result = {}
    for name, (image, labels) in data.items():
        transformed = np.empty(image.shape, dtype=np.uint8)
        for z in range(image.shape[0]):
            plane = np.asarray(image[z], dtype=np.float32) / 255
            processed = transform(plane, **PREPROCESSING[method])
            transformed[z] = np.clip(np.rint(processed * 255), 0, 255).astype(np.uint8)
        result[name] = (transformed, labels)
        print(f"preprocessed {name} with full-slice {method}", flush=True)
    return result


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
    parser.add_argument("--preprocessing", choices=PREPROCESSING, default="raw")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed); patch = tuple(args.patch_zyx)
    train_names = [f"sample_{i:02d}" for i in range(1, 9)]
    validation_names = ["sample_09", "sample_10"]
    train = load(args.roots, train_names); validation = load(args.roots, validation_names)
    train = preprocess_images(train, args.preprocessing)
    validation = preprocess_images(validation, args.preprocessing)
    normalizers = {name: uint8_normalization_stats(image)
                   for name, (image, _) in {**train, **validation}.items()}
    diagnostic_rng = np.random.default_rng(8042)
    train_diagnostic = build_pool(train, diagnostic_rng, patch, 1, 1, 1)
    validation_coordinates = {name: fixed_validation_coordinates(labels, patch, seed=7026 + i)
                              for i, (name, (_, labels)) in enumerate(validation.items())}
    model = UNet3DNNStyle(args.width).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=0.00003)
    kinds = np.array(["foreground", "uniform", "neuron_hard_negative"], dtype=object)
    kind_order = kinds.copy(); cursor = len(kind_order)
    history = []; sampling = []; best_key = (-1.0, float("-inf")); start = time.time()
    for step in range(1, args.steps + 1):
        if cursor == len(kind_order):
            rng.shuffle(kind_order); cursor = 0
        kind = str(kind_order[cursor]); cursor += 1
        raw, target, valid, record = draw_random(train, rng, patch, kind)
        sampling.append({"step": step, **record})
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
            train_score = evaluate_pool(model, train_diagnostic, normalizers, args.device)
            val_score = validate(model, validation, validation_coordinates, patch,
                                 args.device, normalizers)
            row = {"step": step, "train_loss": float(loss.detach().cpu()),
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "elapsed_seconds": time.time() - start,
                   "training_diagnostic": train_score, "validation": val_score}
            history.append(row)
            key = (val_score["average_precision"], val_score["best_threshold_dice"])
            if key > best_key:
                best_key = key
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "width": args.width, "step": step, "validation": val_score,
                            "training_diagnostic": train_score}, args.out / "best.pt")
            print(f"step {step}/{args.steps} loss={row['train_loss']:.4f} "
                  f"trainAP={train_score['average_precision']:.4f} "
                  f"trainDice={train_score['best_threshold_dice']:.4f} "
                  f"valAP={val_score['average_precision']:.4f} "
                  f"valDice={val_score['best_threshold_dice']:.4f} "
                  f"valSep={val_score['foreground_probability_mean'] - val_score['background_probability_mean']:+.3f} "
                  f"elapsed={row['elapsed_seconds']:.0f}s", flush=True)
    best = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)
    counts = {kind: sum(row["kind"] == kind for row in sampling) for kind in kinds.tolist()}
    result = {"best_step": best["step"], "best_validation": best["validation"],
              "best_training_diagnostic": best["training_diagnostic"],
              "last": history[-1], "sampling_counts": counts,
              "elapsed_seconds": time.time() - start,
              "scope": "controlled on-the-fly random sampling; sample_09/10 validation; sample_11/12 untouched"}
    config = {"roots": [str(root) for root in args.roots], "seed": args.seed,
              "steps": args.steps, "patch_zyx": list(patch), "width": args.width,
              "train_samples": train_names, "validation_samples": validation_names,
              "untouched_test_samples": ["sample_11", "sample_12"],
              "sampling": "balanced shuffled cycles: foreground, uniform, neuron_hard_negative",
              "preprocessing_variant": args.preprocessing,
              "preprocessing_parameters": (None if args.preprocessing == "raw" else
                  {"method": {"clahe": "skimage.exposure.equalize_adapthist",
                              "nlm": "skimage.restoration.denoise_nl_means"}[args.preprocessing],
                   "scope": "full XY slice before crop",
                   "input": "float32 uint8/255", "output": "rounded uint8",
                   **PREPROCESSING[args.preprocessing]}),
              "augmentation": ["xy_rot90", "x_flip", "y_flip"],
              "normalization": "per-volume 0.5/99.5 percentile clipping and z-score",
              "architecture": "GroupNorm + LeakyReLU lightweight 3D U-Net",
              "lr_schedule": "Adam 3e-4 to 3e-5 cosine", "eval_every": args.eval_every,
              "validation_coordinates_zyx": validation_coordinates,
              "training_diagnostic_pool": [record for *_, record in train_diagnostic]}
    for name, value in (("config.json", config), ("history.json", history),
                        ("sampling.json", sampling), ("result.json", result)):
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
