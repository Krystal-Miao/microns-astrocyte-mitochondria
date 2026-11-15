"""Memorize one fixed 3D patch from each training cell as a controlled diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_partial_astrocyte_3d import (  # noqa: E402
    UNet3D,
    UNet3DNNStyle,
    draw,
    load,
    normalize_raw,
    partial_objective,
    uint8_normalization_stats,
    validate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--patch-zyx", type=int, nargs=3, default=[16, 128, 128])
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--architecture", choices=["plain", "nnstyle"], default="plain")
    parser.add_argument("--normalization", choices=["divide255", "percentile-zscore"], default="divide255")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--train-samples", nargs="+", default=[f"sample_{i:02d}" for i in range(1, 9)])
    parser.add_argument("--resume-run", type=Path,
                        help="Continue from another run's best.pt and append its history")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    patch = tuple(args.patch_zyx)
    data = load(args.roots, args.train_samples)
    fixed = []
    coordinates = {}
    for name in args.train_samples:
        row = draw({name: data[name]}, rng, patch, foreground_centered=True)
        fixed.append(row)
        coordinates[name] = [(row[3]["z"], row[3]["y"], row[3]["x"])]
    normalizers = ({name: uint8_normalization_stats(image)
                    for name, (image, _) in data.items()}
                   if args.normalization == "percentile-zscore" else None)
    model_class = UNet3DNNStyle if args.architecture == "nnstyle" else UNet3D
    model = model_class(args.width).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=0.00003) if args.lr_schedule == "cosine" else None)
    history = []
    initial_step = 0
    best_key = (-1.0, float("-inf"))
    if args.resume_run:
        checkpoint = torch.load(args.resume_run / "best.pt", map_location=args.device, weights_only=True)
        if checkpoint["width"] != args.width:
            raise ValueError("Resume checkpoint width does not match --width")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        initial_step = int(checkpoint["step"])
        history = json.loads((args.resume_run / "history.json").read_text())
        old = checkpoint["evaluation"]
        best_key = (old["average_precision"], old["best_threshold_dice"])
        torch.save({"model": model.state_dict(), "width": args.width,
                    "step": initial_step, "evaluation": old,
                    "optimizer": optimizer.state_dict()}, args.out / "best.pt")
        if args.steps <= initial_step:
            raise ValueError("--steps must exceed the resumed checkpoint step")
    start = time.time()
    for step in range(initial_step + 1, args.steps + 1):
        raw, target, valid, _ = fixed[(step - 1) % len(fixed)]
        stats = None if normalizers is None else normalizers[fixed[(step - 1) % len(fixed)][3]["sample"]]
        x = normalize_raw(raw, stats)[None, None].to(args.device)
        y = torch.from_numpy(target)[None, None].float().to(args.device)
        v = torch.from_numpy(valid)[None, None].to(args.device)
        model.train(); optimizer.zero_grad()
        loss = partial_objective(model(x), y, v)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward(); optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step % 50 == 0 or step == args.steps:
            score = validate(model, data, coordinates, patch, args.device, normalizers)
            row = {"step": step, "train_loss": float(loss.detach().cpu()),
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "elapsed_seconds": time.time() - start, **score}
            history.append(row)
            key = (score["average_precision"], score["best_threshold_dice"])
            if key > best_key:
                best_key = key
                torch.save({"model": model.state_dict(), "width": args.width,
                            "step": step, "evaluation": score,
                            "optimizer": optimizer.state_dict(),
                            "architecture": args.architecture}, args.out / "best.pt")
            print(f"step {step}/{args.steps} loss={row['train_loss']:.4f} "
                  f"AP={score['average_precision']:.4f} "
                  f"bestDice={score['best_threshold_dice']:.4f} "
                  f"dice05={score['dice']:.4f} "
                  f"Pfg={score['foreground_probability_mean']:.3f} "
                  f"Pbg={score['background_probability_mean']:.3f} "
                  f"elapsed={row['elapsed_seconds']:.0f}s", flush=True)
    best = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)
    prevalence = best["evaluation"]["labeled_foreground_fraction"]
    result = {"best_step": best["step"], "best_evaluation": best["evaluation"],
              "random_ap": prevalence, "all_positive_dice": 2 * prevalence / (1 + prevalence),
              "elapsed_seconds": time.time() - start,
              "scope": "eight fixed training patches; memorization diagnostic only"}
    config = {"roots": [str(root) for root in args.roots], "train_samples": args.train_samples,
              "steps": args.steps, "seed": args.seed, "device": args.device,
              "width": args.width, "patch_zyx": list(patch), "learning_rate": 0.0003,
              "architecture": args.architecture, "normalization": args.normalization,
              "normalization_stats": normalizers, "lr_schedule": args.lr_schedule,
              "resume_run": str(args.resume_run) if args.resume_run else None,
              "initial_step": initial_step,
              "coordinates_zyx": coordinates,
              "positive_codes": [1, 2], "negative_codes": [3], "ignored_codes": [0, 4, 5, 6]}
    for name, value in (("config.json", config), ("history.json", history), ("result.json", result)):
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
