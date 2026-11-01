"""500-step sample_01 pilot with stratified online 256x256 patch sampling."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_astro_unet import save_overlay  # noqa: E402
from evaluate_fixed_patch_checkpoints import predict_tiled  # noqa: E402
from train_astro_unet import UNet, digest, loss_fn, metrics  # noqa: E402


def objective(logits, target, pos_weight):
    """BCE (optionally foreground-weighted) plus the existing soft Dice loss."""
    probability = logits.sigmoid()
    soft_dice = (
        (2 * (probability * target).sum((1, 2, 3)) + 1)
        / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    )
    weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    return bce + (1 - soft_dice).mean()


def target_boundary(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner 4-connected boundary of a binary mask."""
    m = mask.astype(bool)
    eroded = m.copy()
    eroded[1:] &= m[:-1]
    eroded[:-1] &= m[1:]
    eroded[:, 1:] &= m[:, :-1]
    eroded[:, :-1] &= m[:, 1:]
    return m & ~eroded


def top_left_containing(rng, y, x, size, height, width):
    y_min, y_max = max(0, y - size + 1), min(y, height - size)
    x_min, x_max = max(0, x - size + 1), min(x, width - size)
    return int(rng.integers(y_min, y_max + 1)), int(rng.integers(x_min, x_max + 1))


def make_sampler(mask, train_z, patch):
    foreground = {}
    boundary = {}
    for z in train_z:
        foreground[z] = np.argwhere(mask[z] != 0)
        boundary[z] = np.argwhere(target_boundary(mask[z]))
    fg_z = [z for z in train_z if len(foreground[z])]
    boundary_z = [z for z in train_z if len(boundary[z])]
    if not fg_z or not boundary_z:
        raise ValueError("No foreground/boundary coordinates available")

    def draw(rng, category):
        if category == "uniform_random":
            z = int(train_z[int(rng.integers(len(train_z)))])
            y = int(rng.integers(0, mask.shape[1] - patch + 1))
            x = int(rng.integers(0, mask.shape[2] - patch + 1))
            anchor = None
        else:
            table = foreground if category == "foreground_centered" else boundary
            eligible_z = fg_z if category == "foreground_centered" else boundary_z
            z = int(eligible_z[int(rng.integers(len(eligible_z)))])
            anchor_y, anchor_x = table[z][int(rng.integers(len(table[z])))]
            y, x = top_left_containing(rng, int(anchor_y), int(anchor_x), patch, mask.shape[1], mask.shape[2])
            anchor = [int(anchor_y), int(anchor_x)]
        return z, y, x, anchor

    return draw


def batch_metrics(logits, target):
    probability = logits.sigmoid()
    prediction = probability >= 0.5
    truth = target.bool()
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    result = metrics(tp, fp, fn)
    result.update(
        predicted_foreground_fraction=float(prediction.float().mean()),
        labeled_foreground_fraction=float(target.mean()),
        foreground_probability_mean=float(probability[truth].mean()),
        background_probability_mean=float(probability[~truth].mean()),
    )
    return result


@torch.no_grad()
def validate(model, image, mask, validation_z, pos_weight):
    model.eval()
    tp = fp = fn = predicted = total = 0
    losses = []
    foreground_probabilities = []
    background_probabilities = []
    probabilities = {}
    for z in validation_z:
        probability = predict_tiled(model, np.asarray(image[z]), size=256, stride=128)
        probabilities[z] = probability
        truth = np.asarray(mask[z]).astype(bool)
        prediction = probability >= 0.5
        tp += int(np.count_nonzero(prediction & truth))
        fp += int(np.count_nonzero(prediction & ~truth))
        fn += int(np.count_nonzero(~prediction & truth))
        predicted += int(prediction.sum())
        total += prediction.size
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        logits = torch.from_numpy(np.log(clipped / (1 - clipped)))[None, None].float()
        target = torch.from_numpy(np.asarray(mask[z]).copy())[None, None].float()
        losses.append(float(objective(logits, target, pos_weight)))
        foreground_probabilities.append(probability[truth])
        background_probabilities.append(probability[~truth])
    result = metrics(tp, fp, fn)
    result.update(
        loss=float(np.mean(losses)),
        predicted_foreground_fraction=predicted / total,
        labeled_foreground_fraction=(tp + fn) / total,
        foreground_probability_mean=float(np.concatenate(foreground_probabilities).mean()),
        background_probability_mean=float(np.concatenate(background_probabilities).mean()),
        evaluated_slices=len(validation_z),
    )
    return result, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("samples/astro_864691136057675736_run1/sample_01"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--pos-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.pos_weight) or args.pos_weight <= 0:
        parser.error("steps and pos-weight must be positive and finite")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    source_hashes = {name: digest(args.data / name) for name in ("image.tif", "mask.tif", "metadata.json")}
    image = tifffile.memmap(args.data / "image.tif", mode="r")
    mask = tifffile.memmap(args.data / "mask.tif", mode="r")
    if image.shape != (128, 512, 512) or mask.shape != image.shape:
        raise ValueError("Expected aligned 128x512x512 image and mask")
    valid_z = np.flatnonzero(np.ptp(image, axis=(1, 2)) > 0).tolist()
    train_z = [z for z in valid_z if z < 96]
    validation_z = [112, 117, 122, 127]
    if not all(z in valid_z for z in validation_z):
        raise ValueError("A fixed validation slice is constant or absent")
    draw = make_sampler(mask, train_z, 256)

    config = {
        "status": "development_sampling_pilot",
        "sample": "sample_01", "seed": args.seed, "steps": args.steps,
        "device": args.device, "width": 16, "patch": 256, "batch_size": 2,
        "learning_rate": 0.0003, "augmentation": False,
        "loss": "BCEWithLogits + soft Dice", "pos_weight": args.pos_weight, "threshold": 0.5,
        "sampling": {"foreground_centered": 0.5, "boundary_centered": 0.25, "uniform_random": 0.25},
        "schedule": "each batch contains one foreground-centered draw; second draw alternates boundary-centered and uniform-random",
        "train_z": train_z, "guard_z": list(range(96, 112)), "validation_z": validation_z,
        "validation_inference": "nine overlapping 256x256 tiles, stride 128, mean probability in overlaps",
        "checkpoint_policy": "highest validation Dice; ties resolved by lower validation loss",
        "source_sha256": source_hashes,
        "script_sha256": digest(Path(__file__)),
        "model_script_sha256": digest(Path(__file__).with_name("train_astro_unet.py")),
        "scope": "sample_01 within-volume development pilot; not independent test evidence",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    model = UNet(16).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    history = []
    trace_rows = []
    best_key = (-1.0, float("-inf"))
    start = time.time()
    sampling_counts = Counter()

    for step in range(1, args.steps + 1):
        categories = ["foreground_centered", "boundary_centered" if step % 2 else "uniform_random"]
        images, masks = [], []
        for batch_index, category in enumerate(categories):
            z, y, x, anchor = draw(rng, category)
            patch_image = np.asarray(image[z, y:y + 256, x:x + 256]).copy()
            patch_mask = np.asarray(mask[z, y:y + 256, x:x + 256]).copy()
            images.append(patch_image)
            masks.append(patch_mask)
            sampling_counts[category] += 1
            trace_rows.append({
                "step": step, "batch_index": batch_index, "category": category,
                "z": z, "y": y, "x": x, "anchor_y": anchor[0] if anchor else None,
                "anchor_x": anchor[1] if anchor else None,
                "foreground_fraction": float(patch_mask.mean()),
            })
        x_tensor = torch.from_numpy(np.stack(images)[:, None]).float().to(args.device) / 255
        y_tensor = torch.from_numpy(np.stack(masks)[:, None]).float().to(args.device)
        model.train()
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = objective(logits, y_tensor, args.pos_weight)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        batch_score = batch_metrics(logits.detach(), y_tensor)
        loss.backward()
        gradient_l2 = float(torch.sqrt(sum(
            parameter.grad.square().sum()
            for parameter in model.parameters() if parameter.grad is not None
        )))
        head_before = model.head.weight.detach().clone()
        optimizer.step()
        head_update_max = float((model.head.weight.detach() - head_before).abs().max())
        row = {"step": step, "batch_loss": float(loss.detach()),
               "gradient_l2": gradient_l2, "head_update_max": head_update_max,
               **{f"batch_{k}": v for k, v in batch_score.items()}}

        if step % 50 == 0 or step == args.steps:
            validation, probabilities = validate(model, image, mask, validation_z, args.pos_weight)
            row["elapsed_seconds"] = time.time() - start
            row.update({f"validation_{k}": v for k, v in validation.items()})
            history.append(row)
            key = (validation["dice"], -validation["loss"])
            if key > best_key:
                best_key = key
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "numpy_rng_state": rng.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
                            "width": 16, "step": step, "validation": validation}, args.out / "best.pt")
            snapshot = args.out / f"step_{step:04d}"
            snapshot.mkdir()
            for z, probability in probabilities.items():
                save_overlay(snapshot / f"z{z:03d}_prediction.png", np.asarray(image[z]), probability >= 0.5)
            (args.out / "history.json").write_text(json.dumps(history, indent=2))
            with (args.out / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            print(
                f"step {step}/{args.steps} batch_loss={row['batch_loss']:.4f} "
                f"batch_dice={row['batch_dice']:.4f} val_loss={validation['loss']:.4f} "
                f"val_dice={validation['dice']:.4f} val_iou={validation['iou']:.4f} "
                f"val_pred_fg={validation['predicted_foreground_fraction']:.2%} "
                f"P(fg|label_fg)={validation['foreground_probability_mean']:.3f} "
                f"P(fg|label_bg)={validation['background_probability_mean']:.3f} "
                f"grad={gradient_l2:.3g} head_update={head_update_max:.3g} "
                f"elapsed={row['elapsed_seconds']:.0f}s", flush=True,
            )

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "numpy_rng_state": rng.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
                "width": 16, "step": args.steps, "validation": validation}, args.out / "last.pt")
    with (args.out / "sampling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)
    result = {
        "best_step": int(torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)["step"]),
        "best_validation": torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)["validation"],
        "last_validation": validation,
        "sampling_counts": dict(sampling_counts),
        "sampling_foreground_fraction": {
            category: float(np.mean([row["foreground_fraction"] for row in trace_rows if row["category"] == category]))
            for category in sampling_counts
        },
        "elapsed_seconds": time.time() - start,
        "test_evaluated": False,
    }
    for name, expected in source_hashes.items():
        if digest(args.data / name) != expected:
            raise ValueError(f"Source changed: {name}")
    (args.out / "result.json").write_text(json.dumps(result, indent=2))

    html = f'''<!doctype html><meta charset="utf-8"><title>Foreground-aware sampling pilot</title><style>body{{font:16px system-ui;max-width:1100px;margin:30px auto;padding:16px;background:#18232b;color:#eee}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #647482}}img{{width:24%;image-rendering:pixelated}}select{{padding:7px}}a{{color:#8dbaff}}</style><h1>Foreground-aware sampling pilot</h1><p>sample_01 within-volume development；pos_weight={args.pos_weight:g}；augmentation OFF；每 batch 一块 foreground-centered，另一块在 boundary-centered / uniform-random 间交替。Validation 为固定 Z=112/117/122/127 的 tiled-256 inference。未评价 test。</p><table><tr><th>Step</th><th>Batch loss / Dice</th><th>Validation loss / Dice / IoU</th><th>Validation predicted / labeled foreground</th><th>P(fg|foreground) / P(fg|background)</th></tr>'''
    for row in history:
        html += f'<tr><td>{row["step"]}</td><td>{row["batch_loss"]:.4f} / {row["batch_dice"]:.4f}</td><td>{row["validation_loss"]:.4f} / {row["validation_dice"]:.4f} / {row["validation_iou"]:.4f}</td><td>{row["validation_predicted_foreground_fraction"]:.2%} / {row["validation_labeled_foreground_fraction"]:.2%}</td><td>{row["validation_foreground_probability_mean"]:.3f} / {row["validation_background_probability_mean"]:.3f}</td></tr>'
    html += '''</table><h2>Snapshots</h2><select id="step"></select><p>Z=112 / 117 / 122 / 127</p><div id="images"></div><p><a href="metrics.csv">Metrics CSV</a> · <a href="sampling.csv">Sampling log</a> · <a href="result.json">Result JSON</a></p><script>const steps=''' + json.dumps([row["step"] for row in history]) + ''',zs=[112,117,122,127],s=document.getElementById('step');steps.forEach(x=>s.add(new Option('step '+x,x)));function show(){document.getElementById('images').innerHTML=zs.map(z=>`<img src="step_${String(s.value).padStart(4,'0')}/z${String(z).padStart(3,'0')}_prediction.png">`).join('')}s.onchange=show;s.value=steps[steps.length-1];show();</script>'''
    (args.out / "index.html").write_text(html)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
