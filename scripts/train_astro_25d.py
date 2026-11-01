"""2.5D U-Net diagnostics: five adjacent EM slices predict the center mask."""

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
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_astro_unet import save_overlay  # noqa: E402
from train_astro_unet import block, digest, loss_fn, metrics  # noqa: E402
from train_foreground_aware_pilot import make_sampler  # noqa: E402


class UNet25D(nn.Module):
    def __init__(self, width=16, input_slices=5):
        super().__init__()
        self.input_slices = input_slices
        self.down = nn.ModuleList([
            block(input_slices, width), block(width, width * 2), block(width * 2, width * 4)
        ])
        self.bottom = block(width * 4, width * 8)
        self.up = nn.ModuleList([
            block(width * 12, width * 4), block(width * 6, width * 2), block(width * 3, width)
        ])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for layer, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = layer(torch.cat([x, skip], dim=1))
        return self.head(x)


def predict_tiled(model, stack, size=256, stride=128):
    _, height, width = stack.shape
    ys = list(range(0, height - size + 1, stride))
    xs = list(range(0, width - size + 1, stride))
    if ys[-1] != height - size:
        ys.append(height - size)
    if xs[-1] != width - size:
        xs.append(width - size)
    total = np.zeros((height, width), np.float32)
    count = np.zeros((height, width), np.uint16)
    with torch.no_grad():
        model.eval()
        for y in ys:
            for x in xs:
                patch = np.asarray(stack[:, y:y + size, x:x + size]).copy()
                tensor = torch.from_numpy(patch)[None].float() / 255
                probability = model(tensor).sigmoid()[0, 0].numpy()
                total[y:y + size, x:x + size] += probability
                count[y:y + size, x:x + size] += 1
    if not np.all(count):
        raise ValueError("Tiled inference left uncovered pixels")
    return total / count


def measure(probability, target):
    truth = target.astype(bool)
    prediction = probability >= 0.5
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    result = metrics(tp, fp, fn)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = torch.from_numpy(np.log(clipped / (1 - clipped)))[None, None].float()
    y = torch.from_numpy(target.copy())[None, None].float()
    result.update(
        loss=float(loss_fn(logits, y)),
        predicted_foreground_fraction=float(prediction.mean()),
        labeled_foreground_fraction=float(truth.mean()),
        foreground_probability_mean=float(probability[truth].mean()),
        background_probability_mean=float(probability[~truth].mean()),
    )
    return result


def validate(model, image, mask, centers):
    pooled = Counter(tp=0, fp=0, fn=0)
    results = []
    probabilities = {}
    for z in centers:
        probability = predict_tiled(model, np.asarray(image[z - 2:z + 3]))
        score = measure(probability, np.asarray(mask[z]))
        probabilities[z] = probability
        for key in ("tp", "fp", "fn"):
            pooled[key] += score[key]
        results.append(score)
    combined = metrics(pooled["tp"], pooled["fp"], pooled["fn"])
    for key in ("loss", "predicted_foreground_fraction", "labeled_foreground_fraction",
                "foreground_probability_mean", "background_probability_mean"):
        combined[key] = float(np.mean([row[key] for row in results]))
    combined["evaluated_slices"] = len(centers)
    return combined, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("samples/astro_864691136057675736_run1/sample_01"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Repeatedly train on one Z=47 patch")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    image = tifffile.memmap(args.data / "image.tif", mode="r")
    mask = tifffile.memmap(args.data / "mask.tif", mode="r")
    if image.shape != (128, 512, 512) or mask.shape != image.shape:
        raise ValueError("Expected aligned 128x512x512 image and mask")
    hashes = {name: digest(args.data / name) for name in ("image.tif", "mask.tif", "metadata.json")}
    train_z = [z for z in range(2, 94) if np.ptp(image[z]) > 0]
    validation_z = [112, 117, 122, 125]
    sampler = make_sampler(mask, train_z, 256)
    model = UNet25D(16, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    config = {
        "status": "2.5D development diagnostic", "smoke": args.smoke,
        "input": "five adjacent uint8/255 EM slices as channels", "target": "center-slice mask",
        "steps": args.steps, "seed": args.seed, "width": 16, "batch_size": 1 if args.smoke else 2,
        "patch_yx": [256, 256], "learning_rate": 0.0003, "augmentation": False,
        "loss": "BCEWithLogits + soft Dice", "threshold": 0.5,
        "train_center_z": train_z, "validation_center_z": validation_z,
        "source_sha256": hashes, "script_sha256": digest(Path(__file__)),
        "scope": "sample_01 development only; no test or biological generalization",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    history, trace = [], []
    best_key = (-1.0, float("-inf"))
    start = time.time()

    for step in range(1, args.steps + 1):
        if args.smoke:
            draws = [(47, 256, 128, "fixed_smoke", None)]
        else:
            categories = ["foreground_centered", "boundary_centered" if step % 2 else "uniform_random"]
            draws = []
            for category in categories:
                z, y, x, anchor = sampler(rng, category)
                draws.append((z, y, x, category, anchor))
        x_batch, y_batch = [], []
        for batch_index, (z, y, x, category, anchor) in enumerate(draws):
            x_batch.append(np.asarray(image[z - 2:z + 3, y:y + 256, x:x + 256]).copy())
            center_mask = np.asarray(mask[z, y:y + 256, x:x + 256]).copy()
            y_batch.append(center_mask)
            trace.append({"step": step, "batch_index": batch_index, "category": category,
                          "z": z, "y": y, "x": x, "foreground_fraction": float(center_mask.mean())})
        x_tensor = torch.from_numpy(np.stack(x_batch)).float() / 255
        y_tensor = torch.from_numpy(np.stack(y_batch)[:, None]).float()
        model.train()
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = loss_fn(logits, y_tensor)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        probability = logits.detach().sigmoid().numpy()[:, 0]
        batch_score = measure(probability.reshape(-1, 256), np.stack(y_batch).reshape(-1, 256))
        loss.backward()
        gradient_l2 = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
        optimizer.step()

        interval = 50
        if step % interval == 0 or step == args.steps:
            if args.smoke:
                with torch.no_grad():
                    model.eval()
                    smoke_probability = model(x_tensor).sigmoid()[0, 0].numpy()
                validation = measure(smoke_probability, y_batch[0])
                probabilities = {47: smoke_probability}
            else:
                validation, probabilities = validate(model, image, mask, validation_z)
            row = {"step": step, "batch_loss": float(loss.detach()), "gradient_l2": gradient_l2,
                   **{f"batch_{key}": value for key, value in batch_score.items()},
                   **{f"validation_{key}": value for key, value in validation.items()},
                   "elapsed_seconds": time.time() - start}
            history.append(row)
            key = (validation["dice"], -validation["loss"])
            if key > best_key:
                best_key = key
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "width": 16, "input_slices": 5, "step": step,
                            "validation": validation}, args.out / "best.pt")
            snapshot = args.out / f"step_{step:04d}"
            snapshot.mkdir()
            for z, p in probabilities.items():
                background = np.asarray(image[z, 256:512, 128:384]) if args.smoke else np.asarray(image[z])
                save_overlay(snapshot / f"z{z:03d}_prediction.png", background, p >= 0.5)
            print(f"step {step}/{args.steps} loss={float(loss):.4f} batch_dice={batch_score['dice']:.4f} "
                  f"eval_dice={validation['dice']:.4f} eval_iou={validation['iou']:.4f} "
                  f"pred_fg={validation['predicted_foreground_fraction']:.2%} "
                  f"Pfg={validation['foreground_probability_mean']:.3f} "
                  f"Pbg={validation['background_probability_mean']:.3f} grad={gradient_l2:.3g} "
                  f"elapsed={row['elapsed_seconds']:.0f}s", flush=True)

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "width": 16, "input_slices": 5, "step": args.steps,
                "validation": validation}, args.out / "last.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    with (args.out / "sampling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader(); writer.writerows(trace)
    best = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)
    result = {"best_step": best["step"], "best_evaluation": best["validation"],
              "last_evaluation": validation, "elapsed_seconds": time.time() - start,
              "test_evaluated": False}
    for name, expected in hashes.items():
        if digest(args.data / name) != expected:
            raise ValueError(f"Source changed: {name}")
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
