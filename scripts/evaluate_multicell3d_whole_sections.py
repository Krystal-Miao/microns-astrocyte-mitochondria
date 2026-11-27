"""Stitch overlapping 3D predictions and evaluate fixed full XY sections."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_multicell3d_final_test import threshold_metrics  # noqa: E402
from train_foreground_aware_pilot import ranking_metrics  # noqa: E402
from train_partial_astrocyte_3d import (  # noqa: E402
    UNet3DNNStyle,
    arrays_from_labels,
    load,
    normalize_raw,
    uint8_normalization_stats,
)


COLORS = {
    "reference": np.array([235, 72, 72], dtype=np.float32),
    "prediction": np.array([57, 205, 222], dtype=np.float32),
    "tp": np.array([62, 205, 112], dtype=np.float32),
    "fp": np.array([244, 78, 78], dtype=np.float32),
    "fn": np.array([71, 125, 245], dtype=np.float32),
}


def starts(length: int, patch: int, stride: int) -> list[int]:
    values = list(range(0, max(length - patch, 0) + 1, stride))
    last = max(length - patch, 0)
    if not values or values[-1] != last:
        values.append(last)
    return values


def feather(size: int) -> np.ndarray:
    # A nonzero Hann window preserves coverage at the outer image boundary.
    return np.maximum(np.hanning(size).astype(np.float32), 0.05)


@torch.no_grad()
def stitch_section(model, image, z, patch, stride_xy, stats, device, blend="hann"):
    model.eval()
    pz, py, px = patch
    z0 = min(max(int(z) - pz // 2, 0), image.shape[0] - pz)
    local_z = int(z) - z0
    ys = starts(image.shape[1], py, stride_xy)
    xs = starts(image.shape[2], px, stride_xy)
    if blend == "hann":
        weight2d = feather(py)[:, None] * feather(px)[None, :]
    elif blend == "uniform":
        weight2d = 1.0
    else:
        raise ValueError(f"Unknown blending method: {blend}")
    total = np.zeros(image.shape[1:], dtype=np.float32)
    weights = np.zeros_like(total)
    for y0 in ys:
        for x0 in xs:
            raw = np.asarray(image[z0:z0 + pz, y0:y0 + py, x0:x0 + px]).copy()
            tensor = normalize_raw(raw, stats)[None, None].to(device)
            probability = model(tensor).sigmoid()[0, 0, local_z].cpu().numpy()
            total[y0:y0 + py, x0:x0 + px] += probability * weight2d
            weights[y0:y0 + py, x0:x0 + px] += weight2d
    return total / weights, {"z_start": z0, "y_starts": ys, "x_starts": xs}


def summarize(probability, truth, threshold):
    result = threshold_metrics(probability, truth, threshold)
    result.update(ranking_metrics(probability, truth))
    result.update(
        evaluated_pixels=int(truth.size),
        labeled_foreground_fraction=float(truth.mean()),
        foreground_probability_mean=float(probability[truth].mean()),
        background_probability_mean=float(probability[~truth].mean()),
    )
    return result


def seam_diagnostic(probability, y_starts, x_starts):
    dx = np.abs(np.diff(probability, axis=1))
    dy = np.abs(np.diff(probability, axis=0))
    xb = [x - 1 for x in x_starts[1:] if 0 < x < probability.shape[1]]
    yb = [y - 1 for y in y_starts[1:] if 0 < y < probability.shape[0]]
    values = []
    if xb:
        values.append(dx[:, xb].ravel())
    if yb:
        values.append(dy[yb, :].ravel())
    boundary = float(np.concatenate(values).mean()) if values else 0.0
    global_difference = float((dx.mean() + dy.mean()) / 2)
    return {
        "mean_probability_jump_at_tile_starts": boundary,
        "mean_global_neighbor_jump": global_difference,
        "boundary_to_global_jump_ratio": boundary / global_difference if global_difference else 0.0,
    }


def grayscale(raw, stats):
    low, high = stats["low"], stats["high"]
    scaled = np.clip((raw.astype(np.float32) - low) / max(high - low, 1), 0, 1)
    gray = (scaled * 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def overlay(base, mask, color, alpha=0.48):
    result = base.astype(np.float32).copy()
    result[mask] = result[mask] * (1 - alpha) + color * alpha
    return result.astype(np.uint8)


def error_overlay(base, prediction, truth, valid):
    result = base.copy()
    tp = prediction & truth & valid
    fp = prediction & ~truth & valid
    fn = ~prediction & truth & valid
    result = overlay(result, tp, COLORS["tp"], 0.58)
    result = overlay(result, fp, COLORS["fp"], 0.65)
    result = overlay(result, fn, COLORS["fn"], 0.65)
    return result


def label_panel(array, title, subtitle=""):
    image = Image.fromarray(array)
    canvas = Image.new("RGB", (image.width, image.height + 46), (24, 35, 43))
    canvas.paste(image, (0, 46))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="white", font=ImageFont.load_default(size=16))
    if subtitle:
        draw.text((8, 25), subtitle, fill=(190, 205, 215), font=ImageFont.load_default(size=12))
    return canvas


def save_composite(path, raw, audit, probability, threshold, stats, grid, tile_yx=(128, 128)):
    target, valid = arrays_from_labels(audit)
    truth = target.astype(bool)
    prediction = probability >= threshold
    base = grayscale(raw, stats)
    reference = overlay(base, truth, COLORS["reference"])
    predicted = overlay(base, prediction, COLORS["prediction"])
    errors = error_overlay(base, prediction, truth, valid)
    tiled = Image.fromarray(predicted).convert("RGBA")
    guides = Image.new("RGBA", tiled.size)
    draw = ImageDraw.Draw(guides)
    color = (255, 224, 112, 155)
    for x in sorted({edge for start in grid["x_starts"] for edge in (start, start + tile_yx[1])}):
        if 0 < x < tiled.width:
            draw.line((x, 0, x, tiled.height - 1), fill=color)
    for y in sorted({edge for start in grid["y_starts"] for edge in (start, start + tile_yx[0])}):
        if 0 < y < tiled.height:
            draw.line((0, y, tiled.width - 1, y), fill=color)
    tiled = Image.alpha_composite(tiled, guides).convert("RGB")
    panels = [
        label_panel(base, "Raw EM"),
        label_panel(reference, "Reference", "red = labeled astrocyte"),
        label_panel(predicted, "Prediction", "cyan = probability >= locked threshold"),
        label_panel(errors, "Errors", "green TP | red FP | blue FN"),
        label_panel(np.asarray(tiled), "Sliding-window reference", "gold = tile starts and ends"),
    ]
    canvas = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), (24, 35, 43))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(path, optimize=True)


def save_curves(history_path, output_path):
    history = json.loads(history_path.read_text())
    width, height = 1100, 620
    image = Image.new("RGB", (width, height), (24, 35, 43))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    title_font = ImageFont.load_default(size=20)
    draw.text((35, 18), "Training and validation curves", fill="white", font=title_font)
    plots = [
        ("Training loss", [r["train_loss"] for r in history], (255, 180, 84)),
        ("Validation average precision", [r["validation"]["average_precision"] for r in history], (104, 216, 172)),
        ("Validation best-threshold Dice", [r["validation"]["best_threshold_dice"] for r in history], (141, 186, 255)),
    ]
    steps = [r["step"] for r in history]
    for index, (name, values, color) in enumerate(plots):
        top = 65 + index * 180
        left, right, bottom = 90, width - 35, top + 135
        draw.line((left, top, left, bottom), fill=(145, 160, 170), width=1)
        draw.line((left, bottom, right, bottom), fill=(145, 160, 170), width=1)
        minimum = min(values)
        maximum = max(values)
        margin = max((maximum - minimum) * 0.12, 0.01)
        lo, hi = minimum - margin, maximum + margin
        points = []
        for step, value in zip(steps, values):
            x = left + (step - steps[0]) / max(steps[-1] - steps[0], 1) * (right - left)
            y = bottom - (value - lo) / (hi - lo) * (bottom - top)
            points.append((x, y))
        draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        draw.text((left, top - 24), name, fill="white", font=font)
        draw.text((10, top), f"{hi:.3f}", fill=(190, 205, 215), font=font)
        draw.text((10, bottom - 15), f"{lo:.3f}", fill=(190, 205, 215), font=font)
        draw.text((left, bottom + 7), str(steps[0]), fill=(190, 205, 215), font=font)
        draw.text((right - 40, bottom + 7), str(steps[-1]), fill=(190, 205, 215), font=font)
    image.save(output_path, optimize=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sections", type=int, nargs="+", default=[32, 64, 96])
    parser.add_argument("--stride-xy", type=int, default=96)
    parser.add_argument("--blend", choices=["hann", "uniform"], default="hann")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--full-xy", action="store_true")
    parser.add_argument("--threshold-from", type=Path, help="Validation Hann PR summary JSON")
    parser.add_argument("--save-pr-scores", action="store_true",
                        help="Save scored probabilities and labels for precision–recall analysis")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assets = args.out / "assets"
    assets.mkdir()
    torch.set_num_threads(4)

    config = json.loads((args.run / "config.json").read_text())
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    split_key = "validation_samples" if args.split == "validation" else "untouched_test_samples"
    names = config[split_key]
    patch = tuple(config["patch_zyx"])
    threshold = float(checkpoint["validation"]["best_threshold"])
    raw_data = load(args.roots, names)
    from train_multicell3d_controlled_random import preprocess_images
    variant = config.get("preprocessing_variant", "raw")
    if args.threshold_from:
        threshold = float(json.loads(args.threshold_from.read_text())[variant]["threshold_at_60_recall"])
    data = preprocess_images(raw_data, variant)
    normalizers = {name: uint8_normalization_stats(image) for name, (image, _) in data.items()}
    model = UNet3DNNStyle(checkpoint["width"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    start = time.time()
    all_probability, all_truth = [], []
    per_sample = {}
    section_results = []
    protocol_sections = {}
    for name, (image, labels) in data.items():
        raw_image = raw_data[name][0]
        display_stats = normalizers[name] if variant == "raw" else uint8_normalization_stats(raw_image)
        sample_probability, sample_truth = [], []
        protocol_sections[name] = list(args.sections)
        for z in args.sections:
            if args.full_xy:
                z0 = min(max(z - patch[0] // 2, 0), image.shape[0] - patch[0])
                slab = normalize_raw(np.array(image[z0:z0 + patch[0]]), normalizers[name])
                probability = model(slab[None, None].to(args.device))[0, 0, z - z0].sigmoid().cpu().numpy()
                grid = {"x_starts": [0], "y_starts": [0]}
            else:
                probability, grid = stitch_section(
                    model, image, z, patch, args.stride_xy, normalizers[name], args.device, args.blend
                )
            audit = np.asarray(labels[z]).copy()
            target, valid = arrays_from_labels(audit)
            truth = target.astype(bool)
            valid_probability = probability[valid]
            valid_truth = truth[valid]
            metrics = summarize(valid_probability, valid_truth, threshold)
            seam = None if args.full_xy else seam_diagnostic(probability, grid["y_starts"], grid["x_starts"])
            section_results.append({"sample": name, "z": z, "metrics": metrics, "seam": seam, "grid": grid})
            sample_probability.append(valid_probability)
            sample_truth.append(valid_truth)
            save_composite(
                assets / f"{name}_z{z:03d}_whole_section.png",
                np.asarray(raw_image[z]), audit, probability, threshold, display_stats, grid,
                image.shape[1:] if args.full_xy else patch[1:],
            )
        sample_probability = np.concatenate(sample_probability)
        sample_truth = np.concatenate(sample_truth)
        per_sample[name] = summarize(sample_probability, sample_truth, threshold)
        all_probability.append(sample_probability)
        all_truth.append(sample_truth)

    pooled_probability = np.concatenate(all_probability)
    pooled_truth = np.concatenate(all_truth)
    aggregate = summarize(pooled_probability, pooled_truth, threshold)
    if args.save_pr_scores:
        np.savez_compressed(args.out / "pr_scores.npz", probability=pooled_probability,
                            truth=pooled_truth, threshold=threshold)
    result = {
        "checkpoint_step": int(checkpoint["step"]),
        "locked_validation_threshold": threshold,
        "aggregate": aggregate,
        "per_sample": per_sample,
        "per_section": section_results,
        "elapsed_seconds": time.time() - start,
        "scope": f"Fixed full-XY {args.split} sections",
    }
    protocol = {
        "run": str(args.run),
        "checkpoint": str(args.run / "best.pt"),
        "split": args.split,
        "samples": names,
        "preprocessing": variant,
        "inference": "single full-XY forward pass" if args.full_xy else "overlapping 3D sliding window",
        "sections_by_sample": protocol_sections,
        "patch_zyx": list(patch),
        "stride_xy": None if args.full_xy else args.stride_xy,
        "blend": None if args.full_xy else {
            "hann": "separable Hann window clipped to minimum weight 0.05",
            "uniform": "equal-weight average of overlapping probabilities",
        }[args.blend],
        "threshold_source": (str(args.threshold_from) if args.threshold_from else
                             "frozen sample_09/10 validation-patch threshold"),
        "reuse_requirement": "Use these exact samples, Z sections, tiling, blending and scored-label mask for every run.",
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    save_curves(args.run / "history.json", assets / "training_validation_curves.png")
    print(json.dumps({"aggregate": aggregate, "per_sample": per_sample,
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
