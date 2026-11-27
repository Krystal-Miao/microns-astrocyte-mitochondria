"""Compare full-XY and tiled inference on one development section."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_fixed_patch_checkpoints import predict, predict_tiled
from evaluate_multicell3d_whole_sections import save_composite, seam_diagnostic, starts, stitch_section, summarize
from train_astro_unet import UNet
from train_partial_astrocyte_3d import UNet3DNNStyle, arrays_from_labels, load, normalize_raw, uint8_normalization_stats


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["2d", "3d"], required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", default="sample_09")
    parser.add_argument("--z", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    image, labels = load([args.root], [args.sample])[args.sample]
    z = args.z
    if not 0 <= z < len(image) or np.ptp(image[z]) == 0:
        parser.error("Choose an in-range, nonconstant section")
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    model_type = UNet if args.mode == "2d" else UNet3DNNStyle
    model = model_type(checkpoint["width"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    raw, audit = image[z], labels[z]
    truth, valid = arrays_from_labels(audit)
    truth = truth.astype(bool)
    stats = uint8_normalization_stats(image)
    threshold = 0.5
    if args.mode == "2d":
        tile_yx = (256, 256)
        grid = {"y_starts": starts(raw.shape[0], 256, 128),
                "x_starts": starts(raw.shape[1], 256, 128)}
        full = lambda: predict(model, raw)
        tiled = lambda: predict_tiled(model, raw, 256, 128)
    else:
        patch = tuple(json.loads((args.run / "config.json").read_text())["patch_zyx"])
        tile_yx = patch[1:]
        threshold = float(checkpoint["validation"]["best_threshold"])
        z0 = min(max(z - patch[0] // 2, 0), len(image) - patch[0])
        grid = {"y_starts": starts(raw.shape[0], patch[1], 64),
                "x_starts": starts(raw.shape[1], patch[2], 64)}

        def full():
            slab = normalize_raw(np.array(image[z0:z0 + patch[0]]), stats)
            return model(slab[None, None])[0, 0, z - z0].sigmoid().numpy()

        def tiled():
            return stitch_section(model, image, z, patch, 64, stats, "cpu")[0]

    args.out.mkdir(parents=True, exist_ok=False)
    result = {"sample": args.sample, "z": z, "checkpoint": str(args.run / "best.pt"),
              "step": checkpoint["step"], "threshold": threshold}
    for name, infer in [("full", full), ("tiled", tiled)]:
        start = time.perf_counter()
        probability = infer()
        elapsed = time.perf_counter() - start
        if probability.shape != raw.shape or not np.isfinite(probability).all():
            raise ValueError("Expected a finite probability map matching the section")
        result[name] = {"seconds": elapsed, "scores": summarize(probability[valid], truth[valid], threshold),
                        "jump_at_tiled_grid": seam_diagnostic(probability, **grid)}
        save_composite(args.out / f"{name}.png", raw, audit, probability, threshold, stats, grid, tile_yx)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
