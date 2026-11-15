"""Evaluate one 3D checkpoint on fixed patches from train/validation/test cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_partial_astrocyte_3d import (  # noqa: E402
    UNet3D,
    fixed_validation_coordinates,
    load,
    validate,
)


def baselines(score):
    prevalence = score["labeled_foreground_fraction"]
    return {"random_ap": prevalence, "all_positive_dice": 2 * prevalence / (1 + prevalence)}


def evaluate_group(model, data, patch, device, seed):
    coordinates = {
        name: fixed_validation_coordinates(labels, patch, seed=seed + i)
        for i, (name, (_, labels)) in enumerate(data.items())
    }
    aggregate = validate(model, data, coordinates, patch, device)
    aggregate.update(baselines(aggregate))
    per_cell = {}
    for name, arrays in data.items():
        score = validate(model, {name: arrays}, {name: coordinates[name]}, patch, device)
        score.update(baselines(score))
        per_cell[name] = score
    return {"aggregate": aggregate, "per_cell": per_cell, "coordinates_zyx": coordinates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    checkpoint = torch.load(args.run / "best.pt", map_location=args.device, weights_only=True)
    model = UNet3D(checkpoint["width"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    patch = tuple(config["patch_zyx"])
    roots = [Path(root) for root in config["roots"]]
    groups = {
        "train": load(roots, config["train_samples"]),
        "validation": load(roots, config["validation_samples"]),
        "test": load(roots, config["test_samples"]),
    }
    result = {
        "checkpoint_step": checkpoint["step"],
        "protocol": "8 fixed patches/cell: 6 foreground-centered, 2 spatially uniform",
        "splits": {
            name: evaluate_group(model, data, patch, args.device, seed)
            for seed, (name, data) in zip((4026, 5026, 6026), groups.items())
        },
    }
    (args.run / "split_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, row in result["splits"].items():
        score = row["aggregate"]
        print(f"{name}: AP={score['average_precision']:.4f}/{score['random_ap']:.4f} "
              f"bestDice={score['best_threshold_dice']:.4f}/{score['all_positive_dice']:.4f} "
              f"Pfg={score['foreground_probability_mean']:.4f} "
              f"Pbg={score['background_probability_mean']:.4f}")


if __name__ == "__main__":
    main()
