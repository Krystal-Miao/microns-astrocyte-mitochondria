# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
2D EM+label overlays from build_full_mask_labels.py's full-mask tifs
(complete shapes, not the trimmed sliver).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import io_utils

FULL_MASK_NAME = "interaction_labels_full_mask.tif"
OVERLAY_DIR_NAME = "full_mask_overlays"


def build(run_dir: str, label: str, out_dir: str) -> None:
    with open(os.path.join(run_dir, "interaction_label_map.json")) as f:
        entries = json.load(f)["entries"]
    max_label = max(e["local_id"] for e in entries)
    print(f"[{label}] {len(entries)} entries, max_label={max_label}")

    image_path = io_utils.image_path(out_dir)
    label_path = os.path.join(run_dir, FULL_MASK_NAME)
    overlay_out_dir = os.path.join(run_dir, OVERLAY_DIR_NAME)
    os.makedirs(overlay_out_dir, exist_ok=True)
    # save_astrocyte_neuron_overlay_stack reads interaction_label_map.json from out_dir, so give this new dir its own copy.
    io_utils.save_interaction_label_map(overlay_out_dir, entries)

    # chunk_z=4, not default 10: avoids repeated OOM kills on the proofread pass.
    n_sections = io_utils.save_astrocyte_neuron_overlay_stack(overlay_out_dir, image_path, label_path, max_label, downsample=10,
                                                               chunk_z=4, )
    print(f"[{label}] {n_sections} sections -> {overlay_out_dir}/astrocyte_neuron_overlays/")

    legend_path = os.path.join(overlay_out_dir, "astrocyte_neuron_overlays", "legend.png")
    io_utils.save_label_legend(legend_path, entries)
    print(f"[{label}] legend -> {legend_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Path to run directory (e.g. astro_<root_id>/runs/<date>)")
    parser.add_argument("--data-dir", default=None, help="Path to astro_<root_id> directory")
    args = parser.parse_args()

    out_dir, run_dir = io_utils.resolve_script_dirs(args.run_dir, args.data_dir)
    strict_dir = os.path.join(run_dir, "strict_view")

    build(run_dir, "loose", out_dir)
    build(strict_dir, "proofread", out_dir)

