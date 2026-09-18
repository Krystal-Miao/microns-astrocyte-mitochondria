# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import io_utils


def main():
    parser = argparse.ArgumentParser(description="Filters label map to proofread subset for strict view overlays.")
    parser.add_argument("--run-dir", default=None, help="Path to run directory (e.g. astro_<root_id>/runs/<date>)")
    parser.add_argument("--data-dir", default=None, help="Path to astro_<root_id> directory")
    args = parser.parse_args()

    out_dir, run_dir = io_utils.resolve_script_dirs(args.run_dir, args.data_dir)
    strict_dir = os.path.join(run_dir, "strict_view")
    os.makedirs(strict_dir, exist_ok=True)

    with open(io_utils.interaction_label_map_path(run_dir)) as f:
        full_entries = json.load(f)["entries"]

    strict_entries = [e for e in full_entries if e["proofread"]]
    print(f"full entries: {len(full_entries)}, strict entries: {len(strict_entries)}")

    io_utils.save_interaction_label_map(strict_dir, strict_entries)

    max_label = max(e["local_id"] for e in full_entries)
    print("max_label (max local_id in current, corrected id map):", max_label)

    image_path = io_utils.image_path(out_dir)
    label_path = io_utils.interaction_labels_path(run_dir)

    n_sections = io_utils.save_astrocyte_neuron_overlay_stack(
        strict_dir, image_path, label_path, max_label, downsample=10,
    )
    print(f"strict-view overlay: {n_sections} sections -> {strict_dir}/astrocyte_neuron_overlays/")


if __name__ == "__main__":
    main()

