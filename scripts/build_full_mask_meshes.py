# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
3D meshes from build_full_mask_labels.py's COMPLETE-shape tifs.
reconstruct_local_meshes hard-codes its input filename, so this symlinks a
throwaway view dir's interaction_labels.tif to "../interaction_labels_full_mask.tif".
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import io_utils, mesh_export

FULL_MASK_NAME = "interaction_labels_full_mask.tif"
VIEW_DIR_NAME = "full_mask_view"
MESH_DIR_NAME = "meshes_full_mask_ds10"


def build(run_dir: str, label: str, out_dir: str) -> None:
    view_dir = os.path.join(run_dir, VIEW_DIR_NAME)
    os.makedirs(view_dir, exist_ok=True)

    link_path = os.path.join(view_dir, "interaction_labels.tif")
    target = os.path.join(run_dir, FULL_MASK_NAME)
    if os.path.islink(link_path) or os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(target, link_path)

    with open(os.path.join(run_dir, "interaction_label_map.json")) as f:
        entries = json.load(f)["entries"]
    io_utils.save_interaction_label_map(view_dir, entries)

    # chunk_z=2, not default 64: avoids OOM on this volume size (uncheckpointed step).
    results = mesh_export.reconstruct_local_meshes(out_dir, view_dir, downsample=10, mesh_dir_name=MESH_DIR_NAME,
                                                   skip_existing=True, chunk_z=2, )
    ok = sum(1 for _, _, status in results if status.startswith("OK") or status == "already saved")
    failed = [r for r in results if r[2].startswith("FAILED")]
    skipped = [r for r in results if r[2].startswith("SKIPPED")]
    print(f"[{label}] {ok}/{len(results)} meshes OK, {len(skipped)} skipped, {len(failed)} failed")
    for root_id, category, status in failed[:10]:
        print(f"  FAILED {category} {root_id}: {status}")
    print(f"[{label}] meshes -> {view_dir}/{MESH_DIR_NAME}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Path to run directory (e.g. astro_<root_id>/runs/<date>)")
    parser.add_argument("--data-dir", default=None, help="Path to astro_<root_id> directory")
    args = parser.parse_args()

    out_dir, run_dir = io_utils.resolve_script_dirs(args.run_dir, args.data_dir)
    strict_dir = os.path.join(run_dir, "strict_view")

    build(run_dir, "loose", out_dir)
    build(strict_dir, "proofread", out_dir)

