# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
Rebuild full mask label volumes from pre-filter slabs for loose and proofread views.
"""

import argparse
import glob
import json
import mmap
import numpy as np
import os
import sys
import tifffile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import io_utils

OUT_NAME = "interaction_labels_full_mask.tif"


def _load_lut(run_dir: str) -> np.ndarray:
    with open(os.path.join(run_dir, "interaction_label_map.json")) as f:
        entries = json.load(f)["entries"]
    lut = np.zeros(65536, dtype=np.uint16)  # identity for kept ids, 0 for dropped
    for e in entries:
        lid = e["local_id"]
        lut[lid] = lid
    print(f"  {run_dir}: {len(entries)} qualifying local_ids")
    return lut


def _slab_z_range(path: str) -> tuple[int, int]:
    name = os.path.splitext(os.path.basename(path))[0]
    z0, z1 = name.split("_")
    return int(z0), int(z1)


def build_full_mask_tif(lut: np.ndarray, out_path: str, label: str, metadata_path: str, slabs_dir: str) -> None:
    with open(metadata_path) as f:
        bbox_min_z = int(json.load(f)["voxel_bbox"]["min"][2])

    paths = sorted(glob.glob(os.path.join(slabs_dir, "*.tif")), key=_slab_z_range)
    if not paths:
        print(f"[{label}] No slab TIFs found in {slabs_dir}")
        return

    first = tifffile.imread(paths[0])
    ny, nx = first.shape[1:]
    total_z = sum(b - a for a, b in (_slab_z_range(p) for p in paths))
    dtype = first.dtype
    del first

    final_shape = (total_z, ny, nx)
    bytes_per_slice = ny * nx * np.dtype(dtype).itemsize
    page_size = mmap.PAGESIZE

    print(f"[{label}] writing {out_path} shape={final_shape} dtype={dtype}")
    mm = tifffile.memmap(out_path, shape=final_shape, dtype=dtype, bigtiff=True, photometric="minisblack")
    mmap_len = len(mm._mmap)

    t0 = time.time()
    for i, path in enumerate(paths):
        z0_abs, z1_abs = _slab_z_range(path)
        z0 = z0_abs - bbox_min_z
        n = z1_abs - z0_abs

        slab = tifffile.imread(path)
        filtered = lut[slab]  # fancy-index remap: non-qualifying local_ids -> 0
        mm[z0: z0 + n] = filtered
        mm.flush()

        byte_start = z0 * bytes_per_slice
        byte_end = min(byte_start + n * bytes_per_slice, mmap_len)
        aligned_start = (byte_start // page_size) * page_size
        if byte_end > aligned_start:
            mm._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)

        if i % 20 == 0:
            print(f"  [{label}] slab {i + 1}/{len(paths)} ({time.time() - t0:.0f}s elapsed)")

    del mm
    print(f"[{label}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Path to run directory (e.g. astro_<root_id>/runs/<date>)")
    parser.add_argument("--data-dir", default=None, help="Path to astro_<root_id> directory")
    args = parser.parse_args()

    out_dir, run_dir = io_utils.resolve_script_dirs(args.run_dir, args.data_dir)
    strict_dir = os.path.join(run_dir, "strict_view")
    slabs_dir = os.path.join(run_dir, "_slabs", "interaction_label")
    metadata_path = io_utils.metadata_path(out_dir)

    print("loading qualifying-id lookup tables...")
    loose_lut = _load_lut(run_dir)
    strict_lut = _load_lut(strict_dir)

    build_full_mask_tif(loose_lut, os.path.join(run_dir, OUT_NAME), "loose", metadata_path, slabs_dir)
    build_full_mask_tif(strict_lut, os.path.join(strict_dir, OUT_NAME), "proofread", metadata_path, slabs_dir)

