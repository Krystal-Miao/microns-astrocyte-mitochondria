# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
Analyze contact distance threshold sensitivity by binning per-voxel physical distances into histograms.
"""

import json
import math
import mmap
import os
import sys
import time

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt

# This project is flattened (no "01_astrocyte_pipeline/" nesting, "src/" is a
# direct sibling of "scripts/"), unlike the toolkit this was copied from.
PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PIPELINE_DIR)
from src import contact_filter  # noqa: E402  (safe_contact_filter_chunk_and_tile_voxels)

import argparse
import json
import math
import mmap
import os
import sys
import time

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PIPELINE_DIR)
from src import contact_filter, io_utils  # noqa: E402

BIN_WIDTH_NM = 5.0
MAX_BIN_NM = 80.0
N_BINS = int(MAX_BIN_NM / BIN_WIDTH_NM)
REPORT_THRESHOLDS_NM = [40.0, 50.0, 60.0, 70.0, 80.0]

MAX_MEMORY_GB = 25.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Path to run directory (e.g. astro_<root_id>/runs/<date>)")
    parser.add_argument("--data-dir", default=None, help="Path to astro_<root_id> directory")
    args = parser.parse_args()

    out_dir, run_dir = io_utils.resolve_script_dirs(args.run_dir, args.data_dir)

    mask_path = io_utils.mask_path(out_dir)
    full_mask_label_path = os.path.join(run_dir, "interaction_labels_full_mask.tif")
    result_path = os.path.join(run_dir, "threshold_sensitivity_result.json")
    histogram_path = os.path.join(run_dir, "threshold_sensitivity_histogram.npz")

    with open(os.path.join(run_dir, "interaction_label_map.json")) as f:
        entries = json.load(f)["entries"]
    local_id_to_entry = {e["local_id"]: e for e in entries}
    max_local_id = max(local_id_to_entry)
    print(f"{len(entries)} loose entries, max_local_id={max_local_id}")

    with open(io_utils.metadata_path(out_dir)) as f:
        metadata = json.load(f)
    res_x, res_y, res_z = metadata["seg_resolution_nm"]
    resolution_nm_xyz = (res_x, res_y, res_z)

    mask_arr = tifffile.memmap(mask_path)
    label_arr = tifffile.memmap(full_mask_label_path)
    nz, ny, nx = mask_arr.shape
    assert label_arr.shape == mask_arr.shape

    mask_bytes_per_plane = ny * nx * mask_arr.dtype.itemsize
    label_bytes_per_plane = ny * nx * label_arr.dtype.itemsize
    mask_mmap_len = len(mask_arr._mmap)
    label_mmap_len = len(label_arr._mmap)
    page_size = mmap.PAGESIZE
    mask_fd = os.open(mask_path, os.O_RDONLY)
    label_fd = os.open(full_mask_label_path, os.O_RDONLY)

    def _release_z_range(arr, fd, mmap_len, bytes_per_plane, z0, z1):
        byte_start = z0 * bytes_per_plane
        byte_end = min(z1 * bytes_per_plane, mmap_len)
        aligned_start = (byte_start // page_size) * page_size
        if byte_end > aligned_start:
            arr._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)
            os.posix_fadvise(fd, aligned_start, byte_end - aligned_start, os.POSIX_FADV_DONTNEED)

    chunk_z, tile_y, tile_x = contact_filter.safe_contact_filter_chunk_and_tile_voxels(
        bbox_min_vx=(0, 0, 0), bbox_max_vx=(nx, ny, nz),
        max_memory_gb=MAX_MEMORY_GB, configured_max_chunk_z=32,
        resolution_nm_xyz=resolution_nm_xyz, threshold_nm=MAX_BIN_NM,
    )
    print(f"chunk_z={chunk_z} tile_y={tile_y} tile_x={tile_x}")

    halo_z = contact_filter._halo_z_voxels(MAX_BIN_NM, res_z)
    halo_y = contact_filter._halo_z_voxels(MAX_BIN_NM, res_y)
    halo_x = contact_filter._halo_z_voxels(MAX_BIN_NM, res_x)

    histogram = np.zeros((max_local_id + 1, N_BINS), dtype=np.int64)

    tiles = [
        (y0, min(y0 + tile_y, ny), x0, min(x0 + tile_x, nx))
        for y0 in range(0, ny, tile_y)
        for x0 in range(0, nx, tile_x)
    ]
    t0 = time.time()
    n_chunks_done = 0
    for tile_idx, (y0, y1, x0, x1) in enumerate(tiles):
        halo_y0, halo_y1 = max(0, y0 - halo_y), min(ny, y1 + halo_y)
        halo_x0, halo_x1 = max(0, x0 - halo_x), min(nx, x1 + halo_x)

        for z0 in range(0, nz, chunk_z):
            z1 = min(z0 + chunk_z, nz)
            halo_z0, halo_z1 = max(0, z0 - halo_z), min(nz, z1 + halo_z)

            mask_chunk = np.array(mask_arr[halo_z0:halo_z1, halo_y0:halo_y1, halo_x0:halo_x1]) != 0
            dist = distance_transform_edt(~mask_chunk, sampling=(res_z, res_y, res_x))
            del mask_chunk

            dist_core = dist[
                z0 - halo_z0: z0 - halo_z0 + (z1 - z0),
                y0 - halo_y0: y0 - halo_y0 + (y1 - y0),
                x0 - halo_x0: x0 - halo_x0 + (x1 - x0),
            ]
            del dist

            label_chunk = np.array(label_arr[z0:z1, y0:y1, x0:x1])
            nz_mask = label_chunk != 0
            if nz_mask.any():
                labels_nz = label_chunk[nz_mask]
                dist_nz = dist_core[nz_mask]
                in_range = dist_nz < MAX_BIN_NM
                if in_range.any():
                    bin_idx = (dist_nz[in_range] // BIN_WIDTH_NM).astype(np.int64)
                    flat_idx = labels_nz[in_range].astype(np.int64) * N_BINS + bin_idx
                    flat_counts = np.bincount(flat_idx, minlength=(max_local_id + 1) * N_BINS)
                    histogram += flat_counts.reshape(max_local_id + 1, N_BINS)
            del label_chunk, dist_core, nz_mask

            _release_z_range(mask_arr, mask_fd, mask_mmap_len, mask_bytes_per_plane, halo_z0, halo_z1)
            _release_z_range(label_arr, label_fd, label_mmap_len, label_bytes_per_plane, z0, z1)

            n_chunks_done += 1
            if n_chunks_done % 20 == 0:
                print(f"  tile {tile_idx + 1}/{len(tiles)}, chunk z={z0}:{z1} ({n_chunks_done} chunks total, {time.time() - t0:.0f}s elapsed)", flush=True)

        print(f"tile {tile_idx + 1}/{len(tiles)} done ({time.time() - t0:.0f}s elapsed)", flush=True)

    def category(entry):
        if entry.get("is_target_astrocyte"):
            return "target"
        if entry.get("cell_type") == "astrocyte":
            return "astrocyte"
        if entry.get("classification_system") in ("excitatory_neuron", "inhibitory_neuron"):
            return "neuron"
        return "other"

    np.savez_compressed(histogram_path, histogram=histogram, local_ids=np.array(sorted(local_id_to_entry)),
                         bin_width_nm=BIN_WIDTH_NM, max_bin_nm=MAX_BIN_NM)
    print(f"histogram saved -> {histogram_path}")

    cum = np.cumsum(histogram, axis=1)
    summary = {}
    for th in REPORT_THRESHOLDS_NM:
        b = int(th / BIN_WIDTH_NM) - 1
        c = cum[:, b]
        surviving_ids = [lid for lid in local_id_to_entry if c[lid] > 0]
        by_cat = {}
        for lid in surviving_ids:
            cat = category(local_id_to_entry[lid])
            by_cat[cat] = by_cat.get(cat, 0) + 1
        summary[th] = {
            "n_surviving_root_ids": len(surviving_ids),
            "by_category": by_cat,
            "total_surviving_voxels": int(c.sum()),
        }
        print(f"threshold={th}nm: {len(surviving_ids)} root_ids survive {by_cat}")

    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {result_path}")


if __name__ == "__main__":
    main()

