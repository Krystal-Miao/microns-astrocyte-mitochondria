# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
Distance-based contact filtering for interaction labels against target astrocyte mask.
"""
from __future__ import annotations

import math
import mmap
import os

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt

from . import io_utils

# See volume_extractor.build_interaction_id_map's docstring: the target
# astrocyte is always assigned local_id 1. Its own voxels are always kept
# (distance 0 from itself, by definition); never run through the contact
# check, so a chunk with no other candidate voxels at all is a cheap no-op
# scan rather than something that needs special-casing.
TARGET_LOCAL_ID = 1

CONTACT_FILTER_PROGRESS_FILENAME = "_contact_filter_progress.json"

# Empirically measured (ru_maxrss on synthetic chunks) peak bytes per voxel for
# one haloed chunk: distance_transform_edt's buffers cost ~51-55, plus the
# label chunk (up to 4) and bool temporaries (~2). 80 gives real headroom.
_BYTES_PER_VOXEL_BUDGET = 80


def contact_filter_progress_path(run_dir: str) -> str:
    return os.path.join(run_dir, CONTACT_FILTER_PROGRESS_FILENAME)


def _halo_z_voxels(threshold_nm: float, res_z_nm: float) -> int:
    return max(1, math.ceil(threshold_nm / res_z_nm))


def safe_contact_filter_chunk_and_tile_voxels(
    bbox_min_vx, bbox_max_vx, max_memory_gb: float, configured_max_chunk_z: int,
    resolution_nm_xyz, threshold_nm: float,
    configured_max_tile_xy_voxels: int | None = None,
    safety_factor: float = 0.8,
) -> tuple[int, int, int]:
    """
    Calculate safe chunk and tile dimensions to fit within memory limits.
    """
    res_x, res_y, res_z = (float(v) for v in resolution_nm_xyz)
    halo_z = _halo_z_voxels(threshold_nm, res_z)
    halo_y = _halo_z_voxels(threshold_nm, res_y)
    halo_x = _halo_z_voxels(threshold_nm, res_x)

    nx_full = int(bbox_max_vx[0] - bbox_min_vx[0])
    ny_full = int(bbox_max_vx[1] - bbox_min_vx[1])
    tile_y = ny_full if configured_max_tile_xy_voxels is None else min(ny_full, configured_max_tile_xy_voxels)
    tile_x = nx_full if configured_max_tile_xy_voxels is None else min(nx_full, configured_max_tile_xy_voxels)

    budget_voxels = max(1, int(max_memory_gb * 1e9 * safety_factor // _BYTES_PER_VOXEL_BUDGET))

    while True:
        padded_voxels_per_z = (tile_y + 2 * halo_y) * (tile_x + 2 * halo_x)
        min_voxels_at_chunk_z_1 = (1 + 2 * halo_z) * padded_voxels_per_z
        if min_voxels_at_chunk_z_1 <= budget_voxels or (tile_y <= 1 and tile_x <= 1):
            break
        tile_y = max(1, tile_y // 2)
        tile_x = max(1, tile_x // 2)

    dynamic_max_chunk_z_with_halo = max(1, budget_voxels // padded_voxels_per_z)
    chunk_z = max(1, min(configured_max_chunk_z, dynamic_max_chunk_z_with_halo - 2 * halo_z))
    return chunk_z, tile_y, tile_x


def filter_interaction_labels_by_contact(
    run_dir: str,
    mask_path: str,
    label_path: str,
    resolution_nm_xyz,
    threshold_nm: float,
    request_key: dict,
    chunk_z_voxels: int = 32,
    target_local_id: int = TARGET_LOCAL_ID,
    tile_y_voxels: int | None = None,
    tile_x_voxels: int | None = None,
) -> set[int]:
    """
    Filter interaction labels by physical distance threshold to the target mask in place.
    """
    prog_path = contact_filter_progress_path(run_dir)
    progress = io_utils.load_progress(prog_path)

    res_x, res_y, res_z = (float(v) for v in resolution_nm_xyz)
    halo_z = _halo_z_voxels(threshold_nm, res_z)
    halo_y = _halo_z_voxels(threshold_nm, res_y)
    halo_x = _halo_z_voxels(threshold_nm, res_x)

    mask_arr = tifffile.memmap(mask_path)
    label_arr = tifffile.memmap(label_path, mode="r+")
    nz, ny, nx = mask_arr.shape
    if label_arr.shape != mask_arr.shape:
        raise ValueError(f"interaction_labels.tif shape {label_arr.shape} != mask.tif shape {mask_arr.shape}")

    tile_y = ny if tile_y_voxels is None else min(tile_y_voxels, ny)
    tile_x = nx if tile_x_voxels is None else min(tile_x_voxels, nx)
    tiles = [
        (y0, min(y0 + tile_y, ny), x0, min(x0 + tile_x, nx))
        for y0 in range(0, ny, tile_y)
        for x0 in range(0, nx, tile_x)
    ]

    internal_key = dict(request_key)
    internal_key["tile_shape"] = [tile_y, tile_x]
    stale = progress is None or progress.get("request_key") != internal_key
    if not stale and progress.get("completed"):
        return set(progress["surviving_local_ids"])

    if not stale:
        completed_tile_indices = set(progress.get("completed_tile_indices", []))
        current_tile_index = progress.get("current_tile_index", 0)
        last_completed_z_in_tile = progress.get("last_completed_z_in_tile", -1)
        surviving: set[int] = set(progress.get("surviving_local_ids_partial", []))
    else:
        completed_tile_indices = set()
        current_tile_index = 0
        last_completed_z_in_tile = -1
        surviving = set()

    mask_bytes_per_plane = ny * nx * mask_arr.dtype.itemsize
    mask_mmap_len = len(mask_arr._mmap)
    label_bytes_per_plane = ny * nx * label_arr.dtype.itemsize
    label_mmap_len = len(label_arr._mmap)
    page_size = mmap.PAGESIZE

    for tile_idx, (y0, y1, x0, x1) in enumerate(tiles):
        if tile_idx in completed_tile_indices:
            continue
        z_start = last_completed_z_in_tile + 1 if tile_idx == current_tile_index else 0
        halo_y0 = max(0, y0 - halo_y)
        halo_y1 = min(ny, y1 + halo_y)
        halo_x0 = max(0, x0 - halo_x)
        halo_x1 = min(nx, x1 + halo_x)

        for z0 in range(z_start, nz, chunk_z_voxels):
            z1 = min(z0 + chunk_z_voxels, nz)
            halo_z0 = max(0, z0 - halo_z)
            halo_z1 = min(nz, z1 + halo_z)

            mask_chunk = np.array(mask_arr[halo_z0:halo_z1, halo_y0:halo_y1, halo_x0:halo_x1]) != 0
            dist = distance_transform_edt(~mask_chunk, sampling=(res_z, res_y, res_x))
            in_contact_haloed = dist <= threshold_nm
            del mask_chunk, dist
            # Slice the halo-padded result back down to just this tile's own
            # [z0:z1, y0:y1, x0:x1) core; the halo only existed to make
            # THIS slice correct.
            in_contact = in_contact_haloed[
                z0 - halo_z0: z0 - halo_z0 + (z1 - z0),
                y0 - halo_y0: y0 - halo_y0 + (y1 - y0),
                x0 - halo_x0: x0 - halo_x0 + (x1 - x0),
            ]
            del in_contact_haloed

            label_chunk = np.array(label_arr[z0:z1, y0:y1, x0:x1])
            keep = (label_chunk == target_local_id) | in_contact
            label_chunk[~keep] = 0
            label_arr[z0:z1, y0:y1, x0:x1] = label_chunk
            surviving.update(int(v) for v in np.unique(label_chunk) if v != 0)
            del label_chunk, keep, in_contact

            byte_start = halo_z0 * mask_bytes_per_plane
            byte_end = min(halo_z1 * mask_bytes_per_plane, mask_mmap_len)
            aligned_start = (byte_start // page_size) * page_size
            if byte_end > aligned_start:
                mask_arr._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)

            label_arr.flush()
            byte_start = z0 * label_bytes_per_plane
            byte_end = min(z1 * label_bytes_per_plane, label_mmap_len)
            aligned_start = (byte_start // page_size) * page_size
            if byte_end > aligned_start:
                label_arr._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)

            io_utils.save_progress(prog_path, {
                "request_key": internal_key,
                "completed_tile_indices": sorted(completed_tile_indices),
                "current_tile_index": tile_idx,
                "last_completed_z_in_tile": z1 - 1,
                "completed": False,
                "surviving_local_ids_partial": sorted(surviving),
            })

        completed_tile_indices.add(tile_idx)

    io_utils.save_progress(prog_path, {
        "request_key": internal_key,
        "completed_tile_indices": sorted(completed_tile_indices),
        "current_tile_index": len(tiles),
        "last_completed_z_in_tile": -1,
        "completed": True,
        "surviving_local_ids": sorted(surviving),
    })
    return surviving
