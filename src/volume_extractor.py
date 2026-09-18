# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
Volume extraction module for downloading EM and segmentation subvolumes and building binary masks.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import os
import shutil
import time

import numpy as np

from . import coordinate_utils as cu
from . import io_utils

logger = logging.getLogger(__name__)


class ExtractionSafetyError(Exception):
    pass


class AstrocyteNotFoundError(Exception):
    pass


def estimate_bytes(n_voxels: int, em_itemsize: int, seg_itemsize: int, mask_itemsize: int = 1) -> int:
    """
    Peak in-memory bytes needed while both EM and segmentation blocks (plus
    the derived mask) are held at once for a region of n_voxels voxels.
    """
    return n_voxels * (em_itemsize + seg_itemsize + mask_itemsize)


def physical_dims_um(voxel_shape: np.ndarray, resolution_nm: np.ndarray) -> np.ndarray:
    return voxel_shape * resolution_nm / 1000.0


def effective_max_voxels(max_voxels: int, max_memory_gb: float, em_itemsize: int, seg_itemsize: int, extra_bytes_per_voxel: int = 0) -> int:
    bytes_per_voxel = em_itemsize + seg_itemsize + 1 + extra_bytes_per_voxel  # +1 for the mask
    memory_limited_voxels = int((max_memory_gb * 1e9) // bytes_per_voxel)
    return min(max_voxels, memory_limited_voxels)


def log_extraction_request(voxel_shape: np.ndarray, resolution_nm: np.ndarray, em_itemsize: int, seg_itemsize: int):
    n_voxels = int(np.prod(voxel_shape))
    dims_um = physical_dims_um(voxel_shape, resolution_nm)
    total_bytes = estimate_bytes(n_voxels, em_itemsize, seg_itemsize)
    logger.info(
        "Requested region: voxel_shape(xyz)=%s, physical size=%.2f x %.2f x %.2f um, "
        "voxels=%d, estimated peak memory=%.3f GB",
        voxel_shape.tolist(), dims_um[0], dims_um[1], dims_um[2], n_voxels, total_bytes / 1e9,
    )
    return n_voxels, total_bytes


def build_mask(seg_zyx: np.ndarray, root_id: int) -> np.ndarray:
    return (seg_zyx == root_id).astype(np.uint8)


def validate_counts(positive_voxels: int, total_voxels: int, root_id: int, occupancy_warn_threshold: float = 1e-4):
    if positive_voxels == 0:
        raise AstrocyteNotFoundError(
            f"root_id {root_id} did not appear anywhere in the extracted segmentation volume. "
            "This indicates a coordinate-system or materialization-version mismatch, not an "
            "empty astrocyte -- do not silently continue."
        )
    occupancy = positive_voxels / total_voxels
    if occupancy < occupancy_warn_threshold:
        logger.warning(
            "Astrocyte occupancy is suspiciously small: %d / %d voxels (%.6f). "
            "Check margin/bbox settings if this is unexpected.",
            positive_voxels, total_voxels, occupancy,
        )
    return occupancy


def validate_extraction(image_zyx: np.ndarray, mask_zyx: np.ndarray, root_id: int, occupancy_warn_threshold: float = 1e-4):
    if image_zyx.shape != mask_zyx.shape:
        raise ValueError(f"image.shape {image_zyx.shape} != mask.shape {mask_zyx.shape}")
    if mask_zyx.max() > 1:
        raise ValueError(f"mask contains values > 1 (max={mask_zyx.max()})")
    positive_voxels = int(mask_zyx.sum())
    occupancy = validate_counts(positive_voxels, mask_zyx.size, root_id, occupancy_warn_threshold)
    return positive_voxels, occupancy


def extract_full_bbox(client, root_id, bbox_min_vx, bbox_max_vx):
    em_raw = client.download_em(bbox_min_vx, bbox_max_vx)
    seg_raw = client.download_segmentation(bbox_min_vx, bbox_max_vx)

    image_zyx = cu.xyz_to_zyx(em_raw)
    seg_zyx = cu.xyz_to_zyx(seg_raw)
    mask_zyx = build_mask(seg_zyx, root_id)
    return image_zyx, mask_zyx


def retry_with_backoff(fn, max_attempts: int = 5, base_delay_s: float = 8.0, description: str = "operation"):
    """Retry a flaky network call with linear backoff. The MICrONS and CAVE
    backend has been observed to return transient 502, 503, and 504 errors under
    load; worth retrying rather than aborting a long chunked run.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            delay = base_delay_s * (attempt + 1)
            logger.warning("%s failed (attempt %d/%d): %s. Retrying in %.0fs...", description, attempt + 1, max_attempts, e, delay)
            time.sleep(delay)
    raise last_exc


def safe_chunk_z_voxels(bbox_min_vx, bbox_max_vx, budget_voxels: int, configured_max_chunk_z: int, safety_factor: float = 0.8) -> int:
    """
    Z-slab thickness that keeps one XY-footprint-sized slab within
    budget_voxels regardless of XY extent; a fixed config chunk_z_voxels
    alone is only safe for small footprints.
    """
    nx = int(bbox_max_vx[0] - bbox_min_vx[0])
    ny = int(bbox_max_vx[1] - bbox_min_vx[1])
    per_slice_voxels = max(nx * ny, 1)
    dynamic_max_chunk_z = max(1, int((budget_voxels * safety_factor) // per_slice_voxels))
    return max(1, min(configured_max_chunk_z, dynamic_max_chunk_z))


def _download_em_and_seg(client, slab_min, slab_max, retry_max_attempts, retry_base_delay_s, slab_start, slab_end):
    """
    Fetch one slab's EM and segmentation concurrently (two independent
    downloads; no reason to serialize them) and return both raw arrays.
    Runs inside a download-pool worker thread; see extract_chunked_streaming.
    """
    with cf.ThreadPoolExecutor(max_workers=2) as pair_pool:
        em_future = pair_pool.submit(
            retry_with_backoff, lambda: client.download_em(slab_min, slab_max),
            max_attempts=retry_max_attempts, base_delay_s=retry_base_delay_s,
            description=f"download_em z=[{slab_start},{slab_end})",
        )
        seg_future = pair_pool.submit(
            retry_with_backoff, lambda: client.download_segmentation(slab_min, slab_max),
            max_attempts=retry_max_attempts, base_delay_s=retry_base_delay_s,
            description=f"download_segmentation z=[{slab_start},{slab_end})",
        )
        return em_future.result(), seg_future.result()


def extract_chunked_streaming(client, root_id, bbox_min_vx, bbox_max_vx, chunk_z_voxels: int, out_dir: str, image_path: str, mask_path: str, retry_max_attempts: int = 5, retry_base_delay_s: float = 8.0, keep_slabs: bool = False):
    """Downloads in Z-slabs, resumable via progress.json checkpointed before
    each next slab; slabs processed one at a time (own EM+seg fetched
    concurrently) to avoid oversubscribing the HTTP pool, keeping peak
    memory to one slab. Merges into image.tif and mask.tif once all slabs are
    done. Returns (shape_zyx, positive_voxels, occupancy)."""
    z0, z1 = int(bbox_min_vx[2]), int(bbox_max_vx[2])
    prog_path = io_utils.progress_path(out_dir)

    request_key = {
        "root_id": int(root_id),
        "bbox_min_vx": [int(v) for v in bbox_min_vx],
        "bbox_max_vx": [int(v) for v in bbox_max_vx],
        "chunk_z_voxels": int(chunk_z_voxels),
    }

    progress = io_utils.load_progress(prog_path)
    if progress is not None and progress.get("request_key") != request_key:
        logger.warning(
            "Existing progress file at %s does not match the current request "
            "(root_id/bbox/chunk size changed) -- restarting this astrocyte from scratch.",
            prog_path,
        )
        progress = None

    if progress is None:
        progress = {
            "request_key": request_key,
            "completed_slabs": [],
            "positive_voxels": 0,
            "total_voxels": 0,
            "download_complete": False,
            "merged": False,
            "started_at": time.time(),
            "last_updated": time.time(),
        }
        io_utils.save_progress(prog_path, progress)
    elif progress.get("merged"):
        logger.info("root_id %d already fully extracted and merged per progress file; reusing result.", root_id)
        shape = io_utils.peek_tiff_shape(image_path)
        occupancy = progress["positive_voxels"] / progress["total_voxels"]
        return tuple(shape), progress["positive_voxels"], occupancy

    image_slab_dir = io_utils.slab_dir(out_dir, "image")
    mask_slab_dir = io_utils.slab_dir(out_dir, "mask")
    completed_starts = {s[0] for s in progress["completed_slabs"]}

    if not progress["download_complete"]:
        n_total_slabs = -(-(z1 - z0) // chunk_z_voxels)
        for slab_start in range(z0, z1, chunk_z_voxels):
            slab_end = min(slab_start + chunk_z_voxels, z1)
            if slab_start in completed_starts:
                continue

            slab_min = np.array([bbox_min_vx[0], bbox_min_vx[1], slab_start])
            slab_max = np.array([bbox_max_vx[0], bbox_max_vx[1], slab_end])

            em_raw, seg_raw = _download_em_and_seg(client, slab_min, slab_max, retry_max_attempts, retry_base_delay_s, slab_start, slab_end)

            image_zyx_slab = cu.xyz_to_zyx(em_raw)
            seg_zyx_slab = cu.xyz_to_zyx(seg_raw)
            mask_zyx_slab = build_mask(seg_zyx_slab, root_id)

            if image_zyx_slab.shape != mask_zyx_slab.shape:
                raise ValueError(f"slab image.shape {image_zyx_slab.shape} != mask.shape {mask_zyx_slab.shape}")

            io_utils.write_slab(io_utils.slab_file_path(image_slab_dir, slab_start, slab_end), image_zyx_slab, np.uint8)
            io_utils.write_slab(io_utils.slab_file_path(mask_slab_dir, slab_start, slab_end), mask_zyx_slab, np.uint8)

            progress["completed_slabs"].append([slab_start, slab_end])
            progress["positive_voxels"] += int(mask_zyx_slab.sum())
            progress["total_voxels"] += int(mask_zyx_slab.size)
            io_utils.save_progress(prog_path, progress)

            logger.info(
                "chunked extraction: processed z=[%d, %d) of [%d, %d) (%d/%d slabs), running positive_voxels=%d",
                slab_start, slab_end, z0, z1, len(progress["completed_slabs"]), n_total_slabs, progress["positive_voxels"],
            )

        progress["download_complete"] = True
        io_utils.save_progress(prog_path, progress)
    else:
        logger.info("All slabs already downloaded per progress file; skipping straight to merge.")

    ny = int(bbox_max_vx[1] - bbox_min_vx[1])
    nx = int(bbox_max_vx[0] - bbox_min_vx[0])
    logger.info("Merging %d slab files into final image.tif / mask.tif ...", len(progress["completed_slabs"]))
    n_slices = io_utils.merge_slabs_to_stack(image_slab_dir, image_path, progress["completed_slabs"], np.uint8, (ny, nx))
    io_utils.merge_slabs_to_stack(mask_slab_dir, mask_path, progress["completed_slabs"], np.uint8, (ny, nx))

    shape_zyx = io_utils.peek_tiff_shape(image_path)
    if shape_zyx[0] != n_slices:
        raise ValueError(f"Merged image has {shape_zyx[0]} slices, expected {n_slices}")

    occupancy = validate_counts(progress["positive_voxels"], progress["total_voxels"], root_id)

    progress["merged"] = True
    io_utils.save_progress(prog_path, progress)

    if keep_slabs:
        logger.info("keep_slabs=True: leaving _slabs/ in place instead of deleting it.")
    else:
        shutil.rmtree(os.path.join(out_dir, "_slabs"), ignore_errors=True)

    return tuple(shape_zyx), progress["positive_voxels"], occupancy


INTERACTION_LABEL_DTYPE = np.uint16  # default/fallback for callers that don't select a dtype explicitly

_INTERACTION_LABEL_DTYPE_CANDIDATES = (np.uint8, np.uint16, np.uint32)


def select_interaction_label_dtype(n_ids: int) -> np.dtype:
    """Smallest unsigned dtype holding n_ids distinct local ids (1..n_ids;
    0=background); checked live per astrocyte rather than assumed fixed."""
    for dtype in _INTERACTION_LABEL_DTYPE_CANDIDATES:
        if n_ids <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ExtractionSafetyError(
        f"Number of interacting cells ({n_ids}) exceeds uint32 local-label capacity."
    )


def build_interaction_id_map(root_id: int, interacting_root_ids) -> dict:
    """
    root_id -> local-integer-id map for the filtered label volume: target
    astrocyte = 1, plus interacting_root_ids; fixed up front (not grown
    while scanning slabs) so slab-scanning code can assume a stable id set.
    """
    mapping = {int(root_id): 1}
    for rid in sorted(set(int(r) for r in interacting_root_ids)):
        if rid not in mapping:
            mapping[rid] = len(mapping) + 1
    select_interaction_label_dtype(len(mapping))  # raises if absurdly large
    return mapping


def remap_slab_to_known_ids(seg_zyx_slab: np.ndarray, sorted_root_ids: np.ndarray, sorted_local_ids: np.ndarray) -> np.ndarray:
    """
    Remaps to a fixed, known-in-advance id set (no np.unique() needed):
    any segment id not in sorted_root_ids maps to 0.
    """
    idx = np.searchsorted(sorted_root_ids, seg_zyx_slab)
    idx = np.minimum(idx, len(sorted_root_ids) - 1)
    is_match = sorted_root_ids[idx] == seg_zyx_slab
    return np.where(is_match, sorted_local_ids[idx], 0)


def extract_interaction_labels_chunked(client, root_id, bbox_min_vx, bbox_max_vx, chunk_z_voxels: int, out_dir: str, label_path: str, id_map: dict, dtype=INTERACTION_LABEL_DTYPE, retry_max_attempts: int = 5, retry_base_delay_s: float = 8.0, keep_slabs: bool = False):
    """
    Builds interaction_labels.tif (target + id_map cells only),
    re-downloading segmentation in Z-slabs. Resumable via its own progress
    file, keyed on id_map's root_id set so a changed interaction_analysis.json
    restarts cleanly. Sequential per-slab (segmentation only, no EM to
    pair). Returns total Z slices written.
    """
    z0, z1 = int(bbox_min_vx[2]), int(bbox_max_vx[2])
    prog_path = io_utils.interaction_label_progress_path(out_dir)

    sorted_root_ids_list = sorted(int(k) for k in id_map)
    request_key = {
        "root_id": int(root_id),
        "bbox_min_vx": [int(v) for v in bbox_min_vx],
        "bbox_max_vx": [int(v) for v in bbox_max_vx],
        "chunk_z_voxels": int(chunk_z_voxels),
        "id_map_root_ids": sorted_root_ids_list,
    }

    progress = io_utils.load_progress(prog_path)
    if progress is not None and progress.get("request_key") != request_key:
        logger.warning(
            "Existing interaction-label progress at %s does not match the current request "
            "(bbox/chunking/interacting-cell-set changed) -- restarting from scratch.",
            prog_path,
        )
        progress = None

    if progress is None:
        progress = {"request_key": request_key, "completed_slabs": [], "download_complete": False, "merged": False}
        io_utils.save_progress(prog_path, progress)
    elif progress.get("merged"):
        logger.info("root_id %d: interaction label volume already built per progress file; reusing it.", root_id)
        return io_utils.peek_tiff_shape(label_path)[0]

    label_slab_dir = io_utils.slab_dir(out_dir, "interaction_label")
    completed_starts = {s[0] for s in progress["completed_slabs"]}

    sorted_root_ids = np.array(sorted_root_ids_list, dtype=np.int64)
    sorted_local_ids = np.array([id_map[k] for k in sorted_root_ids_list], dtype=dtype)

    if not progress["download_complete"]:
        n_total_slabs = -(-(z1 - z0) // chunk_z_voxels)
        for slab_start in range(z0, z1, chunk_z_voxels):
            slab_end = min(slab_start + chunk_z_voxels, z1)
            if slab_start in completed_starts:
                continue

            slab_min = np.array([bbox_min_vx[0], bbox_min_vx[1], slab_start])
            slab_max = np.array([bbox_max_vx[0], bbox_max_vx[1], slab_end])

            seg_raw = retry_with_backoff(
                lambda: client.download_segmentation(slab_min, slab_max),
                max_attempts=retry_max_attempts, base_delay_s=retry_base_delay_s,
                description=f"download_segmentation (interaction labels) z=[{slab_start},{slab_end})",
            )
            seg_zyx_slab = cu.xyz_to_zyx(seg_raw)
            label_zyx_slab = remap_slab_to_known_ids(seg_zyx_slab, sorted_root_ids, sorted_local_ids)

            io_utils.write_slab(io_utils.slab_file_path(label_slab_dir, slab_start, slab_end), label_zyx_slab, dtype)

            progress["completed_slabs"].append([slab_start, slab_end])
            io_utils.save_progress(prog_path, progress)

            logger.info(
                "interaction-label extraction: processed z=[%d, %d) of [%d, %d) (%d/%d slabs)",
                slab_start, slab_end, z0, z1, len(progress["completed_slabs"]), n_total_slabs,
            )

        progress["download_complete"] = True
        io_utils.save_progress(prog_path, progress)
    else:
        logger.info("All interaction-label slabs already downloaded per progress file; skipping straight to merge.")

    ny = int(bbox_max_vx[1] - bbox_min_vx[1])
    nx = int(bbox_max_vx[0] - bbox_min_vx[0])
    logger.info("Merging %d slab files into final interaction_labels.tif ...", len(progress["completed_slabs"]))
    n_slices = io_utils.merge_slabs_to_stack(label_slab_dir, label_path, progress["completed_slabs"], dtype, (ny, nx))

    shape_zyx = io_utils.peek_tiff_shape(label_path)
    if shape_zyx[0] != n_slices:
        raise ValueError(f"Merged interaction-label volume has {shape_zyx[0]} slices, expected {n_slices}")

    progress["merged"] = True
    io_utils.save_progress(prog_path, progress)

    if keep_slabs:
        logger.info("keep_slabs=True: leaving _slabs/interaction_label in place instead of deleting it.")
    else:
        shutil.rmtree(label_slab_dir, ignore_errors=True)

    return n_slices


def enumerate_root_ids_in_bbox(client, bbox_min_vx, bbox_max_vx, chunk_z_voxels: int, min_voxels: int, retry_max_attempts: int = 5, retry_base_delay_s: float = 8.0, progress_path: str | None = None) -> dict:
    """
    Every root_id with >=min_voxels in the bbox's segmentation, mapped
    to voxel count, discovered via np.unique(return_counts=True)
    per Z-chunk, unlike extract_interaction_labels_chunked's known id set.
    Resumable via progress_path; None skips persistence.
    """
    z0, z1 = int(bbox_min_vx[2]), int(bbox_max_vx[2])
    request_key = {
        "bbox_min_vx": [int(v) for v in bbox_min_vx],
        "bbox_max_vx": [int(v) for v in bbox_max_vx],
        "chunk_z_voxels": int(chunk_z_voxels),
    }

    progress = io_utils.load_progress(progress_path) if progress_path else None
    if progress is not None and progress.get("request_key") != request_key:
        logger.warning(
            "Existing bbox-enumeration progress at %s does not match the current request "
            "(bbox/chunking changed) -- restarting from scratch.",
            progress_path,
        )
        progress = None

    if progress is None:
        progress = {"request_key": request_key, "completed_slabs": [], "counts": {}}
        if progress_path:
            io_utils.save_progress(progress_path, progress)
    elif progress.get("done"):
        logger.info("bbox-segmentation enumeration already complete per progress file; reusing it.")
        return {int(k): v for k, v in progress["counts"].items() if v >= min_voxels}

    completed_starts = {s[0] for s in progress["completed_slabs"]}
    counts: dict[int, int] = {int(k): v for k, v in progress["counts"].items()}

    n_total_slabs = -(-(z1 - z0) // chunk_z_voxels)
    for slab_start in range(z0, z1, chunk_z_voxels):
        slab_end = min(slab_start + chunk_z_voxels, z1)
        if slab_start in completed_starts:
            continue

        slab_min = np.array([bbox_min_vx[0], bbox_min_vx[1], slab_start])
        slab_max = np.array([bbox_max_vx[0], bbox_max_vx[1], slab_end])

        seg_raw = retry_with_backoff(
            lambda: client.download_segmentation(slab_min, slab_max),
            max_attempts=retry_max_attempts, base_delay_s=retry_base_delay_s,
            description=f"download_segmentation (bbox enumeration) z=[{slab_start},{slab_end})",
        )
        ids, slab_counts = np.unique(seg_raw, return_counts=True)
        for rid, cnt in zip(ids.tolist(), slab_counts.tolist()):
            if rid == 0:
                continue  # background, never a real root_id
            counts[rid] = counts.get(rid, 0) + cnt

        progress["completed_slabs"].append([slab_start, slab_end])
        progress["counts"] = counts
        if progress_path:
            io_utils.save_progress(progress_path, progress)

        logger.info(
            "bbox-segmentation enumeration: processed z=[%d, %d) of [%d, %d) (%d/%d slabs), "
            "%d distinct root_ids so far",
            slab_start, slab_end, z0, z1, len(progress["completed_slabs"]), n_total_slabs, len(counts),
        )

    progress["done"] = True
    if progress_path:
        io_utils.save_progress(progress_path, progress)

    return {rid: cnt for rid, cnt in counts.items() if cnt >= min_voxels}


def _compute_extraction_budget(client, extraction_cfg: dict) -> int:
    """
    Effective voxel budget (em+seg+mask) for the given config. Factored
    out so extract_astrocyte's resume shortcut and its fresh-mesh-lookup
    path compute this identically.
    """
    return effective_max_voxels(
        extraction_cfg["max_voxels"],
        extraction_cfg["max_memory_gb"],
        client.em_dtype_itemsize,
        client.seg_dtype_itemsize,
    )


def extract_astrocyte(client, root_id: int, extraction_cfg: dict, out_paths: dict | None = None):
    """
    Full extraction workflow for one root_id. out_paths (image and mask
    paths) is required for 'chunked' mode, which streams to disk and
    returns (None, None, info) with info['streamed_to_disk']=True. Resume
    shortcut: an existing _progress.json for this root_id re-reads the bbox
    from it instead of recomputing from the mesh.
    """
    if extraction_cfg["mode"] == "chunked" and out_paths:
        out_dir = os.path.dirname(out_paths["image"])
        progress = io_utils.load_progress(io_utils.progress_path(out_dir))
        if (
            progress is not None
            and progress["request_key"]["root_id"] == root_id
        ):
            req = progress["request_key"]
            bbox_min_vx = np.array(req["bbox_min_vx"])
            bbox_max_vx = np.array(req["bbox_max_vx"])
            budget = _compute_extraction_budget(client, extraction_cfg)
            chunk_z_voxels = safe_chunk_z_voxels(bbox_min_vx, bbox_max_vx, budget, extraction_cfg.get("chunk_z_voxels", 64))
            logger.info(
                "root_id %d: found existing _progress.json; reusing its bbox and resuming without a mesh lookup "
                "(chunk_z_voxels recomputed from current config: %d; stored progress had %d%s).",
                root_id, chunk_z_voxels, req["chunk_z_voxels"],
                "" if chunk_z_voxels == req["chunk_z_voxels"] else " -- mismatch will trigger a clean restart of this astrocyte's slabs",
            )
            shape_zyx, positive_voxels, occupancy = extract_chunked_streaming(
                client, root_id, bbox_min_vx, bbox_max_vx, chunk_z_voxels, out_dir,
                out_paths["image"], out_paths["mask"],
                keep_slabs=extraction_cfg.get("keep_slabs", False),
            )
            info = {
                "voxel_bbox_min": bbox_min_vx,
                "voxel_bbox_max": bbox_max_vx,
                "mesh_bbox_min_nm": None,
                "mesh_bbox_max_nm": None,
                "was_clipped": None,
                "positive_voxels": positive_voxels,
                "occupancy": occupancy,
                "streamed_to_disk": True,
                "shape_zyx": shape_zyx,
            }
            return None, None, info

    vertices_nm = client.get_mesh_vertices_nm(root_id)

    bbox = cu.mesh_bbox_to_volume_bbox(
        vertices_nm,
        client.resolution_nm,
        extraction_cfg["margin_um"],
        (client.bounds_min_vx, client.bounds_max_vx),
    )
    bbox_min_vx, bbox_max_vx = bbox["min"], bbox["max"]

    test_max_z_voxels = extraction_cfg.get("test_max_z_voxels")
    if test_max_z_voxels is not None:
        bbox_max_vx = bbox_max_vx.copy()
        bbox_max_vx[2] = min(bbox_max_vx[2], bbox_min_vx[2] + test_max_z_voxels)
        logger.info(
            "root_id %d: test_max_z_voxels=%d set -- clipping bbox Z extent to [%d, %d) for a fast test run.",
            root_id, test_max_z_voxels, bbox_min_vx[2], bbox_max_vx[2],
        )

    voxel_shape = cu.voxel_shape(bbox_min_vx, bbox_max_vx)
    n_voxels, _ = log_extraction_request(
        voxel_shape, client.resolution_nm, client.em_dtype_itemsize, client.seg_dtype_itemsize
    )

    budget = _compute_extraction_budget(client, extraction_cfg)

    mode = extraction_cfg["mode"]

    if n_voxels > budget:
        if mode == "full_bbox":
            dims_um = physical_dims_um(voxel_shape, client.resolution_nm)
            raise ExtractionSafetyError(
                f"Requested region for root_id {root_id} is {n_voxels:,} voxels "
                f"({dims_um[0]:.1f} x {dims_um[1]:.1f} x {dims_um[2]:.1f} um), which exceeds the "
                f"configured safety budget of {budget:,} voxels (max_voxels={extraction_cfg['max_voxels']}, "
                f"max_memory_gb={extraction_cfg['max_memory_gb']}). Refusing to download in 'full_bbox' mode. "
                "Use extraction.mode='chunked' in config.yaml, or raise the limits, or reduce margin_um."
            )
        elif mode == "chunked":
            logger.info(
                "root_id %d bbox (%d voxels) exceeds full-load budget (%d voxels); using chunked extraction.",
                root_id, n_voxels, budget,
            )
        else:
            raise ValueError(f"Unknown extraction mode '{mode}'")

    base_info = {
        "voxel_bbox_min": bbox_min_vx,
        "voxel_bbox_max": bbox_max_vx,
        "mesh_bbox_min_nm": bbox["mesh_bbox_min_nm"],
        "mesh_bbox_max_nm": bbox["mesh_bbox_max_nm"],
        "was_clipped": bool(np.any(bbox_min_vx != bbox["unclipped_min"]) or np.any(bbox_max_vx != bbox["unclipped_max"])),
    }

    if mode == "chunked":
        if not out_paths or "image" not in out_paths or "mask" not in out_paths:
            raise ValueError("chunked mode requires out_paths={'image': ..., 'mask': ...} to stream to disk")
        chunk_z_voxels = safe_chunk_z_voxels(bbox_min_vx, bbox_max_vx, budget, extraction_cfg.get("chunk_z_voxels", 64))
        logger.info("chunked extraction: using chunk_z_voxels=%d (dynamically bounded by memory budget)", chunk_z_voxels)
        out_dir = os.path.dirname(out_paths["image"])
        shape_zyx, positive_voxels, occupancy = extract_chunked_streaming(
            client, root_id, bbox_min_vx, bbox_max_vx, chunk_z_voxels, out_dir, out_paths["image"], out_paths["mask"],
            keep_slabs=extraction_cfg.get("keep_slabs", False),
        )
        info = {**base_info, "positive_voxels": positive_voxels, "occupancy": occupancy, "streamed_to_disk": True, "shape_zyx": shape_zyx}
        return None, None, info

    image_zyx, mask_zyx = extract_full_bbox(client, root_id, bbox_min_vx, bbox_max_vx)
    positive_voxels, occupancy = validate_extraction(image_zyx, mask_zyx, root_id)
    info = {**base_info, "positive_voxels": positive_voxels, "occupancy": occupancy, "streamed_to_disk": False}
    return image_zyx, mask_zyx, info
