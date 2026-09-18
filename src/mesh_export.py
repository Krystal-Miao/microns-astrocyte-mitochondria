# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
3D mesh export and local marching-cubes reconstruction for interacting cells.
"""

from __future__ import annotations

import json
import mmap
import os

import numpy as np
import tifffile

from . import io_utils


def _mesh_category(entry: dict) -> str:
    """
    4-way bucket: target astrocyte, other astrocyte, neuron (any
    compartment), dendrite fallback for anything unclassified.
    """
    if entry.get("is_target_astrocyte"):
        return "target_astrocyte"
    if entry.get("classification_system") in ("excitatory_neuron", "inhibitory_neuron"):
        return "neuron"
    if entry.get("cell_type") == "astrocyte":
        return "astrocyte"
    return "dendrite"


def fetch_interacting_meshes(client, run_dir: str, mesh_dir_name: str = "meshes", skip_existing: bool = True):
    """
    Fetch+save one .ply per root_id in interaction_label_map.json, under
    "run_dir/<mesh_dir_name>/{target_astrocyte,astrocyte,dendrite}/<id>.ply".
    skip_existing=True treats a complete file as permanently done (mesh
    geometry for a fixed root_id and segmentation never changes). Returns
    (root_id, category, status) tuples.
    """
    id_map = io_utils.load_interaction_label_map(run_dir)
    if id_map is None:
        raise ValueError(f"No interaction_label_map.json under {run_dir} -- run extract-interaction-labels first.")

    mesh_root = os.path.join(run_dir, mesh_dir_name)
    results = []
    for entry in id_map:
        root_id = entry["root_id"]
        category = _mesh_category(entry)
        cat_dir = os.path.join(mesh_root, category)
        os.makedirs(cat_dir, exist_ok=True)
        out_path = os.path.join(cat_dir, f"{root_id}.ply")

        if skip_existing and os.path.isfile(out_path):
            results.append((root_id, category, "already saved"))
            continue

        try:
            mesh = client.get_mesh(root_id)
            # Atomic (temp file + rename) so a kill mid-write can't leave a
            # corrupt or truncated .ply that a later skip_existing run would
            # then wrongly treat as done.
            tmp_path = out_path + ".tmp"
            mesh.save(tmp_path)
            os.replace(tmp_path, out_path)
            results.append((root_id, category, f"OK ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)"))
        except Exception as e:
            results.append((root_id, category, f"FAILED: {e}"))

    return results


def _write_ply_ascii(path: str, vertices: np.ndarray, faces: np.ndarray):
    """
    Minimal ASCII PLY writer (vertices only, no normals or color); avoids
    adding a trimesh dependency for what's otherwise a two-array dump.
    vertices: (N, 3) float, already in (x, y, z) nm order. faces: (M, 3) int
    vertex indices.
    """
    lines = [
        "ply", "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{x:.3f} {y:.3f} {z:.3f}" for x, y, z in vertices)
    lines.extend(f"3 {a} {b} {c}" for a, b, c in faces)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    os.replace(tmp_path, path)


def _build_downsampled_label_volume(label_path: str, downsample: int, chunk_z: int = 64) -> np.ndarray:
    """
    Reads interaction_labels.tif in Z-chunks (never the full, often
    >100GB, volume in RAM), returns it downsampled `downsample`x in Y and X only
    (Z stays native); small enough at 10x to hold whole.
    """
    label_arr = tifffile.memmap(label_path)
    nz, ny, nx = label_arr.shape
    bytes_per_plane = ny * nx * label_arr.dtype.itemsize
    mmap_len = len(label_arr._mmap)
    page_size = mmap.PAGESIZE

    fd = os.open(label_path, os.O_RDONLY)
    chunks = []
    try:
        for z0 in range(0, nz, chunk_z):
            z1 = min(z0 + chunk_z, nz)
            # .copy() is required: ::downsample slicing returns a VIEW, so
            # without it every chunk keeps its full-res parent array alive
            # (confirmed swapping before this fix).
            full_res_chunk = np.array(label_arr[z0:z1])
            chunks.append(full_res_chunk[:, ::downsample, ::downsample].copy())
            del full_res_chunk

            # Drop unneeded pages immediately (madvise + fadvise DONTNEED):
            # this host's low-memory guard reacts to raw `free`, and a full
            # pass over a 190GB file got the process killed without this.
            byte_start = z0 * bytes_per_plane
            byte_end = min(z1 * bytes_per_plane, mmap_len)
            aligned_start = (byte_start // page_size) * page_size
            if byte_end > aligned_start:
                label_arr._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)
                os.posix_fadvise(fd, aligned_start, byte_end - aligned_start, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

    return np.concatenate(chunks, axis=0)


def reconstruct_local_meshes(
    out_dir: str,
    run_dir: str,
    downsample: int = 10,
    mesh_dir_name: str | None = None,
    chunk_z: int = 64,
    min_voxels: int = 4,
    skip_existing: bool = True,
):
    """
    Network-free alternative to fetch_interacting_meshes: marching cubes
    on interaction_labels.tif, truncated at the bbox edge, unlike CAVE's
    official mesh. A thin fragment can vanish under Y and X downsampling
    (-> "SKIPPED"). Returns (root_id, category, status) tuples.
    """
    from scipy import ndimage
    from skimage import measure

    id_map = io_utils.load_interaction_label_map(run_dir)
    if id_map is None:
        raise ValueError(f"No interaction_label_map.json under {run_dir} -- run extract-interaction-labels first.")

    label_path = io_utils.interaction_labels_path(run_dir)
    if not os.path.isfile(label_path):
        raise ValueError(f"No interaction_labels.tif under {run_dir} -- run extract-interaction-labels first.")

    with open(io_utils.metadata_path(out_dir)) as f:
        metadata = json.load(f)
    # XYZ order, per coordinate_utils.py's convention.
    voxel_bbox_min_xyz = np.array(metadata["voxel_bbox"]["min"], dtype=np.float64)
    resolution_nm_xyz = np.array(metadata["seg_resolution_nm"], dtype=np.float64)

    if mesh_dir_name is None:
        mesh_dir_name = f"meshes_local_ds{downsample}"
    mesh_root = os.path.join(run_dir, mesh_dir_name)

    # Skip the (slow, chunked-disk-read) volume build entirely if every
    # entry is already saved; common on a resumed or rerun.
    results = []
    pending = []
    for e in id_map:
        category = _mesh_category(e)
        out_path = os.path.join(mesh_root, category, f"{e['root_id']}.ply")
        if skip_existing and os.path.isfile(out_path):
            results.append((e["root_id"], category, "already saved"))
        else:
            pending.append(e)
    if not pending:
        return results

    ds_labels = _build_downsampled_label_volume(label_path, downsample, chunk_z=chunk_z)
    # spacing is nm-per-grid-step in (Z, Y, X) order, matching ds_labels'
    # axis order (Z untouched, Y and X strided by `downsample`).
    res_x, res_y, res_z = resolution_nm_xyz
    bbox_min_x, bbox_min_y, bbox_min_z = voxel_bbox_min_xyz
    spacing_zyx = np.array([res_z, res_y * downsample, res_x * downsample])

    # One linear pass over the whole (already-downsampled) volume finds
    # every label's bounding box at once; scanning the full array
    # per-entry instead would cost one full-array pass per label for no
    # reason.
    max_local_id = max(e["local_id"] for e in id_map)
    bboxes = ndimage.find_objects(ds_labels, max_label=max_local_id)

    for entry in pending:
        root_id = entry["root_id"]
        local_id = entry["local_id"]
        category = _mesh_category(entry)
        cat_dir = os.path.join(mesh_root, category)
        os.makedirs(cat_dir, exist_ok=True)
        out_path = os.path.join(cat_dir, f"{root_id}.ply")

        try:
            bbox = bboxes[local_id - 1] if local_id - 1 < len(bboxes) else None
            if bbox is None:
                results.append((root_id, category, "SKIPPED: no voxels at this downsample"))
                continue
            z_slc, y_slc, x_slc = bbox
            crop = (ds_labels[z_slc, y_slc, x_slc] == local_id)
            if crop.sum() < min_voxels:
                results.append((root_id, category, "SKIPPED: no voxels at this downsample"))
                continue
            z0, y0, x0 = z_slc.start, y_slc.start, x_slc.start
            # Zero-pad by 1 voxel on every side so marching cubes always
            # sees a closed surface, including fragments cut off at the
            # bbox edge (crop's own boundary); see docstring.
            padded = np.pad(crop, 1, mode="constant", constant_values=False).astype(np.float32)

            verts_zyx, faces, _normals, _values = measure.marching_cubes(padded, level=0.5, spacing=tuple(spacing_zyx))
            # verts_zyx[k] is grid-step k's position in nm relative to the
            # padded crop's own [0,0,0] corner, which sits at downsampled-
            # volume index (z0-1, y0-1, x0-1); convert that corner's own
            # world nm position (Y and X indices are downsampled, so their
            # native voxel offset is index*downsample; Z is untouched).
            crop_origin_nm_zyx = np.array([
                (bbox_min_z + (z0 - 1)) * res_z,
                (bbox_min_y + (y0 - 1) * downsample) * res_y,
                (bbox_min_x + (x0 - 1) * downsample) * res_x,
            ])
            verts_world_zyx = verts_zyx + crop_origin_nm_zyx
            verts_xyz = verts_world_zyx[:, [2, 1, 0]]

            _write_ply_ascii(out_path, verts_xyz, faces)
            results.append((root_id, category, f"OK ({len(verts_xyz)} verts, {len(faces)} faces)"))
        except Exception as ex:
            results.append((root_id, category, f"FAILED: {ex}"))

    return results
