# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
Coordinate conversions between physical nanometers (XYZ) and voxel indices for MICrONS datasets.
"""

from __future__ import annotations

import numpy as np


def physical_nm_to_voxel(coords_nm, resolution_nm) -> np.ndarray:
    """
    Convert physical nanometer coordinates (XYZ) to fractional voxel indices.
    """
    coords_nm = np.asarray(coords_nm, dtype=np.float64)
    resolution_nm = np.asarray(resolution_nm, dtype=np.float64)
    return coords_nm / resolution_nm


def voxel_to_physical_nm(coords_vx, resolution_nm) -> np.ndarray:
    """
    Convert voxel indices (XYZ) to physical nanometer coordinates.
    """
    coords_vx = np.asarray(coords_vx, dtype=np.float64)
    resolution_nm = np.asarray(resolution_nm, dtype=np.float64)
    return coords_vx * resolution_nm


def um_to_voxel_margin(margin_um: dict, resolution_nm) -> np.ndarray:
    """
    Convert micrometer margin dict {'x','y','z'} to voxel margin.
    """
    resolution_nm = np.asarray(resolution_nm, dtype=np.float64)
    margin_nm = np.array(
        [margin_um["x"], margin_um["y"], margin_um["z"]], dtype=np.float64
    ) * 1000.0
    return np.ceil(margin_nm / resolution_nm).astype(np.int64)


def clip_bbox_to_bounds(bbox_min_vx, bbox_max_vx, bounds_min_vx, bounds_max_vx) -> dict:
    """
    Clip voxel bounding box to dataset bounds.
    """
    bbox_min_vx = np.asarray(bbox_min_vx, dtype=np.int64)
    bbox_max_vx = np.asarray(bbox_max_vx, dtype=np.int64)
    bounds_min_vx = np.asarray(bounds_min_vx, dtype=np.int64)
    bounds_max_vx = np.asarray(bounds_max_vx, dtype=np.int64)

    clipped_min = np.maximum(bbox_min_vx, bounds_min_vx)
    clipped_max = np.minimum(bbox_max_vx, bounds_max_vx)

    if np.any(clipped_max <= clipped_min):
        raise ValueError(
            "Bounding box became empty/invalid after clipping to dataset bounds: "
            f"requested min={bbox_min_vx.tolist()}, max={bbox_max_vx.tolist()}, "
            f"dataset bounds min={bounds_min_vx.tolist()}, max={bounds_max_vx.tolist()}"
        )

    was_clipped = bool(np.any(clipped_min != bbox_min_vx) or np.any(clipped_max != bbox_max_vx))
    return {
        "min": clipped_min,
        "max": clipped_max,
        "was_clipped": was_clipped,
    }


def mesh_bbox_to_volume_bbox(vertices_nm, resolution_nm, margin_um, volume_bounds_vx) -> dict:
    """
    Calculate volume bounding box from mesh vertices (XYZ nm) and margin_um.
    """
    vertices_nm = np.asarray(vertices_nm, dtype=np.float64)
    if vertices_nm.ndim != 2 or vertices_nm.shape[1] != 3:
        raise ValueError(f"Expected vertices of shape (N, 3), got {vertices_nm.shape}")

    mesh_bbox_min_nm = vertices_nm.min(axis=0)
    mesh_bbox_max_nm = vertices_nm.max(axis=0)

    bbox_min_vx = np.floor(physical_nm_to_voxel(mesh_bbox_min_nm, resolution_nm)).astype(np.int64)
    bbox_max_vx = np.ceil(physical_nm_to_voxel(mesh_bbox_max_nm, resolution_nm)).astype(np.int64)

    margin_vx = um_to_voxel_margin(margin_um, resolution_nm)
    bbox_min_vx = bbox_min_vx - margin_vx
    bbox_max_vx = bbox_max_vx + margin_vx

    bounds_min_vx, bounds_max_vx = volume_bounds_vx
    clipped = clip_bbox_to_bounds(bbox_min_vx, bbox_max_vx, bounds_min_vx, bounds_max_vx)

    return {
        "min": clipped["min"],
        "max": clipped["max"],
        "unclipped_min": bbox_min_vx,
        "unclipped_max": bbox_max_vx,
        "was_clipped": clipped["was_clipped"],
        "mesh_bbox_min_nm": mesh_bbox_min_nm,
        "mesh_bbox_max_nm": mesh_bbox_max_nm,
    }


def voxel_shape(bbox_min_vx, bbox_max_vx) -> np.ndarray:
    """
    Calculate voxel shape (XYZ) of a half-open bounding box.
    """
    bbox_min_vx = np.asarray(bbox_min_vx, dtype=np.int64)
    bbox_max_vx = np.asarray(bbox_max_vx, dtype=np.int64)
    return bbox_max_vx - bbox_min_vx


def xyz_to_zyx(volume: np.ndarray) -> np.ndarray:
    """
    Transpose 3D volume array from (X, Y, Z) to (Z, Y, X) order.
    """
    volume = np.asarray(volume)
    if volume.ndim == 4:
        if volume.shape[-1] != 1:
            raise ValueError(
                f"Expected a single-channel volume, got channel dimension of size {volume.shape[-1]}"
            )
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D (or 4D single-channel) volume, got shape {volume.shape}")
    # incoming axes are (X, Y, Z) -> transpose to (Z, Y, X)
    return np.transpose(volume, (2, 1, 0))
