# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""Thin CAVEclient+CloudVolume wrapper for minnie65_public, holding every
dataset-metadata value (resolutions, bounds, timestamps) the pipeline
treats as ground truth. Nothing here hard-codes a coordinate constant --
all of it is read live from CloudVolume and CAVE (see README.md)."""
from __future__ import annotations

import numpy as np
import yaml
from caveclient import CAVEclient
from cloudvolume import CloudVolume

import cloudfiles.cloudfiles as _cloudfiles_mod
import cloudfiles.interfaces as _cloudfiles_interfaces
import requests.adapters as _requests_adapters

# CloudFiles' HttpInterface shares one class-level HTTPAdapter across every
# request, whose urllib3 pool defaults to 10; unrelated to CloudFiles' own
# num_threads. Exceeding 10 tears down and reopens connections every request,
# costing an order of magnitude in throughput. Fix: raise the pool size and
# num_threads together so they stay matched.
_HTTP_CONCURRENCY = 40
_cloudfiles_interfaces.HttpInterface.adaptor = _requests_adapters.HTTPAdapter(
    pool_connections=_HTTP_CONCURRENCY, pool_maxsize=_HTTP_CONCURRENCY
)

_orig_cloudfiles_init = _cloudfiles_mod.CloudFiles.__init__


def _capped_cloudfiles_init(self, cloudpath, *args, **kwargs):
    kwargs["num_threads"] = min(kwargs.get("num_threads", 20), _HTTP_CONCURRENCY)
    return _orig_cloudfiles_init(self, cloudpath, *args, **kwargs)


_cloudfiles_mod.CloudFiles.__init__ = _capped_cloudfiles_init


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class MicronsClient:
    def __init__(self, config: dict):
        self.config = config
        self.dataset = config["dataset"]
        self.materialization_version = config["materialization_version"]

        self.cave = CAVEclient(self.dataset)

        em_cfg = config["em"]
        seg_cfg = config["segmentation"]
        # A static precomputed snapshot (e.g. seg_m1300) already encodes one
        # fixed materialization state; agglomerate and timestamp are graphene-only
        # concepts and get_version_metadata() is not needed to read voxels from
        # it. Only a dynamic graphene table needs the live CAVE materialize
        # service for a timestamp.
        self.seg_is_dynamic = seg_cfg["source"].startswith("graphene://")

        if self.seg_is_dynamic:
            version_meta = self.cave.materialize.get_version_metadata(self.materialization_version)
            self.materialization_timestamp = version_meta["time_stamp"]
        else:
            self.materialization_timestamp = None

        self.em_cv = CloudVolume(
            em_cfg["source"], mip=em_cfg.get("mip", 0), use_https=True, fill_missing=True, progress=False
        )
        self.seg_cv = CloudVolume(
            seg_cfg["source"], mip=seg_cfg.get("mip", 0), use_https=True, fill_missing=True, progress=False
        )

        # Resolutions and bounds are read live from CloudVolume, never assumed.
        self.em_resolution_nm = np.asarray(self.em_cv.resolution, dtype=np.float64)
        self.seg_resolution_nm = np.asarray(self.seg_cv.resolution, dtype=np.float64)

        if not np.allclose(self.em_resolution_nm, self.seg_resolution_nm):
            raise ValueError(
                "EM and segmentation resolution mismatch at the configured mips "
                f"(em={self.em_resolution_nm} nm, seg={self.seg_resolution_nm} nm). "
                "This pipeline requires them to match (or a mip-alignment step to be "
                "added) -- do not proceed by naively resizing one volume to the other."
            )
        self.resolution_nm = self.em_resolution_nm

        em_bounds = self.em_cv.bounds
        seg_bounds = self.seg_cv.bounds
        # Intersect EM and segmentation voxel bounds so any bbox we compute is
        # guaranteed valid in both volumes (they cover slightly different
        # voxel extents even though the resolution matches).
        self.bounds_min_vx = np.maximum(
            np.array(em_bounds.minpt, dtype=np.int64), np.array(seg_bounds.minpt, dtype=np.int64)
        )
        self.bounds_max_vx = np.minimum(
            np.array(em_bounds.maxpt, dtype=np.int64), np.array(seg_bounds.maxpt, dtype=np.int64)
        )

        self.em_dtype_itemsize = np.dtype(self.em_cv.dtype).itemsize
        self.seg_dtype_itemsize = np.dtype(self.seg_cv.dtype).itemsize

    def get_mesh(self, root_id: int):
        """Raw CloudVolume Mesh for one root_id (handles both the {id: Mesh}
        dict and bare-Mesh return conventions). One id per call, not a
        batch: the legacy unsharded mesh source's get() defaults to
        fuse=True, silently MERGING every requested segid's geometry."""
        mesh_result = self.seg_cv.mesh.get(root_id)
        if isinstance(mesh_result, dict):
            if root_id not in mesh_result:
                raise ValueError(
                    f"mesh.get({root_id}) returned a dict but it does not contain the requested root_id "
                    f"(keys={list(mesh_result.keys())})"
                )
            return mesh_result[root_id]
        return mesh_result

    def get_mesh_vertices_nm(self, root_id: int) -> np.ndarray:
        """Fetch the full mesh for root_id and return its vertices as an
        (N, 3) float64 array in physical nanometers, XYZ order.
        """
        mesh = self.get_mesh(root_id)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Unexpected mesh vertices shape {vertices.shape} for root_id {root_id}")
        if vertices.shape[0] == 0:
            raise ValueError(f"Mesh for root_id {root_id} has zero vertices")
        return vertices

    def download_em(self, bbox_min_vx: np.ndarray, bbox_max_vx: np.ndarray) -> np.ndarray:
        """Download the EM subvolume for [bbox_min_vx, bbox_max_vx) (XYZ,
        voxel indices, exclusive max). Returns the raw CloudVolume array
        (X, Y, Z, C); axis normalization happens in coordinate_utils.
        """
        sl = np.s_[
            bbox_min_vx[0] : bbox_max_vx[0],
            bbox_min_vx[1] : bbox_max_vx[1],
            bbox_min_vx[2] : bbox_max_vx[2],
        ]
        return np.asarray(self.em_cv.download(sl))

    def download_segmentation(self, bbox_min_vx: np.ndarray, bbox_max_vx: np.ndarray) -> np.ndarray:
        """Segmentation subvolume for the same region. For a dynamic
        graphene source, agglomerates at the pinned materialization
        version's timestamp (keeps root_ids consistent with the
        proofreading table, not the current live state); a static
        precomputed snapshot needs no such kwargs."""
        sl = np.s_[
            bbox_min_vx[0] : bbox_max_vx[0],
            bbox_min_vx[1] : bbox_max_vx[1],
            bbox_min_vx[2] : bbox_max_vx[2],
        ]
        if self.seg_is_dynamic:
            return np.asarray(
                self.seg_cv.download(sl, agglomerate=True, timestamp=self.materialization_timestamp)
            )
        return np.asarray(self.seg_cv.download(sl))
