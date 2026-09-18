# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
Render contact scenes from full-mask pre-filter slabs downsampled in Y and X.
"""

import glob
import json
import os
import time

import numpy as np
import pyvista as pv
import tifffile
from skimage import measure

pv.OFF_SCREEN = True

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyvista as pv
import tifffile
from skimage import measure

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import io_utils

pv.OFF_SCREEN = True

DOWNSAMPLE = 10

# category codes baked into the lookup tables below
DROP, TARGET, ASTROCYTE, NEURON = 0, 1, 2, 3

COLORS = {
    TARGET: (1.0, 0.0, 0.0),
    ASTROCYTE: (0.0, 0.4, 1.0),
    NEURON: (0.13, 0.7, 0.3),
}


def _category_code(entry: dict) -> int:
    if entry.get("is_target_astrocyte"):
        return TARGET
    if entry.get("cell_type") == "astrocyte":
        return ASTROCYTE
    if entry.get("classification_system") in ("excitatory_neuron", "inhibitory_neuron"):
        return NEURON
    return DROP


def _load_lut(run_dir: str) -> np.ndarray:
    with open(os.path.join(run_dir, "interaction_label_map.json")) as f:
        entries = json.load(f)["entries"]
    lut = np.zeros(65536, dtype=np.uint8)  # local_id is uint16 in the slabs
    for e in entries:
        lut[e["local_id"]] = _category_code(e)
    return lut


def _slab_z_range(path: str) -> tuple[int, int]:
    name = os.path.splitext(os.path.basename(path))[0]
    z0, z1 = name.split("_")
    return int(z0), int(z1)


def build_category_volumes(loose_lut: np.ndarray, strict_lut: np.ndarray, metadata_path: str, slabs_dir: str):
    """
    One pass over every pre-filter slab: downsample Y and X, remap through
    both LUTs, accumulate two (nz_total, ny_ds, nx_ds) uint8 category
    volumes. Z is kept at native resolution (matches mesh_export.py's
    downsampling convention).
    """
    with open(metadata_path) as f:
        bbox_min_z = int(json.load(f)["voxel_bbox"]["min"][2])

    paths = sorted(glob.glob(os.path.join(slabs_dir, "*.tif")), key=_slab_z_range)
    print(f"{len(paths)} pre-filter slabs in {slabs_dir}")

    loose_chunks, strict_chunks = [], []
    t0 = time.time()
    for i, path in enumerate(paths):
        z0_abs, z1_abs = _slab_z_range(path)
        assert z0_abs - bbox_min_z >= 0
        arr = tifffile.imread(path)  # (z, y, x) uint16
        arr_ds = arr[:, ::DOWNSAMPLE, ::DOWNSAMPLE]
        loose_chunks.append(loose_lut[arr_ds])
        strict_chunks.append(strict_lut[arr_ds])
        if i % 20 == 0:
            print(f"  slab {i + 1}/{len(paths)} ({time.time() - t0:.0f}s elapsed)")

    loose_vol = np.concatenate(loose_chunks, axis=0) if loose_chunks else np.zeros((0, 0, 0), dtype=np.uint8)
    strict_vol = np.concatenate(strict_chunks, axis=0) if strict_chunks else np.zeros((0, 0, 0), dtype=np.uint8)
    print(f"volumes built in {time.time() - t0:.0f}s, shape {loose_vol.shape}")
    return loose_vol, strict_vol


def render_category_volume(vol: np.ndarray, out_path: str, label: str) -> None:
    t0 = time.time()
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    plotter.set_background("white")

    for code in (NEURON, ASTROCYTE, TARGET):  # draw target last, semi-transparent, on top
        mask = (vol == code)
        if not mask.any():
            continue
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
        mesh = pv.PolyData(verts, faces_pv)
        opacity = 0.15 if code == TARGET else 1.0
        plotter.add_mesh(mesh, color=COLORS[code], opacity=opacity, smooth_shading=True,
                          specular=0.15, ambient=0.25)

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.25)
    plotter.enable_anti_aliasing("ssaa")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plotter.screenshot(out_path)
    print(f"[{label}] saved {out_path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Path to run directory (e.g. astro_<root_id>/runs/<date>)")
    parser.add_argument("--data-dir", default=None, help="Path to astro_<root_id> directory")
    args = parser.parse_args()

    out_dir, run_dir = io_utils.resolve_script_dirs(args.run_dir, args.data_dir)
    strict_dir = os.path.join(run_dir, "strict_view")
    slabs_dir = os.path.join(run_dir, "_slabs", "interaction_label")
    metadata_path = io_utils.metadata_path(out_dir)

    loose_lut = _load_lut(run_dir)
    strict_lut = _load_lut(strict_dir)

    loose_vol, strict_vol = build_category_volumes(loose_lut, strict_lut, metadata_path, slabs_dir)

    render_category_volume(loose_vol, os.path.join(run_dir, "mesh_renders", "loose_full_mask.png"), "loose-full")
    render_category_volume(strict_vol, os.path.join(strict_dir, "mesh_renders", "proofread_full_mask.png"), "proofread-full")

