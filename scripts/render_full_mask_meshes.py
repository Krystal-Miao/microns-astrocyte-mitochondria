# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
3D renders using each root_id's complete shape (not the trimmed sliver). Same style and camera as render_contact_meshes.py.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pyvista as pv
import time

from src import io_utils

pv.OFF_SCREEN = True

MESH_DIR_NAME = "meshes_full_mask_ds10"

TARGET_COLOR = (1.0, 0.0, 0.0)
ASTROCYTE_COLOR = (0.0, 0.4, 1.0)
NEURON_COLOR = (0.13, 0.7, 0.3)


def _load_merged(paths: list[str], decimate_target_reduction: float = 0.0, label: str = "") -> pv.PolyData | None:
    """decimate_target_reduction > 0 simplifies each mesh on load: the
    full-mask meshes can be tens of GB per category, easily OOMing an
    undecimated merge; fine since this is a display render, not a
    measurement. Prints progress every 100 meshes."""
    t0 = time.time()
    meshes = []
    for i, path in enumerate(paths):
        mesh = pv.read(path)
        if not mesh.n_points:
            continue
        if decimate_target_reduction > 0 and mesh.n_points > 200:
            mesh = mesh.triangulate().decimate(decimate_target_reduction)
        meshes.append(mesh)
        if label and (i + 1) % 100 == 0:
            print(f"    [{label}] {i + 1}/{len(paths)} meshes loaded+decimated ({time.time() - t0:.0f}s)", flush=True)
    if label:
        print(f"    [{label}] all {len(paths)} loaded+decimated in {time.time() - t0:.0f}s, merging...", flush=True)
    return pv.merge(meshes) if meshes else None


def render_view(mesh_root: str, out_path: str, label: str) -> None:
    t0 = time.time()
    target = _load_merged(glob.glob(os.path.join(mesh_root, "target_astrocyte", "*.ply")),
                          decimate_target_reduction=0.7, label=f"{label}/target")
    astro = _load_merged(glob.glob(os.path.join(mesh_root, "astrocyte", "*.ply")), decimate_target_reduction=0.85,
                         label=f"{label}/astro")
    neuron = _load_merged(glob.glob(os.path.join(mesh_root, "neuron", "*.ply")), decimate_target_reduction=0.9,
                          label=f"{label}/neuron")
    print(f"[{label}] loaded in {time.time() - t0:.1f}s")

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    plotter.set_background("white")
    if neuron is not None:
        plotter.add_mesh(neuron, color=NEURON_COLOR, smooth_shading=True, specular=0.15, ambient=0.25)
    if astro is not None:
        plotter.add_mesh(astro, color=ASTROCYTE_COLOR, smooth_shading=True, specular=0.15, ambient=0.25)
    if target is not None:
        plotter.add_mesh(target, color=TARGET_COLOR, opacity=0.15, smooth_shading=True, specular=0.1, ambient=0.3)

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.25)
    plotter.enable_anti_aliasing("ssaa")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plotter.screenshot(out_path)
    print(f"[{label}] saved {out_path} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Path to run directory (e.g. astro_<root_id>/runs/<date>)")
    parser.add_argument("--data-dir", default=None, help="Path to astro_<root_id> directory")
    args = parser.parse_args()

    out_dir, run_dir = io_utils.resolve_script_dirs(args.run_dir, args.data_dir)

    render_view(
        os.path.join(run_dir, "full_mask_view", MESH_DIR_NAME),
        os.path.join(run_dir, "mesh_renders", "loose_full_mask.png"),
        "loose",
    )
    render_view(
        os.path.join(run_dir, "strict_view", "full_mask_view", MESH_DIR_NAME),
        os.path.join(run_dir, "strict_view", "mesh_renders", "proofread_full_mask.png"),
        "proofread",
    )

