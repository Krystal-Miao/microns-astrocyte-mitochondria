# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
Static 3D renders of loose and proofread contact scenes from mesh_export's .ply files.
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

MESH_DIR_NAME = "meshes_local_ds10"

# target and astrocyte match "io_utils.py"'s 2D overlay colors; neuron deliberately doesn't (green here, not orange).
TARGET_COLOR = (1.0, 0.0, 0.0)
ASTROCYTE_COLOR = (0.0, 0.4, 1.0)
NEURON_COLOR = (0.13, 0.7, 0.3)


def _load_merged(paths: list[str]) -> pv.PolyData | None:
    """Merges one category's meshes into one PolyData: one draw call instead of ~1,400."""
    meshes = []
    for path in paths:
        mesh = pv.read(path)
        if mesh.n_points:
            meshes.append(mesh)
    return pv.merge(meshes) if meshes else None


def render_view(mesh_root: str, out_path: str, label: str) -> None:
    t0 = time.time()
    target = _load_merged(glob.glob(os.path.join(mesh_root, "target_astrocyte", "*.ply")))
    astro = _load_merged(glob.glob(os.path.join(mesh_root, "astrocyte", "*.ply")))
    neuron = _load_merged(glob.glob(os.path.join(mesh_root, "neuron", "*.ply")))
    print(f"[{label}] loaded in {time.time() - t0:.1f}s")

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    plotter.set_background("white")
    # neuron and astrocyte drawn before the (semi-transparent) target so they
    # composite correctly underneath it.
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
        os.path.join(run_dir, MESH_DIR_NAME),
        os.path.join(run_dir, "mesh_renders", "loose.png"),
        "loose",
    )
    render_view(
        os.path.join(run_dir, "strict_view", MESH_DIR_NAME),
        os.path.join(run_dir, "strict_view", "mesh_renders", "proofread.png"),
        "proofread",
    )

