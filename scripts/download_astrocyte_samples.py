"""Download bounded, native-resolution astrocyte samples (not whole-cell crops).

Example: python -B scripts/download_astrocyte_samples.py --output-dir samples/run1
Existing output directories are refused to avoid mixing or overwriting samples.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.coordinate_utils import xyz_to_zyx
from src.microns_client import MicronsClient, load_config


def bounded_box(center, shape, lower, upper):
    """Shift a requested XYZ box inside dataset bounds without changing its size."""
    shape = np.asarray(shape, dtype=np.int64)
    if np.any(shape <= 0) or np.any(shape > upper - lower):
        raise ValueError("Sample dimensions must be positive and fit dataset bounds")
    start = np.clip(np.asarray(center, dtype=np.int64) - shape // 2, lower, upper - shape)
    return start, start + shape


def overlaps(start, end, previous):
    return any(np.all(start < b) and np.all(a < end) for a, b in previous)


def save_preview(path, image, mask, resolution):
    """Show the XY slice with most target voxels, raw and red-overlay side by side."""
    z = int(np.argmax(mask.sum(axis=(1, 2))))
    raw = np.repeat(image[z, :, :, None], 3, axis=2)
    overlay = raw.copy()
    selected = mask[z].astype(bool)
    overlay[selected] = (0.55 * overlay[selected] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    h, w = image.shape[1:]
    canvas = Image.new("RGB", (2 * w, h + 32), "white")
    canvas.paste(Image.fromarray(raw), (0, 32))
    canvas.paste(Image.fromarray(overlay), (w, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"Raw EM | local z={z}", fill="black")
    draw.text((w + 8, 8), "Red = target astrocyte", fill="black")
    bar_pixels = round(1000 / float(resolution[0]))
    if bar_pixels < w - 32:
        draw.line((16, h + 16, 16 + bar_pixels, h + 16), fill="white", width=4)
        draw.text((16, h), "1 um", fill="white")
    canvas.save(path)
    return z


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-id", type=int, default=864691136057675736)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--shape-xyz", type=int, nargs=3, default=[512, 512, 128])
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-occupancy", type=float, default=0.01)
    args = parser.parse_args()
    shape = np.asarray(args.shape_xyz, dtype=np.int64)
    if np.any(shape <= 0) or any(int(x) > 1024 for x in shape):
        parser.error("Each dimension must be between 1 and 1024")
    voxels = int(np.prod(shape))
    if voxels > 512 * 512 * 128:
        parser.error("This small-sample tool limits each crop to 33,554,432 voxels")
    if not 1 <= args.count <= 10 or not args.count <= args.max_attempts <= 30:
        parser.error("Use 1-10 samples and count <= max-attempts <= 30")
    if not 0 < args.min_occupancy <= 1:
        parser.error("min-occupancy must be in (0, 1]")
    output = Path(args.output_dir).resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new run directory")
    existing_parent = output.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < 2 * voxels * args.count + 2 * 1024**3:
        parser.error("Insufficient disk space including 2 GiB headroom")

    config = load_config(args.config)
    if config['em'].get('mip', 0) != 0 or config['segmentation'].get('mip', 0) != 0:
        parser.error("This tool requires native-resolution mip 0 for both sources")
    print("Connecting and fetching target mesh (no full-volume download)...", flush=True)
    client = MicronsClient(config)
    # Missing chunks must raise instead of silently appearing as empty tissue.
    client.em_cv.fill_missing = False
    client.seg_cv.fill_missing = False
    if np.dtype(client.em_cv.dtype) != np.dtype('uint8'):
        raise ValueError("Expected uint8 EM; refusing implicit intensity conversion")
    vertices = client.get_mesh_vertices_nm(args.root_id)
    indices = np.random.default_rng(args.seed).permutation(len(vertices))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root_id": str(args.root_id), "dataset": client.dataset,
        "materialization_version": client.materialization_version,
        "seed": args.seed, "requested_count": args.count,
        "shape_xyz": shape.tolist(), "min_occupancy": args.min_occupancy,
        "max_attempts": args.max_attempts, "attempts": [], "samples": [],
        "selection": "Seeded mesh-vertex candidates, nonoverlapping accepted boxes; occupancy checked in segmentation. Not representative sampling or mitochondria selection.",
    }
    previous = []
    seen = set()
    for index in indices:
        if len(manifest['samples']) >= args.count or len(manifest['attempts']) >= args.max_attempts:
            break
        center = np.floor(vertices[index] / client.resolution_nm).astype(np.int64)
        start, end = bounded_box(center, shape, client.bounds_min_vx, client.bounds_max_vx)
        key = tuple(start)
        if key in seen or overlaps(start, end, previous):
            continue
        seen.add(key)
        print(f"Checking candidate {len(manifest['attempts']) + 1}: XYZ {start.tolist()}", flush=True)
        seg = xyz_to_zyx(client.download_segmentation(start, end))
        if seg.shape != tuple(shape[::-1]):
            raise ValueError(f"Unexpected segmentation shape: {seg.shape}")
        mask = (seg == args.root_id).astype(np.uint8)
        del seg
        occupancy = float(mask.mean())
        manifest['attempts'].append({"bbox_min_xyz": start.tolist(), "occupancy": occupancy})
        write_json(output / 'manifest.json', manifest)
        if occupancy < args.min_occupancy:
            print(f"  skipped: target occupancy {occupancy:.2%}", flush=True)
            continue
        print(f"  target occupancy {occupancy:.2%}; downloading EM...", flush=True)
        image = xyz_to_zyx(client.download_em(start, end))
        if image.shape != mask.shape or image.dtype != np.uint8:
            raise ValueError("EM shape/dtype differs from the expected aligned uint8 crop")
        folder = output / f"sample_{len(manifest['samples']) + 1:02d}"
        folder.mkdir()
        tifffile.imwrite(folder / 'image.tif', image, photometric='minisblack', metadata={'axes': 'ZYX'})
        tifffile.imwrite(folder / 'mask.tif', mask, photometric='minisblack', metadata={'axes': 'ZYX'})
        preview_z = save_preview(folder / 'preview.png', image, mask, client.resolution_nm)
        metadata = {
            "root_id": str(args.root_id), "dataset": client.dataset,
            "materialization_version": client.materialization_version,
            "materialization_timestamp": str(client.materialization_timestamp),
            "em_source": config['em']['source'], "segmentation_source": config['segmentation']['source'],
            "em_mip": 0, "segmentation_mip": 0, "fill_missing": False,
            "voxel_bbox": {"min": start.tolist(), "max": end.tolist()},
            "bbox_max_is_exclusive": True, "saved_axis_order": "ZYX",
            "em_resolution_nm": client.em_resolution_nm.tolist(),
            "seg_resolution_nm": client.seg_resolution_nm.tolist(),
            "image_shape": list(image.shape), "mask_shape": list(mask.shape),
            "positive_voxels": int(mask.sum()), "occupancy": occupancy,
            "mesh_vertex_index": int(index), "mesh_vertex_nm": vertices[index].tolist(),
            "preview_local_z": preview_z, "extraction_mode": "small_roi",
            "note": "Original EM; mask is target-cell membership, not mitochondria labels. No denoising or intensity normalization applied.",
        }
        write_json(folder / 'metadata.json', metadata)
        previous.append((start, end))
        manifest['samples'].append(folder.name)
        write_json(output / 'manifest.json', manifest)
        print(f"  saved {folder}", flush=True)
        del image, mask
    manifest['complete'] = len(manifest['samples']) == args.count
    write_json(output / 'manifest.json', manifest)
    print(f"Saved {len(manifest['samples'])}/{args.count} samples to {output}", flush=True)
    return 0 if manifest['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
