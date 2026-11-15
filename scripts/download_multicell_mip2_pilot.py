"""Download one fixed-size mip2 EM/target-mask context crop per proofread astrocyte."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.coordinate_utils import xyz_to_zyx  # noqa: E402
from src.microns_client import MicronsClient, load_config  # noqa: E402


def bounded_box(center, shape, lower, upper):
    shape = np.asarray(shape, dtype=np.int64)
    start = np.clip(np.asarray(center, dtype=np.int64) - shape // 2, lower, upper - shape)
    return start, start + shape


def dense_mesh_center(vertices_nm, resolution_nm, shape_xyz):
    """Choose the occupied coarse bin containing the most mesh vertices."""
    vertices_vx = np.floor(vertices_nm / resolution_nm).astype(np.int64)
    bin_size = np.maximum(np.asarray(shape_xyz, dtype=np.int64) // 2, 1)
    bins = np.floor_divide(vertices_vx, bin_size)
    unique, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    winning = int(np.argmax(counts))
    points = vertices_vx[inverse == winning]
    return np.rint(np.median(points, axis=0)).astype(np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0,
                        help="Zero-based manifest offset; output numbering continues from this offset")
    parser.add_argument("--shape-xyz", type=int, nargs=3, default=[512, 512, 128])
    parser.add_argument("--mip", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists")
    manifest = json.loads(args.manifest.read_text())
    if args.start_index < 0 or args.limit < 1:
        parser.error("start-index must be nonnegative and limit must be positive")
    cells = manifest["cells"][args.start_index:args.start_index + args.limit]
    if not cells:
        parser.error("Manifest contains no selected cells")
    estimated = len(cells) * int(np.prod(args.shape_xyz)) * 2
    if shutil.disk_usage(args.out.parent).free < estimated + 2 * 1024**3:
        parser.error("Insufficient free space including 2 GiB headroom")
    config = load_config(args.config)
    config["em"]["mip"] = args.mip
    config["segmentation"]["mip"] = args.mip
    client = MicronsClient(config)
    client.em_cv.fill_missing = False
    client.seg_cv.fill_missing = False
    shape = np.asarray(args.shape_xyz, dtype=np.int64)
    args.out.mkdir(parents=True)
    output_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": client.dataset, "materialization_version": client.materialization_version,
        "mip": args.mip, "resolution_nm": client.resolution_nm.tolist(),
        "shape_xyz": shape.tolist(), "samples": [],
        "selection": "Center of densest half-crop-sized mesh-vertex bin; one crop per cell.",
        "label_semantics": "Binary target-instance membership only; partial class labels are built separately.",
    }
    try:
        for batch_index, cell in enumerate(cells, 1):
            index = args.start_index + batch_index
            root_id = int(cell["root_id"])
            print(f"[{batch_index}/{len(cells)}] mesh {root_id}", flush=True)
            vertices = client.get_mesh_vertices_nm(root_id)
            center = dense_mesh_center(vertices, client.resolution_nm, shape)
            start, end = bounded_box(center, shape, client.bounds_min_vx, client.bounds_max_vx)
            print(f"[{batch_index}/{len(cells)}] segmentation {start.tolist()}..{end.tolist()}", flush=True)
            segmentation = xyz_to_zyx(client.download_segmentation(start, end)).astype(np.uint64)
            if segmentation.shape != tuple(shape[::-1]):
                raise ValueError(f"Unexpected segmentation shape {segmentation.shape}")
            mask = (segmentation == root_id).astype(np.uint8)
            del segmentation
            print(f"[{batch_index}/{len(cells)}] EM; target occupancy {mask.mean():.2%}", flush=True)
            image = xyz_to_zyx(client.download_em(start, end))
            if image.shape != mask.shape or image.dtype != np.uint8:
                raise ValueError("Unexpected EM shape or dtype")
            name = f"sample_{index:02d}"
            sample = args.out / name
            sample.mkdir()
            tifffile.imwrite(sample / "image.tif", image, photometric="minisblack", metadata={"axes": "ZYX"})
            tifffile.imwrite(sample / "mask.tif", mask, photometric="minisblack", metadata={"axes": "ZYX"})
            metadata = {
                "root_id": str(root_id), "dataset": client.dataset,
                "materialization_version": client.materialization_version,
                "em_source": config["em"]["source"], "segmentation_source": config["segmentation"]["source"],
                "em_mip": args.mip, "segmentation_mip": args.mip,
                "em_resolution_nm": client.em_resolution_nm.tolist(),
                "seg_resolution_nm": client.seg_resolution_nm.tolist(),
                "voxel_bbox": {"min": start.tolist(), "max": end.tolist()},
                "saved_axis_order": "ZYX", "image_shape": list(image.shape), "mask_shape": list(mask.shape),
                "positive_voxels": int(mask.sum()), "occupancy": float(mask.mean()),
                "selection_center_xyz": center.tolist(), "selection_method": output_manifest["selection"],
            }
            (sample / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            output_manifest["samples"].append({"name": name, "root_id": str(root_id),
                                                "occupancy": float(mask.mean())})
            (args.out / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
            del image, mask
    except Exception:
        output_manifest["status"] = "incomplete"
        (args.out / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
        raise
    output_manifest["status"] = "complete"
    (args.out / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(f"Saved {len(cells)} cell-level volumes to {args.out}")


if __name__ == "__main__":
    main()
