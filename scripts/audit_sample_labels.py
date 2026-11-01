"""Audit target-astrocyte sample labels against MICrONS segmentation and cell types.

This script never edits the source image.tif, mask.tif, or metadata.json files.
It reconstructs the target mask from the pinned segmentation snapshot, classifies
large-enough root IDs when annotation exists, and builds an offline overlay viewer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coordinate_utils import xyz_to_zyx
from src.interaction_query import find_bbox_segmentation_cells
from src.microns_client import MicronsClient, load_config


LABELS = {
    0: ("no_segment", (0, 0, 0)),
    1: ("target_astrocyte", (255, 55, 55)),
    2: ("confirmed_other_astrocyte", (0, 230, 210)),
    3: ("confirmed_neuron", (80, 145, 255)),
    4: ("other_classified_cell", (255, 155, 55)),
    5: ("unclassified_queried_segment", (235, 215, 70)),
    6: ("tiny_unqueried_segment", (150, 150, 150)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source(sample_dir: Path, metadata: dict) -> tuple[np.memmap, np.memmap]:
    image = tifffile.memmap(sample_dir / "image.tif", mode="r")
    mask = tifffile.memmap(sample_dir / "mask.tif", mode="r")
    bbox = metadata["voxel_bbox"]
    expected = tuple((np.asarray(bbox["max"]) - np.asarray(bbox["min"]))[::-1])
    if metadata.get("saved_axis_order") != "ZYX":
        raise ValueError(f"{sample_dir}: expected saved_axis_order=ZYX")
    if image.shape != expected or mask.shape != expected:
        raise ValueError(
            f"{sample_dir}: TIFF shape mismatch; expected {expected}, "
            f"got image={image.shape}, mask={mask.shape}"
        )
    if image.ndim != 3 or image.dtype != np.uint8:
        raise ValueError(f"{sample_dir}: expected a 3-D uint8 EM image")
    return image, mask


def classification_rows(cells: list[dict], counts: Counter, target_root_id: int) -> list[dict]:
    by_root = {int(cell["root_id"]): cell for cell in cells}
    rows = []
    total = sum(counts.values())
    for root_id, voxel_count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        cell = by_root.get(int(root_id))
        queried = cell is not None
        category = cell.get("category") if cell else None
        cell_type = cell.get("cell_type") if cell else None
        if int(root_id) == target_root_id:
            label_code = 1
        elif category == "astrocyte":
            label_code = 2
        elif category == "neuron":
            label_code = 3
        elif cell_type is not None:
            label_code = 4
        elif queried:
            label_code = 5
        else:
            label_code = 6
        rows.append({
            "root_id": int(root_id),
            "voxel_count": int(voxel_count),
            "voxel_fraction": float(voxel_count / total),
            "queried": queried,
            "nucleus_id": cell.get("nucleus_id") if cell else None,
            "classification_system": cell.get("classification_system") if cell else None,
            "cell_type": cell_type,
            "annotation_category": category,
            "audit_label_code": label_code,
            "audit_label": LABELS[label_code][0],
        })
    return rows


def label_plane(segmentation: np.ndarray, code_by_root: dict[int, int]) -> np.ndarray:
    result = np.zeros(segmentation.shape, dtype=np.uint8)
    for root_id, code in code_by_root.items():
        result[segmentation == root_id] = code
    return result


def overlay(image: np.ndarray, labels: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    rgb = np.repeat(np.asarray(image, dtype=np.uint8)[..., None], 3, axis=2)
    out = rgb.astype(np.float32)
    for code, (_, color) in LABELS.items():
        if code == 0:
            continue
        selected = labels == code
        out[selected] = (1.0 - alpha) * out[selected] + alpha * np.asarray(color)
    return np.clip(out, 0, 255).astype(np.uint8)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_sample(client, sample_dir: Path, output_dir: Path, min_voxels: int, slab_z: int) -> dict:
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    image, source_mask = validate_source(sample_dir, metadata)
    source_hashes_before = {
        name: sha256(sample_dir / name) for name in ("image.tif", "mask.tif", "metadata.json")
    }

    if metadata["dataset"] != client.dataset:
        raise ValueError(f"{sample_dir}: dataset differs from config")
    if int(metadata["materialization_version"]) != client.materialization_version:
        raise ValueError(f"{sample_dir}: materialization version differs from config")
    if metadata["segmentation_source"] != client.config["segmentation"]["source"]:
        raise ValueError(f"{sample_dir}: segmentation source differs from config")

    bbox_min = np.asarray(metadata["voxel_bbox"]["min"], dtype=np.int64)
    bbox_max = np.asarray(metadata["voxel_bbox"]["max"], dtype=np.int64)
    target_root_id = int(metadata["root_id"])
    counts: Counter = Counter()
    mismatches = 0

    output_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=f"label-audit-{sample_dir.name}-") as temp:
        seg_path = Path(temp) / "segmentation.npy"
        segmentation = np.lib.format.open_memmap(
            seg_path, mode="w+", dtype=np.uint64, shape=image.shape
        )
        for z0 in range(0, image.shape[0], slab_z):
            z1 = min(z0 + slab_z, image.shape[0])
            slab_min = bbox_min.copy()
            slab_max = bbox_max.copy()
            slab_min[2], slab_max[2] = bbox_min[2] + z0, bbox_min[2] + z1
            slab = xyz_to_zyx(client.download_segmentation(slab_min, slab_max)).astype(np.uint64)
            segmentation[z0:z1] = slab
            ids, id_counts = np.unique(slab, return_counts=True)
            counts.update({int(rid): int(n) for rid, n in zip(ids, id_counts) if rid != 0})
            reconstructed = slab == target_root_id
            mismatches += int(np.count_nonzero(reconstructed != (np.asarray(source_mask[z0:z1]) != 0)))
            print(f"{sample_dir.name}: downloaded Z {z0}:{z1}", flush=True)
        segmentation.flush()

        candidate_ids = sorted(rid for rid, count in counts.items() if count >= min_voxels)
        if target_root_id not in candidate_ids:
            candidate_ids.append(target_root_id)
        cells = find_bbox_segmentation_cells(
            client.cave, client.materialization_version, target_root_id, candidate_ids
        )
        rows = classification_rows(cells, counts, target_root_id)
        write_csv(output_dir / "root_ids.csv", rows)
        code_by_root = {row["root_id"]: row["audit_label_code"] for row in rows}

        labels_path = output_dir / "audit_labels.tif"
        labels = tifffile.memmap(labels_path, shape=image.shape, dtype=np.uint8, bigtiff=True)
        assets = output_dir / "assets"
        assets.mkdir()
        for z in range(image.shape[0]):
            labels[z] = label_plane(np.asarray(segmentation[z]), code_by_root)
            Image.fromarray(overlay(np.asarray(image[z]), labels[z])).save(
                assets / f"overlay_{z:03d}.png", optimize=False
            )
        labels.flush()

    label_voxels = Counter()
    for row in rows:
        label_voxels[row["audit_label_code"]] += row["voxel_count"]
    total_voxels = int(np.prod(image.shape))
    segmented_voxels = sum(counts.values())
    summary = {
        "sample": sample_dir.name,
        "root_id": str(target_root_id),
        "shape_zyx": list(image.shape),
        "voxel_bbox_xyz": metadata["voxel_bbox"],
        "min_voxels_for_annotation_query": min_voxels,
        "root_id_count_excluding_zero": len(counts),
        "queried_root_id_count": sum(row["queried"] for row in rows),
        "target_mask_reconstruction": {
            "mismatched_voxels": mismatches,
            "exact_match": mismatches == 0,
        },
        "voxel_counts": {
            LABELS[code][0]: int(label_voxels[code]) for code in sorted(LABELS) if code != 0
        },
        "fractions_of_full_volume": {
            LABELS[code][0]: float(label_voxels[code] / total_voxels)
            for code in sorted(LABELS) if code != 0
        },
        "zero_segmentation_fraction": float((total_voxels - segmented_voxels) / total_voxels),
        "source_sha256_before": source_hashes_before,
        "limitations": [
            "Cell-type annotation requires a detected nucleus; unclassified does not mean non-astrocyte.",
            "Tiny segments below the query threshold are retained visually but are not queried.",
            "This is a label/data audit, not an expert morphology review.",
        ],
    }
    source_hashes_after = {
        name: sha256(sample_dir / name) for name in ("image.tif", "mask.tif", "metadata.json")
    }
    summary["source_sha256_after"] = source_hashes_after
    summary["source_files_unchanged"] = source_hashes_before == source_hashes_after
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    del image, source_mask
    return summary


def build_viewer(output: Path, summaries: list[dict]) -> None:
    legend = "".join(
        f'<span><i style="background:rgb{color}"></i>{name}</span>'
        for code, (name, color) in LABELS.items() if code
    )
    payload = json.dumps(summaries).replace("<", "\\u003c")
    html = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>MICrONS label audit</title><style>
body{{font:16px system-ui;max-width:1050px;margin:28px auto;background:#15222b;color:#eef;padding:16px}}
img{{width:min(90vw,768px);image-rendering:pixelated;border:2px solid #52636e}}
input{{width:min(85vw,720px)}} select{{padding:8px}} .legend span{{display:inline-block;margin:6px 14px 6px 0}}
.legend i{{display:inline-block;width:14px;height:14px;margin-right:5px;vertical-align:-2px}}
code{{color:#9dd7ff}} a{{color:#8fcfff}} p{{line-height:1.55}}</style>
<h1>MICrONS target-label audit</h1>
<p>这是自动 data/label audit，不要求你凭形态识别 astrocyte。颜色来自固定 segmentation root ID 与公开 cell-type annotations；黄色和灰色都表示 unknown，不能解释成 non-astrocyte。</p>
<div class="legend">{legend}</div>
<p><select id="sample"></select> Z=<b id="ztext"></b></p><input id="z" type="range" min="0" value="0">
<p id="status"></p><img id="view"><p><a id="csv">root_ids.csv</a> · <a id="summary">summary.json</a> · <a id="labels">audit_labels.tif</a></p>
<script>const data={payload};const s=document.getElementById('sample'),z=document.getElementById('z');
data.forEach((x,i)=>s.add(new Option(x.sample,i)));function show(){{const x=data[+s.value];z.max=x.shape_zyx[0]-1;
const zi=+z.value;document.getElementById('ztext').textContent=zi;document.getElementById('view').src=x.sample+'/assets/overlay_'+String(zi).padStart(3,'0')+'.png';
document.getElementById('status').textContent='target mask exact match: '+x.target_mask_reconstruction.exact_match+'; mismatched voxels: '+x.target_mask_reconstruction.mismatched_voxels+'; annotated/query-eligible roots: '+x.queried_root_id_count+'/'+x.root_id_count_excluding_zero;
['csv','summary','labels'].forEach(id=>document.getElementById(id).href=x.sample+'/'+(id==='csv'?'root_ids.csv':id==='summary'?'summary.json':'audit_labels.tif'));}}
s.onchange=()=>{{z.value=0;show()}};z.oninput=show;show();</script></html>"""
    (output / "index.html").write_text(html)


def self_test() -> None:
    counts = Counter({11: 10, 22: 8, 33: 4, 44: 2})
    cells = [
        {"root_id": 11, "category": "astrocyte", "cell_type": "astrocyte", "nucleus_id": 1,
         "classification_system": "nonneuron", "cell_type_source": "x"},
        {"root_id": 22, "category": "neuron", "cell_type": "23P", "nucleus_id": 2,
         "classification_system": "excitatory_neuron", "cell_type_source": "x"},
        {"root_id": 33, "category": None, "cell_type": None, "nucleus_id": None,
         "classification_system": None, "cell_type_source": None},
    ]
    rows = classification_rows(cells, counts, target_root_id=11)
    assert {row["root_id"]: row["audit_label_code"] for row in rows} == {11: 1, 22: 3, 33: 5, 44: 6}
    seg = np.array([[0, 11], [22, 44]], dtype=np.uint64)
    assert np.array_equal(label_plane(seg, {11: 1, 22: 3, 44: 6}), [[0, 1], [3, 6]])
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("samples/astro_864691136057675736_run1"))
    parser.add_argument("--output-dir", type=Path, default=Path("samples/astro_864691136057675736_run1/label_audit_01"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--min-voxels", type=int, default=50)
    parser.add_argument("--slab-z", type=int, default=16)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output_dir.exists():
        raise ValueError(f"Output already exists: {args.output_dir}; choose a new directory")
    sample_dirs = sorted(path.parent for path in args.data_dir.glob("sample_*/metadata.json"))
    if not sample_dirs:
        raise ValueError(f"No sample metadata found under {args.data_dir}")
    client = MicronsClient(load_config(args.config))
    args.output_dir.mkdir(parents=True)
    try:
        summaries = [
            build_sample(client, sample, args.output_dir / sample.name, args.min_voxels, args.slab_z)
            for sample in sample_dirs
        ]
        (args.output_dir / "summary.json").write_text(json.dumps({
            "status": "automatic_audit_complete",
            "samples": summaries,
            "label_codes": {str(code): {"name": name, "rgb": color} for code, (name, color) in LABELS.items()},
        }, indent=2))
        build_viewer(args.output_dir, summaries)
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f"Open: {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
