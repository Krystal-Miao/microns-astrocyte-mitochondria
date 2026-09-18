# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
CLI entry point; see README's Usage section for commands, the
--data-dir design, and how output is laid out under it.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys

import numpy as np

from . import astrocyte_query as aq
from . import contact_filter as cf
from . import coordinate_utils as cu
from . import interaction_query as iq
from . import io_utils
from . import mesh_export as mx
from . import volume_extractor as ve
from .microns_client import MicronsClient, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MICrONS astrocyte-interaction extractor")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List clean proofread astrocytes")

    p_extract = sub.add_parser("extract", help="Extract EM + mask for one or more astrocytes (reuses existing image.tif/mask.tif if already present)")
    p_extract.add_argument("--data-dir", required=True, help="Root output directory -- data lands under <data-dir>/astro_<root_id>/")
    p_extract.add_argument("--root-id", type=int, action="append", default=None, help="Root ID to extract (repeatable)")
    p_extract.add_argument("--all-clean", action="store_true", help="Extract every clean proofread astrocyte")
    p_extract.add_argument("--margin-um", type=float, default=None, help="Override margin_um uniformly for x/y/z")
    p_extract.add_argument("--mode", choices=["full_bbox", "chunked"], default=None, help="Override extraction.mode")
    p_extract.add_argument("--max-voxels", type=int, default=None, help="Override extraction.max_voxels")
    p_extract.add_argument("--test-max-z-voxels", type=int, default=None, help="TEST ONLY: clip the mesh-derived bbox to this many Z-voxels from its start, for fast pipeline verification runs. No effect on default behavior.")

    p_status = sub.add_parser("status", help="Show chunked-extraction progress for a root_id (reads local files only, no network)")
    p_status.add_argument("--data-dir", required=True, help="Root output directory -- reads <data-dir>/astro_<root_id>/")
    p_status.add_argument("--root-id", type=int, required=True)

    p_interact = sub.add_parser(
        "find-interactions",
        help="Find real cells + synapse-compartment interactions near an already-extracted astrocyte, "
             "scoped to config's interaction_categories.",
    )
    p_interact.add_argument("--data-dir", required=True, help="Root output directory -- reads/writes under <data-dir>/astro_<root_id>/")
    p_interact.add_argument("--root-id", type=int, action="append", default=None, help="Root ID to analyze (repeatable)")
    p_interact.add_argument("--all-fetched", action="store_true", help="Analyze every astrocyte with a metadata.json under --data-dir")

    p_ilabels = sub.add_parser(
        "extract-interaction-labels",
        help="Build interaction_labels.tif (target astrocyte + every interacting cell find-interactions found, "
             "including synapse-implicated dendrite/axon owners whose soma is outside the bbox) plus the "
             "synapse-compartment marker overlay. Requires find-interactions to have been run first.",
    )
    p_ilabels.add_argument("--data-dir", required=True, help="Root output directory -- reads/writes under <data-dir>/astro_<root_id>/")
    p_ilabels.add_argument("--root-id", type=int, action="append", default=None, help="Root ID to build (repeatable)")
    p_ilabels.add_argument("--all-fetched", action="store_true", help="Run for every astrocyte with an interaction_analysis.json under --data-dir")
    p_ilabels.add_argument("--mode", choices=["full_bbox", "chunked"], default="chunked", help="How to re-download segmentation for labeling (default: chunked)")

    p_meshes = sub.add_parser(
        "fetch-meshes",
        help="Fetch+save one .ply per root_id in interaction_label_map.json -- the target astrocyte, its "
             "interacting astrocytes, and its interacting dendrites only, never the whole segmentation. "
             "Requires extract-interaction-labels to have been run first.",
    )
    p_meshes.add_argument("--data-dir", required=True, help="Root output directory -- reads/writes under <data-dir>/astro_<root_id>/")
    p_meshes.add_argument("--root-id", type=int, action="append", default=None, help="Root ID to fetch meshes for (repeatable)")
    p_meshes.add_argument("--all-fetched", action="store_true", help="Run for every astrocyte with an interaction_label_map.json under --data-dir")

    p_local_meshes = sub.add_parser(
        "reconstruct-local-meshes",
        help="Network-free alternative to fetch-meshes: marching-cubes reconstruction directly from the "
             "already-downloaded interaction_labels.tif, downsampled in Y/X. Requires "
             "extract-interaction-labels to have been run first. See mesh_export.reconstruct_local_meshes's "
             "docstring for how this differs from the official CAVE mesh.",
    )
    p_local_meshes.add_argument("--data-dir", required=True, help="Root output directory -- reads/writes under <data-dir>/astro_<root_id>/")
    p_local_meshes.add_argument("--root-id", type=int, action="append", default=None, help="Root ID to reconstruct meshes for (repeatable)")
    p_local_meshes.add_argument("--all-fetched", action="store_true", help="Run for every astrocyte with an interaction_label_map.json under --data-dir")
    p_local_meshes.add_argument("--downsample", type=int, default=10, help="Y/X downsample factor (default: 10, matching output.overlay_downsample)")

    p_clean_split = sub.add_parser(
        "clean-and-split-labels",
        help="Repair/export utility for an already-completed extract-interaction-labels run: zeroes out any "
             "interaction_labels.tif voxel whose label id is no longer in interaction_label_map.json (e.g. left "
             "over from a category rule change), and writes two new compressed per-category volumes alongside "
             "it (other_astrocyte_labels.tif, neuron_labels.tif). See "
             "io_utils.clean_and_split_interaction_labels's docstring for why the combined file stays "
             "uncompressed while the two new ones don't.",
    )
    p_clean_split.add_argument("--data-dir", required=True, help="Root output directory -- reads/writes under <data-dir>/astro_<root_id>/")
    p_clean_split.add_argument("--root-id", type=int, action="append", default=None, help="Root ID to clean/split (repeatable)")
    p_clean_split.add_argument("--all-fetched", action="store_true", help="Run for every astrocyte with an interaction_label_map.json under --data-dir")
    p_clean_split.add_argument("--skip-clean", action="store_true", help="Skip the orphan-cleaning pass (use when interaction_labels.tif is already known-clean, e.g. re-running just the category split)")

    return parser


def get_clean_astrocyte_df(client: MicronsClient):
    return aq.query_clean_astrocytes(
        client.cave,
        client.materialization_version,
        client.config["proofreading_table"],
        client.config.get("proofreading_status_value", "clean"),
    )


def cmd_list(client: MicronsClient):
    df = get_clean_astrocyte_df(client)
    if df.empty:
        print("No clean proofread astrocytes found.")
        return
    cols = [c for c in ["pt_root_id", "status", "valid_id"] if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\n{len(df)} clean proofread astrocyte(s) at materialization version {client.materialization_version}.")


def build_metadata(client: MicronsClient, root_id: int, extraction_cfg: dict, image_shape, mask_shape, info: dict) -> dict:
    return {
        "root_id": str(root_id),
        "dataset": client.dataset,
        "materialization_version": client.materialization_version,
        "materialization_timestamp": str(client.materialization_timestamp),
        "em_source": client.config["em"]["source"],
        "segmentation_source": client.config["segmentation"]["source"],
        "mesh_bbox_nm": {
            "min": info["mesh_bbox_min_nm"],
            "max": info["mesh_bbox_max_nm"],
        },
        "voxel_bbox": {
            "min": info["voxel_bbox_min"],
            "max": info["voxel_bbox_max"],
        },
        "bbox_was_clipped_or_shrunk": info["was_clipped"],
        "margin_um": extraction_cfg["margin_um"],
        "extraction_mode": extraction_cfg["mode"],
        "em_resolution_nm": client.em_resolution_nm,
        "seg_resolution_nm": client.seg_resolution_nm,
        "saved_axis_order": "ZYX",
        "image_shape": list(image_shape),
        "mask_shape": list(mask_shape),
        "positive_voxels": info["positive_voxels"],
        "occupancy": info["occupancy"],
        "streamed_to_disk": info.get("streamed_to_disk", False),
    }


def extract_one(client: MicronsClient, root_id: int, data_dir: str, extraction_cfg: dict, output_cfg: dict):
    out_dir = io_utils.astro_output_dir(data_dir, root_id)
    image_path = io_utils.image_path(out_dir)
    mask_path = io_utils.mask_path(out_dir)

    progress = io_utils.load_progress(io_utils.progress_path(out_dir))
    already_done = progress is not None and progress.get("merged") and os.path.isfile(image_path) and os.path.isfile(mask_path)
    if already_done:
        logger.info("root_id %d: image.tif/mask.tif already present and merged at %s -- reusing, no download.", root_id, out_dir)
        # generate the overlay stack too if wanted but missing (checked by directory
        # presence; there's no progress-file marker for "overlay stack already built")
        overlay_dir = os.path.join(out_dir, "astrocyte_overlays")
        if output_cfg.get("save_astrocyte_overlay_stack", False) and not os.path.isdir(overlay_dir):
            n_sections = io_utils.save_astrocyte_overlay_stack(
                out_dir, image_path, mask_path, downsample=output_cfg.get("overlay_downsample", 10),
            )
            logger.info("root_id %d: astrocyte_overlays/ was missing -- generated %d section(s) from existing image.tif/mask.tif.", root_id, n_sections)
        return out_dir

    logger.info("Extracting astrocyte root_id=%d", root_id)
    image_zyx, mask_zyx, info = ve.extract_astrocyte(client, root_id, extraction_cfg, out_paths={"image": image_path, "mask": mask_path})

    if info["streamed_to_disk"]:
        image_shape = mask_shape = info["shape_zyx"]
        if output_cfg.get("save_previews", True):
            io_utils.save_previews_from_disk(out_dir, image_path, mask_path)
    else:
        image_shape, mask_shape = image_zyx.shape, mask_zyx.shape
        io_utils.save_image_tiff(image_path, image_zyx)
        io_utils.save_mask_tiff(mask_path, mask_zyx)
        if output_cfg.get("save_previews", True):
            io_utils.save_previews(out_dir, image_zyx, mask_zyx)

    if output_cfg.get("save_astrocyte_overlay_stack", False):
        n_sections = io_utils.save_astrocyte_overlay_stack(
            out_dir, image_path, mask_path, downsample=output_cfg.get("overlay_downsample", 10),
        )
        logger.info("Saved %d per-section astrocyte overlay(s) -> %s/astrocyte_overlays/", n_sections, out_dir)

    metadata = build_metadata(client, root_id, extraction_cfg, image_shape, mask_shape, info)
    io_utils.save_metadata(io_utils.metadata_path(out_dir), metadata)

    logger.info(
        "Saved root_id=%d to %s (image.shape=%s, positive_voxels=%d, occupancy=%.6f)",
        root_id, out_dir, image_shape, info["positive_voxels"], info["occupancy"],
    )
    return out_dir


def cmd_extract(client: MicronsClient, args):
    extraction_cfg = copy.deepcopy(client.config["extraction"])
    if args.margin_um is not None:
        extraction_cfg["margin_um"] = {"x": args.margin_um, "y": args.margin_um, "z": args.margin_um}
    if args.mode is not None:
        extraction_cfg["mode"] = args.mode
    if args.max_voxels is not None:
        extraction_cfg["max_voxels"] = args.max_voxels
    if args.test_max_z_voxels is not None:
        extraction_cfg["test_max_z_voxels"] = args.test_max_z_voxels

    manual_ids = list(client.config.get("manual_root_ids", []))
    if args.root_id:
        manual_ids.extend(args.root_id)

    clean_df = None
    if args.all_clean:
        clean_df = get_clean_astrocyte_df(client)

    root_ids = aq.resolve_root_ids(manual_ids, use_query=args.all_clean, clean_df=clean_df)
    if not root_ids:
        print("No root_ids to extract. Use --root-id, --all-clean, or set manual_root_ids in config.yaml.")
        return

    output_cfg = client.config["output"]
    results = []
    for root_id in root_ids:
        try:
            out_dir = extract_one(client, root_id, args.data_dir, extraction_cfg, output_cfg)
            results.append((root_id, "OK", out_dir))
        except Exception as e:
            logger.exception("Extraction failed for root_id=%d", root_id)
            results.append((root_id, f"FAILED: {e}", None))

    print("\n=== Extraction summary ===")
    for root_id, status, out_dir in results:
        print(f"  {root_id}: {status}" + (f" -> {out_dir}" if out_dir else ""))
    return any(status.startswith("FAILED") for _, status, _ in results)


def _format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


def cmd_status(data_dir: str, root_id: int):
    """
    Report chunked-extraction progress for root_id, reading only local
    files; no CAVE or CloudVolume connection needed.
    """
    out_dir = os.path.join(data_dir, f"astro_{root_id}")
    prog_path = io_utils.progress_path(out_dir)
    progress = io_utils.load_progress(prog_path)

    if progress is None:
        print(f"No progress file found for root_id {root_id} at {prog_path}. Extraction hasn't started (or used a non-chunked mode).")
        return

    req = progress["request_key"]
    z0, z1, chunk_z = req["bbox_min_vx"][2], req["bbox_max_vx"][2], req["chunk_z_voxels"]
    n_total_slabs = -(-(z1 - z0) // chunk_z)
    n_done = len(progress["completed_slabs"])
    percent = 100.0 * n_done / n_total_slabs if n_total_slabs else 0.0

    if progress.get("merged"):
        print(f"root_id {root_id}: COMPLETE ({n_done}/{n_total_slabs} slabs, 100%)")
        print(f"  positive_voxels={progress['positive_voxels']:,}, total_voxels={progress['total_voxels']:,}, "
              f"occupancy={progress['positive_voxels']/progress['total_voxels']:.6f}")
        return

    started_at = progress.get("started_at")
    last_updated = progress.get("last_updated")
    if started_at is None or last_updated is None:
        image_slab_dir = io_utils.slab_dir(out_dir, "image")
        mtimes = [os.path.getmtime(os.path.join(image_slab_dir, f)) for f in os.listdir(image_slab_dir)] if os.path.isdir(image_slab_dir) else []
        started_at = min(mtimes) if mtimes else None
        last_updated = max(mtimes) if mtimes else None

    print(f"root_id {root_id}: IN PROGRESS -- {n_done}/{n_total_slabs} slabs ({percent:.1f}%)")
    print(f"  positive_voxels so far: {progress['positive_voxels']:,}")

    if started_at is not None and last_updated is not None and n_done > 0:
        elapsed = last_updated - started_at
        rate_s_per_slab = elapsed / n_done if n_done else None
        remaining_slabs = n_total_slabs - n_done
        if rate_s_per_slab:
            eta_s = remaining_slabs * rate_s_per_slab
            print(f"  elapsed: {_format_duration(elapsed)}, avg rate: {_format_duration(rate_s_per_slab)}/slab")
            print(f"  estimated remaining: {_format_duration(eta_s)} ({remaining_slabs} slabs left)")
    print(f"  progress file: {prog_path}")


def _discover_astro_dirs_with_metadata(output_root: str):
    """
    Yield (root_id, out_dir) for every astro_<root_id> directory directly
    under output_root that has a metadata.json.
    """
    if not os.path.isdir(output_root):
        return
    for name in sorted(os.listdir(output_root)):
        if not name.startswith("astro_"):
            continue
        out_dir = os.path.join(output_root, name)
        if not os.path.isdir(out_dir) or not os.path.isfile(io_utils.metadata_path(out_dir)):
            continue
        try:
            root_id = int(name[len("astro_"):])
        except ValueError:
            continue
        yield root_id, out_dir


def _discover_astro_dirs_with_run_marker(output_root: str, marker_filename: str):
    """
    (root_id, out_dir) for every astro_<root_id> dir whose latest
    "runs/<date>/" (or flat out_dir, pre-"runs/" convention) has
    marker_filename; backs every --all-fetched flow.
    """
    for root_id, out_dir in _discover_astro_dirs_with_metadata(output_root):
        if io_utils.latest_run_dir(out_dir, marker_filename) is not None:
            yield root_id, out_dir


def cmd_find_interactions(client: MicronsClient, args):
    output_root = args.data_dir
    categories = client.config.get("interaction_categories") or iq.default_categories()

    # queried once and reused across every astrocyte in this run, not per-astrocyte
    clean_astrocyte_df = get_clean_astrocyte_df(client)
    clean_astrocyte_root_ids = set(int(r) for r in clean_astrocyte_df["pt_root_id"])

    if args.all_fetched:
        targets = list(_discover_astro_dirs_with_metadata(output_root))
        if not targets:
            print(f"No extracted astrocytes (with metadata.json) found under {output_root}.")
            return
    elif args.root_id:
        targets = [(root_id, io_utils.astro_output_dir(output_root, root_id)) for root_id in args.root_id]
    else:
        print("Use --root-id (repeatable) or --all-fetched.")
        return

    any_failed = False
    for root_id, out_dir in targets:
        meta_path = io_utils.metadata_path(out_dir)
        if not os.path.isfile(meta_path):
            print(f"  {root_id}: SKIPPED (no metadata.json at {out_dir} -- run 'extract' first)")
            continue

        with open(meta_path) as f:
            metadata = json.load(f)
        bbox_min_nm, bbox_max_nm = iq.bbox_min_max_nm_from_metadata(metadata)

        try:
            result = iq.find_interactions(
                client.cave, client.materialization_version, root_id, bbox_min_nm, bbox_max_nm,
                categories=categories, clean_astrocyte_root_ids=clean_astrocyte_root_ids,
            )

            # Layer 3: every root_id present in the bbox's segmentation,
            # classified directly, catching axon-astrocyte contact that
            # layers 1-2 can't see. Only run when "neuron" is enabled, since
            # this is a full segmentation download, not a CAVE table query.
            bbox_segmentation_cells = []
            if categories.get("neuron", True):
                voxel_bbox = metadata["voxel_bbox"]
                bbox_min_vx = np.array(voxel_bbox["min"])
                bbox_max_vx = np.array(voxel_bbox["max"])
                enum_cfg = client.config.get("bbox_enumeration", {})
                min_voxels = enum_cfg.get("min_voxels", 50)
                max_memory_gb = client.config["extraction"].get("max_memory_gb", 8)
                configured_max_chunk_z = client.config["extraction"].get("chunk_z_voxels", 64)
                # Flat 16 bytes per voxel covers the raw uint64 array plus
                # np.unique's working copy, with headroom (no fixed output
                # dtype to size against here, unlike extract-interaction-labels).
                budget_voxels = int(max_memory_gb * 1e9 // 16)
                chunk_z_voxels = ve.safe_chunk_z_voxels(bbox_min_vx, bbox_max_vx, budget_voxels, configured_max_chunk_z)

                counts = ve.enumerate_root_ids_in_bbox(
                    client, bbox_min_vx, bbox_max_vx, chunk_z_voxels, min_voxels,
                    progress_path=io_utils.bbox_enumeration_progress_path(out_dir),
                )
                bbox_segmentation_cells = iq.find_bbox_segmentation_cells(
                    client.cave, client.materialization_version, root_id, counts.keys(),
                )
                for cell in bbox_segmentation_cells:
                    cell["voxel_count"] = counts[cell["root_id"]]

                # Candidates for a loose and strict "neuron" dual view: union of
                # layer 3's classification-confirmed hits and layer 2's
                # dendrite-tagged synapse post_root_ids; a tag-only root_id
                # can still be proofread-confirmed independently.
                neuron_candidate_root_ids = {
                    c["root_id"] for c in bbox_segmentation_cells if c["category"] == "neuron"
                }
                neuron_candidate_root_ids |= {
                    syn["post_root_id"] for syn in result["synapses_in_bbox"]
                    if iq.synapse_post_category(syn["compartment_tag"]) == "dendrite"
                }
                neuron_proofread_cfg = client.config.get("neuron_proofreading", {})
                require_axon = neuron_proofread_cfg.get("require_axon", True)
                proofreading_df = iq.query_neuron_proofreading_status(
                    client.cave, client.materialization_version, neuron_candidate_root_ids,
                )
                neuron_proofread_ids = iq.neuron_proofread_root_ids(proofreading_df, require_axon=require_axon)

                # Recompute both views to include layer 3 (network-free;
                # only the neuron-proofreading query above was new).
                result["bbox_segmentation_cells"] = bbox_segmentation_cells
                result["n_bbox_segmentation_root_ids"] = len(bbox_segmentation_cells)
                result["n_neuron_proofread_root_ids"] = len(neuron_proofread_ids)
                result["interacting_root_ids"] = iq.compute_interacting_root_ids(
                    root_id, result["cells_in_bbox"], result["synapses_in_bbox"], categories,
                    require_astrocyte_proofread=False, bbox_segmentation_cells=bbox_segmentation_cells,
                    require_neuron_proofread=False,
                )
                result["n_interacting_root_ids"] = len(result["interacting_root_ids"])
                result["interacting_root_ids_proofread"] = iq.compute_interacting_root_ids(
                    root_id, result["cells_in_bbox"], result["synapses_in_bbox"], categories,
                    require_astrocyte_proofread=True, bbox_segmentation_cells=bbox_segmentation_cells,
                    require_neuron_proofread=True, neuron_proofread_root_ids=neuron_proofread_ids,
                )
                result["n_interacting_root_ids_proofread"] = len(result["interacting_root_ids_proofread"])
        except Exception:
            logger.exception("find-interactions failed for root_id=%d", root_id)
            print(f"  {root_id}: FAILED (see log)")
            any_failed = True
            continue

        # config.yaml's pipeline_date if pinned, else today's date (see its comment there).
        pipeline_date = client.config.get("pipeline_date") or io_utils.run_date_today()
        this_run_dir = io_utils.run_dir(out_dir, pipeline_date)
        io_utils.save_metadata(io_utils.interaction_analysis_path(this_run_dir), result)

        type_counts: dict[str, int] = {}
        for cell in result["cells_in_bbox"]:
            label = cell["cell_type"] or "unclassified"
            type_counts[label] = type_counts.get(label, 0) + 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
        tag_breakdown = ", ".join(f"{k}={v}" for k, v in sorted(result["compartment_tag_counts"].items()))
        bbox_seg_note = ""
        if "bbox_segmentation_cells" in result:
            n_neuron = sum(1 for c in result["bbox_segmentation_cells"] if c["category"] == "neuron")
            bbox_seg_note = (
                f"; {result['n_bbox_segmentation_root_ids']} root_ids present in bbox segmentation "
                f"({n_neuron} classified neuron, any compartment), "
                f"{result['n_neuron_proofread_root_ids']} neuron-proofread-confirmed"
            )
        print(
            f"  {root_id}: {result['n_cells_in_bbox']} real cells in bbox ({breakdown}); "
            f"{result['n_synapses_in_bbox']} synapses in bbox ({tag_breakdown}){bbox_seg_note}; "
            f"{result['n_interacting_root_ids']} total interacting root_ids (category-filtered, "
            f"astrocyte=classifier, neuron=classifier+dendrite-tag), "
            f"{result['n_interacting_root_ids_proofread']} in the strict proofread-confirmed subset "
            f"(astrocyte+neuron both proofread-verified) -> {this_run_dir}/interaction_analysis.json"
        )
    return any_failed


def cmd_extract_interaction_labels(client: MicronsClient, args):
    output_root = args.data_dir
    output_cfg = client.config["output"]
    overlay_downsample = output_cfg.get("overlay_downsample", 10)
    save_overlay_stack = output_cfg.get("save_astrocyte_neuron_overlay_stack", True)
    save_synapse_overlays = output_cfg.get("save_synapse_overlay_stack", True)
    synapse_z_tolerance = output_cfg.get("synapse_overlay_z_tolerance", 2)
    keep_slabs = client.config["extraction"].get("keep_slabs", False)
    max_memory_gb = client.config["extraction"].get("max_memory_gb", 8)
    configured_max_chunk_z = client.config["extraction"].get("chunk_z_voxels", 64)

    if args.all_fetched:
        targets = list(_discover_astro_dirs_with_run_marker(output_root, io_utils.INTERACTION_ANALYSIS_FILENAME))
        if not targets:
            print(f"No astrocytes with interaction_analysis.json found under {output_root} -- run find-interactions first.")
            return
    elif args.root_id:
        targets = [(root_id, io_utils.astro_output_dir(output_root, root_id)) for root_id in args.root_id]
    else:
        print("Use --root-id (repeatable) or --all-fetched.")
        return

    results = []
    for root_id, out_dir in targets:
        # _progress.json's "merged" flag is chunked-mode-only; checking the
        # actual output files instead recognizes full_bbox extractions too.
        if not (os.path.isfile(io_utils.metadata_path(out_dir))
                and os.path.isfile(io_utils.image_path(out_dir))
                and os.path.isfile(io_utils.mask_path(out_dir))):
            print(f"  {root_id}: SKIPPED (no metadata.json/image.tif/mask.tif at {out_dir} -- run 'extract' first)")
            continue
        # Everything below writes back into this same run dir, so a chained
        # find-interactions -> extract-interaction-labels run stays together.
        this_run_dir = io_utils.latest_run_dir(out_dir, io_utils.INTERACTION_ANALYSIS_FILENAME)
        if this_run_dir is None:
            print(f"  {root_id}: SKIPPED (no interaction_analysis.json at {out_dir} -- run 'find-interactions' first)")
            continue
        analysis_path = io_utils.interaction_analysis_path(this_run_dir)

        with open(analysis_path) as f:
            interactions = json.load(f)
        cells_in_bbox = interactions["cells_in_bbox"]
        synapses = interactions.get("synapses_in_bbox", [])
        bbox_segmentation_cells = interactions.get("bbox_segmentation_cells", [])
        # Recomputed from the CURRENT interaction_categories (network-free),
        # not interactions["interacting_root_ids"]'s stale category set --
        # flipping a category off takes effect without re-running find-interactions.
        categories = client.config.get("interaction_categories") or iq.default_categories()
        interacting_root_ids = iq.compute_interacting_root_ids(
            root_id, cells_in_bbox, synapses, categories, bbox_segmentation_cells=bbox_segmentation_cells,
        )
        # Strict view is NOT recomputed like the loose one: it's a live
        # CAVE query result, so we trust find-interactions' saved answer;
        # missing key (older saved file) means "nothing strict-verified".
        strict_root_ids = set(interactions.get("interacting_root_ids_proofread", []))

        label_path = io_utils.interaction_labels_path(this_run_dir)
        # metadata.json's voxel_bbox is written for both extraction modes,
        # unlike _progress.json's request_key (chunked-only).
        with open(io_utils.metadata_path(out_dir)) as f:
            metadata = json.load(f)
        bbox_min_vx = np.array(metadata["voxel_bbox"]["min"])
        bbox_max_vx = np.array(metadata["voxel_bbox"]["max"])

        id_map = ve.build_interaction_id_map(root_id, interacting_root_ids)
        dtype = ve.select_interaction_label_dtype(len(id_map))
        budget_voxels = int(max_memory_gb * 1e9 // (8 + np.dtype(dtype).itemsize))
        chunk_z_voxels = ve.safe_chunk_z_voxels(bbox_min_vx, bbox_max_vx, budget_voxels, configured_max_chunk_z)
        logger.info(
            "root_id %d: interaction-label volume covers %d labeled cells (target + %d interacting) "
            "using dtype=%s, chunk_z_voxels=%d",
            root_id, len(id_map), len(id_map) - 1, dtype.name, chunk_z_voxels,
        )

        try:
            n_slices = ve.extract_interaction_labels_chunked(
                client, root_id, bbox_min_vx, bbox_max_vx, chunk_z_voxels, this_run_dir, label_path,
                id_map, dtype=dtype, keep_slabs=keep_slabs,
            )

            # interaction_labels.tif above is still candidate-based; this
            # is the real gate: zero any voxel beyond threshold_nm of the
            # target's own mask.tif; only >=1-surviving-voxel ids are kept.
            threshold_nm = float(client.config.get("contact_filter", {}).get("threshold_nm", 40.0))
            contact_request_key = {
                "root_id": int(root_id),
                "bbox_min_vx": [int(v) for v in bbox_min_vx],
                "bbox_max_vx": [int(v) for v in bbox_max_vx],
                "id_map_root_ids": sorted(int(k) for k in id_map),
                "threshold_nm": threshold_nm,
            }
            configured_max_tile_xy_voxels = client.config.get("contact_filter", {}).get("max_tile_xy_voxels")
            contact_chunk_z_voxels, contact_tile_y_voxels, contact_tile_x_voxels = cf.safe_contact_filter_chunk_and_tile_voxels(
                bbox_min_vx, bbox_max_vx, max_memory_gb, configured_max_chunk_z,
                client.resolution_nm, threshold_nm, configured_max_tile_xy_voxels,
            )
            logger.info(
                "root_id %d: contact filter chunk_z_voxels=%d, xy tile=%dx%d "
                "(z-halo=%d, y-halo=%d, x-halo=%d voxels each side)",
                root_id, contact_chunk_z_voxels, contact_tile_y_voxels, contact_tile_x_voxels,
                cf._halo_z_voxels(threshold_nm, float(client.resolution_nm[2])),
                cf._halo_z_voxels(threshold_nm, float(client.resolution_nm[1])),
                cf._halo_z_voxels(threshold_nm, float(client.resolution_nm[0])),
            )
            surviving_local_ids = cf.filter_interaction_labels_by_contact(
                this_run_dir, io_utils.mask_path(out_dir), label_path,
                client.resolution_nm, threshold_nm, contact_request_key,
                chunk_z_voxels=contact_chunk_z_voxels,
                tile_y_voxels=contact_tile_y_voxels, tile_x_voxels=contact_tile_x_voxels,
            )
            logger.info(
                "root_id %d: contact filter (threshold=%.0fnm) kept %d/%d candidate cells",
                root_id, threshold_nm, len(surviving_local_ids), len(id_map),
            )

            # bbox_segmentation_cells first, cells_in_bbox second: on
            # collision, keeps cells_in_bbox's real proofread_clean check
            # over layer 3's (always None there).
            cell_by_root_id = {c["root_id"]: c for c in bbox_segmentation_cells}
            cell_by_root_id.update({c["root_id"]: c for c in cells_in_bbox})
            # pre_root_id always gets "presynaptic"; post_root_id gets its
            # actual compartment tag when classified. A root_id can
            # accumulate both across different synapses.
            owner_tags: dict[int, set] = {}
            for syn in synapses:
                owner_tags.setdefault(syn["pre_root_id"], set()).add("presynaptic")
                if syn["post_category"] is not None:
                    owner_tags.setdefault(syn["post_root_id"], set()).add(syn["compartment_tag"])

            # Only candidates the contact filter actually kept (>=1 surviving
            # voxel) get an entry; a dropped root_id's voxels are already
            # zero throughout label_path, so leaving it out of the map here
            # just means nothing downstream (overlays, mesh export) ever
            # looks for it.
            entries = []
            for rid, local_id in sorted(id_map.items(), key=lambda kv: kv[1]):
                if local_id not in surviving_local_ids:
                    continue
                cell = cell_by_root_id.get(rid)
                entries.append({
                    "local_id": local_id,
                    "root_id": rid,
                    "is_target_astrocyte": rid == root_id,
                    "cell_type": cell["cell_type"] if cell else None,
                    "proofread_clean": cell["proofread_clean"] if cell else None,
                    "classification_system": cell["classification_system"] if cell else None,
                    "nucleus_id": cell["nucleus_id"] if cell else None,
                    "synapse_compartment_tags": sorted(owner_tags.get(rid, [])),
                    # True = in the strict (proofread) view; target astrocyte
                    # is always included, checked explicitly here.
                    "proofread": rid == root_id or rid in strict_root_ids,
                })
            io_utils.save_interaction_label_map(this_run_dir, entries)
            n_strict = sum(1 for e in entries if e["proofread"])

            detail = (
                f"{label_path} ({n_slices} slices, {len(entries)} contact-verified cells "
                f"of {len(id_map)} candidates, threshold={threshold_nm:.0f}nm; "
                f"{n_strict}/{len(entries)} pass the strict proofread view -- see each entry's "
                f"'proofread' field)"
            )
            img_path = io_utils.image_path(out_dir)
            if save_overlay_stack:
                io_utils.save_astrocyte_neuron_overlay_stack(
                    this_run_dir, img_path, label_path, len(id_map), downsample=overlay_downsample,
                )
                detail += f", per-section overlays (downsample={overlay_downsample}x) -> astrocyte_neuron_overlays/section_z*.png"

            if save_synapse_overlays and synapses:
                local_synapses = []
                for syn in synapses:
                    global_vx = cu.physical_nm_to_voxel(syn["position_nm"], client.resolution_nm)
                    local_vx = global_vx - bbox_min_vx  # XYZ
                    local_synapses.append({
                        "x_vx": float(local_vx[0]),
                        "y_vx": float(local_vx[1]),
                        "z_vx": int(round(local_vx[2])),
                        "compartment_tag": syn["compartment_tag"] or "unclassified",
                    })
                io_utils.save_synapse_marker_overlay(
                    this_run_dir, img_path, local_synapses, downsample=overlay_downsample, z_tolerance=synapse_z_tolerance,
                )
                detail += f", synapse-marker overlays -> synapse_overlays/section_z*.png"

            results.append((root_id, "OK", detail))
        except Exception as e:
            logger.exception("Interaction-label extraction failed for root_id=%d", root_id)
            results.append((root_id, f"FAILED: {e}", None))

    print("\n=== Interaction-label extraction summary ===")
    for root_id, status, detail in results:
        print(f"  {root_id}: {status}" + (f" -> {detail}" if detail else ""))
    return any(status.startswith("FAILED") for _, status, _ in results)


def cmd_fetch_meshes(client: MicronsClient, args):
    output_root = args.data_dir

    if args.all_fetched:
        targets = list(_discover_astro_dirs_with_run_marker(output_root, io_utils.INTERACTION_LABEL_MAP_FILENAME))
        if not targets:
            print(f"No astrocytes with interaction_label_map.json found under {output_root} -- run extract-interaction-labels first.")
            return
    elif args.root_id:
        targets = [(root_id, io_utils.astro_output_dir(output_root, root_id)) for root_id in args.root_id]
    else:
        print("Use --root-id (repeatable) or --all-fetched.")
        return

    any_failed = False
    for root_id, out_dir in targets:
        this_run_dir = io_utils.latest_run_dir(out_dir, io_utils.INTERACTION_LABEL_MAP_FILENAME)
        if this_run_dir is None:
            print(f"  {root_id}: SKIPPED (no interaction_label_map.json at {out_dir} -- run 'extract-interaction-labels' first)")
            continue

        try:
            results = mx.fetch_interacting_meshes(client, this_run_dir)
        except Exception:
            logger.exception("Mesh fetch failed for root_id=%d", root_id)
            print(f"  {root_id}: FAILED (see log)")
            any_failed = True
            continue

        by_category: dict[str, int] = {}
        failures = []
        for rid, category, status in results:
            by_category[category] = by_category.get(category, 0) + 1
            if status.startswith("FAILED"):
                failures.append((rid, status))

        breakdown = ", ".join(f"{cat}={n}" for cat, n in sorted(by_category.items()))
        print(f"  {root_id}: {len(results)} mesh(es) ({breakdown}), {len(failures)} failed -> {this_run_dir}/meshes/")
        for rid, status in failures:
            print(f"    {rid}: {status}")
        if failures:
            any_failed = True
    return any_failed


def cmd_reconstruct_local_meshes(client: MicronsClient, args):
    output_root = args.data_dir

    if args.all_fetched:
        targets = list(_discover_astro_dirs_with_run_marker(output_root, io_utils.INTERACTION_LABEL_MAP_FILENAME))
        if not targets:
            print(f"No astrocytes with interaction_label_map.json found under {output_root} -- run extract-interaction-labels first.")
            return
    elif args.root_id:
        targets = [(root_id, io_utils.astro_output_dir(output_root, root_id)) for root_id in args.root_id]
    else:
        print("Use --root-id (repeatable) or --all-fetched.")
        return

    any_failed = False
    for root_id, out_dir in targets:
        this_run_dir = io_utils.latest_run_dir(out_dir, io_utils.INTERACTION_LABEL_MAP_FILENAME)
        if this_run_dir is None:
            print(f"  {root_id}: SKIPPED (no interaction_label_map.json at {out_dir} -- run 'extract-interaction-labels' first)")
            continue

        try:
            results = mx.reconstruct_local_meshes(out_dir, this_run_dir, downsample=args.downsample)
        except Exception:
            logger.exception("Local mesh reconstruction failed for root_id=%d", root_id)
            print(f"  {root_id}: FAILED (see log)")
            any_failed = True
            continue

        by_category: dict[str, int] = {}
        failures = []
        skipped = []
        for rid, category, status in results:
            by_category[category] = by_category.get(category, 0) + 1
            if status.startswith("FAILED"):
                failures.append((rid, status))
            elif status.startswith("SKIPPED"):
                skipped.append((rid, status))

        breakdown = ", ".join(f"{cat}={n}" for cat, n in sorted(by_category.items()))
        print(
            f"  {root_id}: {len(results)} mesh(es) ({breakdown}), {len(skipped)} skipped (no voxels at this "
            f"downsample), {len(failures)} failed -> {this_run_dir}/meshes_local_ds{args.downsample}/"
        )
        for rid, status in failures:
            print(f"    {rid}: {status}")
        if failures:
            any_failed = True
    return any_failed


def cmd_clean_and_split_labels(client: MicronsClient, args):
    output_root = args.data_dir

    if args.all_fetched:
        targets = list(_discover_astro_dirs_with_run_marker(output_root, io_utils.INTERACTION_LABEL_MAP_FILENAME))
        if not targets:
            print(f"No astrocytes with interaction_label_map.json found under {output_root} -- run extract-interaction-labels first.")
            return
    elif args.root_id:
        targets = [(root_id, io_utils.astro_output_dir(output_root, root_id)) for root_id in args.root_id]
    else:
        print("Use --root-id (repeatable) or --all-fetched.")
        return

    any_failed = False
    for root_id, out_dir in targets:
        this_run_dir = io_utils.latest_run_dir(out_dir, io_utils.INTERACTION_LABEL_MAP_FILENAME)
        if this_run_dir is None:
            print(f"  {root_id}: SKIPPED (no interaction_label_map.json at {out_dir} -- run 'extract-interaction-labels' first)")
            continue

        id_map = io_utils.load_interaction_label_map(this_run_dir)
        try:
            result = io_utils.clean_and_split_interaction_labels(this_run_dir, id_map, skip_clean=args.skip_clean)
        except Exception:
            logger.exception("clean-and-split-labels failed for root_id=%d", root_id)
            print(f"  {root_id}: FAILED (see log)")
            any_failed = True
            continue

        print(
            f"  {root_id}: zeroed {result['n_voxels_zeroed']} orphan voxel(s); wrote "
            f"other_astrocyte_labels.tif ({result['other_astrocyte_ids']} ids), "
            f"neuron_labels.tif ({result['neuron_ids']} ids) -> {this_run_dir}/"
        )
    return any_failed


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.command == "status":
        cmd_status(args.data_dir, args.root_id)
        return

    client = MicronsClient(config)

    any_failed = False
    if args.command == "list":
        cmd_list(client)
    elif args.command == "extract":
        any_failed = cmd_extract(client, args)
    elif args.command == "find-interactions":
        any_failed = cmd_find_interactions(client, args)
    elif args.command == "extract-interaction-labels":
        any_failed = cmd_extract_interaction_labels(client, args)
    elif args.command == "fetch-meshes":
        any_failed = cmd_fetch_meshes(client, args)
    elif args.command == "reconstruct-local-meshes":
        any_failed = cmd_reconstruct_local_meshes(client, args)
    elif args.command == "clean-and-split-labels":
        any_failed = cmd_clean_and_split_labels(client, args)
    else:
        parser.print_help()
        sys.exit(1)

    # Per-root_id failures are caught and printed inline (so a batch run
    # keeps going across root_ids) but must still surface as a nonzero exit
    # code, or a caller like run_pipeline.py that checks returncode to
    # "stop on first failure" silently continues past a real failure.
    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
