# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""
Saves extracted volumes, metadata, and QC previews and overlays to
"outputs/astro_<root_id>/". Large-TIFF streaming functions read in Z-chunks
via tifffile.memmap then release each chunk with madvise or fadvise
DONTNEED, or RSS stays resident and can get the process OOM-killed.
"""
from __future__ import annotations

import datetime
import json
import mmap
import os

import numpy as np
import tifffile


def astro_output_dir(output_root: str, root_id: int) -> str:
    d = os.path.join(output_root, f"astro_{root_id}")
    os.makedirs(d, exist_ok=True)
    return d


IMAGE_FILENAME = "image.tif"
MASK_FILENAME = "mask.tif"
METADATA_FILENAME = "metadata.json"


def image_path(out_dir: str) -> str:
    return os.path.join(out_dir, IMAGE_FILENAME)


def mask_path(out_dir: str) -> str:
    return os.path.join(out_dir, MASK_FILENAME)


def metadata_path(out_dir: str) -> str:
    return os.path.join(out_dir, METADATA_FILENAME)


RUNS_DIRNAME = "runs"


def run_date_today() -> str:
    return datetime.date.today().isoformat()


def run_dir(out_dir: str, date_str: str | None = None) -> str:
    """Holds every derived analysis output at "astro_<root_id>/runs/<date_str>/"
    (interactions, label volumes, overlays, meshes) for one dated pass, kept
    separate from the raw image.tif, mask.tif, and metadata.json so re-running
    analysis never overwrites a previous run in place. Created if missing."""
    d = os.path.join(out_dir, RUNS_DIRNAME, date_str or run_date_today())
    os.makedirs(d, exist_ok=True)
    return d


def latest_run_dir(out_dir: str, marker_filename: str) -> str | None:
    """Most recent "runs/<date>/" dir containing marker_filename, newest first.
    Falls back to out_dir itself (pre-"runs/" layout) if found there instead.
    None if not found in either place."""
    runs_root = os.path.join(out_dir, RUNS_DIRNAME)
    if os.path.isdir(runs_root):
        for name in sorted(os.listdir(runs_root), reverse=True):
            candidate = os.path.join(runs_root, name)
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, marker_filename)):
                return candidate
    if os.path.isfile(os.path.join(out_dir, marker_filename)):
        return out_dir
    return None


def resolve_script_dirs(run_dir_arg: str | None = None, data_dir_arg: str | None = None) -> tuple[str, str]:
    """Resolves (out_dir, run_dir) pair from CLI arguments."""
    if run_dir_arg:
        r_dir = os.path.abspath(run_dir_arg)
        o_dir = data_dir_arg or os.path.dirname(os.path.dirname(r_dir))
        return o_dir, r_dir
    elif data_dir_arg:
        o_dir = os.path.abspath(data_dir_arg)
        r_dir = latest_run_dir(o_dir, "interaction_label_map.json") or run_dir(o_dir)
        return o_dir, r_dir
    else:
        raise ValueError("Must specify either --run-dir or --data-dir")



def save_image_tiff(path: str, image_zyx: np.ndarray):
    tifffile.imwrite(path, image_zyx, photometric="minisblack")


def save_mask_tiff(path: str, mask_zyx: np.ndarray):
    tifffile.imwrite(path, mask_zyx.astype(np.uint8), photometric="minisblack")


def save_metadata(path: str, metadata: dict):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


PROGRESS_FILENAME = "_progress.json"
INTERACTION_LABEL_PROGRESS_FILENAME = "_interaction_label_progress.json"
BBOX_ENUMERATION_PROGRESS_FILENAME = "_bbox_enumeration_progress.json"
INTERACTION_LABEL_MAP_FILENAME = "interaction_label_map.json"
INTERACTION_ANALYSIS_FILENAME = "interaction_analysis.json"  # cells_in_bbox + synapses_in_bbox + derived interacting_root_ids sets
INTERACTION_LABELS_FILENAME = "interaction_labels.tif"


def progress_path(out_dir: str) -> str:
    return os.path.join(out_dir, PROGRESS_FILENAME)


def interaction_labels_path(run_dir: str) -> str:
    return os.path.join(run_dir, INTERACTION_LABELS_FILENAME)


def other_astrocyte_labels_path(run_dir: str) -> str:
    return os.path.join(run_dir, "other_astrocyte_labels.tif")


def neuron_labels_path(run_dir: str) -> str:
    return os.path.join(run_dir, "neuron_labels.tif")


def interaction_label_progress_path(out_dir: str) -> str:
    return os.path.join(out_dir, INTERACTION_LABEL_PROGRESS_FILENAME)


def bbox_enumeration_progress_path(out_dir: str) -> str:
    return os.path.join(out_dir, BBOX_ENUMERATION_PROGRESS_FILENAME)


def interaction_label_map_path(out_dir: str) -> str:
    return os.path.join(out_dir, INTERACTION_LABEL_MAP_FILENAME)


def interaction_analysis_path(out_dir: str) -> str:
    return os.path.join(out_dir, INTERACTION_ANALYSIS_FILENAME)


def save_interaction_label_map(out_dir: str, entries: list):
    """entries: one dict per labeled cell (local_id, root_id, cell_type,
    classification_system, etc.; see interaction_query.py)."""
    save_progress(interaction_label_map_path(out_dir), {"entries": entries})


def load_interaction_label_map(out_dir: str):
    record = load_progress(interaction_label_map_path(out_dir))
    return record["entries"] if record else None


def clean_and_split_interaction_labels(run_dir: str, id_map: list, chunk_z: int = 8, compression: str = "zlib", skip_clean: bool = False) -> dict:
    """Repair and export: zeroes orphan-labeled voxels (not in id_map) in place,
    then writes COMPRESSED other_astrocyte_labels.tif and neuron_labels.tif
    (combined file stays uncompressed for memmap access). Three sequential
    single-chunk passes for a low memory ceiling. Not resumable but safe to
    rerun; skip_clean=True skips zeroing."""
    other_astro_ids = set()
    neuron_ids = set()
    valid_ids = set()
    for e in id_map:
        cat = _entry_category(e)
        valid_ids.add(e["local_id"])
        if cat == "astrocyte":
            other_astro_ids.add(e["local_id"])
        elif cat == "neuron":
            neuron_ids.add(e["local_id"])

    label_path = interaction_labels_path(run_dir)

    def _grow_luts(max_id, chunk_max, luts):
        if chunk_max <= max_id:
            return max_id, luts
        max_id = chunk_max
        return max_id, tuple(np.pad(lut, (0, max_id + 1 - len(lut))) for lut in luts)

    page_size = mmap.PAGESIZE
    n_zeroed = 0
    if not skip_clean:
        keep_lut = np.zeros(max(valid_ids) + 1, dtype=bool)
        keep_lut[list(valid_ids)] = True
        keep_lut[0] = True
        max_id = max(valid_ids)

        src = tifffile.memmap(label_path, mode="r+")
        nz, ny, nx = src.shape
        dtype = src.dtype
        bytes_per_z = ny * nx * dtype.itemsize
        mmap_len = len(src._mmap)
        fd = os.open(label_path, os.O_RDONLY)

        for z0 in range(0, nz, chunk_z):
            z1 = min(z0 + chunk_z, nz)
            chunk = np.array(src[z0:z1])

            max_id, (keep_lut,) = _grow_luts(max_id, int(chunk.max()), (keep_lut,))

            zero_mask = ~keep_lut[chunk]
            n_zeroed += int(zero_mask.sum())
            if zero_mask.any():
                chunk[zero_mask] = 0
                src[z0:z1] = chunk

            del chunk, zero_mask
            byte_start = z0 * bytes_per_z
            byte_end = min(z1 * bytes_per_z, mmap_len)
            aligned_start = (byte_start // page_size) * page_size
            if byte_end > aligned_start:
                src._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)
            os.posix_fadvise(fd, byte_start, byte_end - byte_start, os.POSIX_FADV_DONTNEED)

            if z0 % (chunk_z * 10) == 0:
                print(f"...clean-and-split-labels: cleaning z={z0}/{nz} (zeroed so far: {n_zeroed})", flush=True)

        src._mmap.flush()
        del src
        os.close(fd)
    print(f"...clean-and-split-labels: cleaning done, {n_zeroed} voxel(s) zeroed.", flush=True)

    def _write_split(out_path, id_set, label):
        lut = np.zeros(max(max(id_set, default=0), 1) + 1, dtype=bool)
        lut[list(id_set)] = True
        local_max_id = len(lut) - 1

        tmp_path = out_path + ".tmp"
        src = tifffile.memmap(label_path, mode="r")
        nz, ny, nx = src.shape
        bytes_per_z = ny * nx * src.dtype.itemsize
        mmap_len = len(src._mmap)
        fd = os.open(label_path, os.O_RDONLY)

        def _pages():
            nonlocal lut, local_max_id
            try:
                for z0 in range(0, nz, chunk_z):
                    z1 = min(z0 + chunk_z, nz)
                    chunk = np.array(src[z0:z1])
                    local_max_id_new, (lut,) = _grow_luts(local_max_id, int(chunk.max()), (lut,))
                    local_max_id = local_max_id_new
                    out_chunk = np.where(lut[chunk], chunk, 0).astype(src.dtype)

                    byte_start = z0 * bytes_per_z
                    byte_end = min(z1 * bytes_per_z, mmap_len)
                    aligned_start = (byte_start // page_size) * page_size
                    if byte_end > aligned_start:
                        src._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)
                    os.posix_fadvise(fd, byte_start, byte_end - byte_start, os.POSIX_FADV_DONTNEED)

                    if z0 % (chunk_z * 10) == 0:
                        print(f"...clean-and-split-labels: {label} z={z0}/{nz}", flush=True)

                    for i in range(z1 - z0):
                        yield out_chunk[i]
                    del chunk, out_chunk
            finally:
                os.close(fd)

        # a compressed TIFF series needs one write() call with shape and dtype stated
        # upfront; repeated per-page write() calls don't reliably merge into one
        # coherent series on read-back, so this streams pages via a generator instead
        with tifffile.TiffWriter(tmp_path, bigtiff=True) as w:
            w.write(_pages(), shape=(nz, ny, nx), dtype=src.dtype, compression=compression, photometric="minisblack")

        del src
        os.replace(tmp_path, out_path)
        print(f"...clean-and-split-labels: {label} done -> {out_path}", flush=True)

    _write_split(other_astrocyte_labels_path(run_dir), other_astro_ids, "other_astrocyte")
    _write_split(neuron_labels_path(run_dir), neuron_ids, "neuron")

    return {
        "n_voxels_zeroed": n_zeroed,
        "other_astrocyte_ids": len(other_astro_ids),
        "neuron_ids": len(neuron_ids),
    }


def save_progress(path: str, progress: dict):
    """Atomic write (temp file + rename) so a crash mid-write can't leave a corrupt file."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(progress, f, default=_json_default)
    os.replace(tmp_path, path)


def load_progress(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def slab_dir(out_dir: str, kind: str) -> str:
    d = os.path.join(out_dir, "_slabs", kind)
    os.makedirs(d, exist_ok=True)
    return d


def slab_file_path(a_slab_dir: str, slab_start: int, slab_end: int) -> str:
    return os.path.join(a_slab_dir, f"{slab_start:012d}_{slab_end:012d}.tif")


def write_slab(path: str, slab_zyx: np.ndarray, dtype):
    """One Z-slab as one small, complete TIFF file (single tifffile.imwrite
    call): a killed process leaves every already-finished slab valid, so
    resuming just means "redo whichever slab file is missing"."""
    tifffile.imwrite(path, np.ascontiguousarray(slab_zyx, dtype=dtype), photometric="minisblack")


def peek_tiff_shape(path: str):
    with tifffile.TiffFile(path) as tf:
        return tf.series[0].shape


def merge_slabs_to_stack(a_slab_dir: str, out_path: str, slab_ranges, dtype, yx_shape) -> int:
    """Concatenates per-slab TIFFs into one BigTIFF via a memmapped output
    array (repeated TiffWriter.write() calls don't merge 3D slabs reliably
    on read-back), madvise(DONTNEED) per slab to keep peak memory ~one slab.
    Returns total Z slices written."""
    slab_ranges = sorted(slab_ranges)
    total_z = sum(end - start for start, end in slab_ranges)
    final_shape = (total_z,) + tuple(yx_shape)
    bytes_per_slice = int(np.prod(yx_shape)) * np.dtype(dtype).itemsize
    page_size = mmap.PAGESIZE

    mm = tifffile.memmap(out_path, shape=final_shape, dtype=dtype, bigtiff=True, photometric="minisblack")
    # mm._mmap covers only the array's data region (starting at mm.offset in the
    # file); offsets into it are array-relative, do not add mm.offset
    mmap_len = len(mm._mmap)
    z_offset = 0
    for slab_start, slab_end in slab_ranges:
        path = slab_file_path(a_slab_dir, slab_start, slab_end)
        slab = tifffile.imread(path).astype(dtype, copy=False)
        n = slab.shape[0]
        mm[z_offset : z_offset + n] = slab
        mm.flush()
        byte_start = z_offset * bytes_per_slice
        byte_end = min(byte_start + n * bytes_per_slice, mmap_len)
        aligned_start = (byte_start // page_size) * page_size
        if byte_end > aligned_start:
            mm._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)
        z_offset += n
    del mm
    return total_z


def read_central_slices(path: str, chunk_z: int = 10):
    """Central XY, XZ, and YZ slices of a saved TIFF stack via memmap, without
    loading the full volume: reads in Z-chunks (not a naive constant-X-and-Y
    column read, which would fault in nearly the whole file for a
    Z-contiguous array) and releases each chunk via madvise. Returns
    (xy_slice, xz_slice, yz_slice)."""
    arr = tifffile.memmap(path)
    nz, ny, nx = arr.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    xy = np.array(arr[cz, :, :])

    xz = np.empty((nz, nx), dtype=arr.dtype)
    yz = np.empty((nz, ny), dtype=arr.dtype)

    bytes_per_plane = ny * nx * arr.dtype.itemsize
    mmap_len = len(arr._mmap)
    page_size = mmap.PAGESIZE
    for z0 in range(0, nz, chunk_z):
        z1 = min(z0 + chunk_z, nz)
        chunk = np.array(arr[z0:z1])
        xz[z0:z1] = chunk[:, cy, :]
        yz[z0:z1] = chunk[:, :, cx]
        byte_start = z0 * bytes_per_plane
        byte_end = min(z1 * bytes_per_plane, mmap_len)
        aligned_start = (byte_start // page_size) * page_size
        if byte_end > aligned_start:
            arr._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)

    return xy, xz, yz


def _save_preview_pair(out_path: str, img_slice: np.ndarray, mask_slice: np.ndarray):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img_slice, cmap="gray")
    ax[0].set_title("EM")
    ax[0].axis("off")

    ax[1].imshow(img_slice, cmap="gray")
    ax[1].imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap="autumn", alpha=0.5)
    ax[1].set_title("EM + mask overlay")
    ax[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def save_previews(out_dir: str, image_zyx: np.ndarray, mask_zyx: np.ndarray):
    """Lightweight central-slice QC previews (xy, xz, and yz) with mask outline; sanity check only."""
    nz, ny, nx = image_zyx.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    slices = {
        "preview_xy.png": (image_zyx[cz, :, :], mask_zyx[cz, :, :]),
        "preview_xz.png": (image_zyx[:, cy, :], mask_zyx[:, cy, :]),
        "preview_yz.png": (image_zyx[:, :, cx], mask_zyx[:, :, cx]),
    }
    for fname, (img_slice, mask_slice) in slices.items():
        _save_preview_pair(os.path.join(out_dir, fname), img_slice, mask_slice)


def save_previews_from_disk(out_dir: str, image_path: str, mask_path: str):
    """Same as save_previews, for volumes too large for memory; reads only
    the three central slices back via memmap."""
    img_xy, img_xz, img_yz = read_central_slices(image_path)
    mask_xy, mask_xz, mask_yz = read_central_slices(mask_path)

    _save_preview_pair(os.path.join(out_dir, "preview_xy.png"), img_xy, mask_xy)
    _save_preview_pair(os.path.join(out_dir, "preview_xz.png"), img_xz, mask_xz)
    _save_preview_pair(os.path.join(out_dir, "preview_yz.png"), img_yz, mask_yz)


# _save_mask_overlay_single's constant-1 masked array happens to render this exact
# red under cmap="autumn" (Normalize collapses a single-valued array to 0.0, not
# 1.0). Hardcoded and shared here instead of re-deriving it from that edge case, so
# every place that colors "the target astrocyte" agrees by construction.
_TARGET_ASTROCYTE_COLOR = (1.0, 0.0, 0.0, 1.0)


def save_astrocyte_legend(out_path: str):
    """One-entry legend for the "astrocyte_overlays/" stack."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    handle = mpatches.Patch(color=_TARGET_ASTROCYTE_COLOR, label="target astrocyte")
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis("off")
    ax.legend(handles=[handle], loc="center", fontsize=9, frameon=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _overlay_progress_path(overlay_dir: str) -> str:
    return os.path.join(overlay_dir, "_overlay_progress.json")


def save_astrocyte_overlay_stack(out_dir: str, image_path: str, mask_path: str, downsample: int = 10, chunk_z: int = 10, resume: bool = True):
    """One EM+mask overlay PNG per XY section under "astrocyte_overlays/",
    plus legend.png. resume=True checkpoints highest fully-written z to
    _overlay_progress.json; only that marker is trusted, not an existing
    PNG's mere presence (could be a stale render)."""
    import matplotlib

    matplotlib.use("Agg")

    overlay_dir = os.path.join(out_dir, "astrocyte_overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    save_astrocyte_legend(os.path.join(overlay_dir, "legend.png"))

    prog_path = _overlay_progress_path(overlay_dir)
    start_z = 0
    if resume:
        progress = load_progress(prog_path)
        if progress:
            if progress.get("completed"):
                return progress["n_slices"]
            start_z = progress.get("last_completed_z", -1) + 1

    img_arr = tifffile.memmap(image_path)
    mask_arr = tifffile.memmap(mask_path)
    nz, ny, nx = mask_arr.shape

    img_bytes_per_plane = ny * nx * img_arr.dtype.itemsize
    mask_bytes_per_plane = ny * nx * mask_arr.dtype.itemsize
    img_mmap_len = len(img_arr._mmap)
    mask_mmap_len = len(mask_arr._mmap)
    page_size = mmap.PAGESIZE

    for z0 in range(0, nz, chunk_z):
        z1 = min(z0 + chunk_z, nz)
        if z1 <= start_z:
            continue
        img_chunk = np.array(img_arr[z0:z1])
        mask_chunk = np.array(mask_arr[z0:z1])

        for i, z in enumerate(range(z0, z1)):
            if z < start_z:
                continue
            img_slice = img_chunk[i]
            mask_slice = mask_chunk[i]
            if downsample > 1:
                img_slice = img_slice[::downsample, ::downsample]
                mask_slice = mask_slice[::downsample, ::downsample]
            out_path = os.path.join(overlay_dir, f"section_z{z:05d}.png")
            _save_mask_overlay_single(out_path, img_slice, mask_slice)
            save_progress(prog_path, {"last_completed_z": z, "completed": False})

        for arr, mmap_len, bytes_per_plane in (
            (img_arr, img_mmap_len, img_bytes_per_plane),
            (mask_arr, mask_mmap_len, mask_bytes_per_plane),
        ):
            byte_start = z0 * bytes_per_plane
            byte_end = min(z1 * bytes_per_plane, mmap_len)
            aligned_start = (byte_start // page_size) * page_size
            if byte_end > aligned_start:
                arr._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)

    save_progress(prog_path, {"last_completed_z": nz - 1, "completed": True, "n_slices": nz})
    return nz


# Category (not per-instance) coloring for interaction_labels.tif: with thousands
# of interacting root_ids in a typical bbox, per-instance color is unreadable and
# the legend useless; a fixed 4-way scheme is what's actually legible.
_ASTROCYTE_CATEGORY_COLOR = (0.0, 0.4, 1.0, 1.0)    # blue: interacting astrocytes, not the target
_NEURON_CATEGORY_COLOR = (1.0, 0.55, 0.0, 1.0)      # orange: classification_system match, any compartment
_DENDRITE_CATEGORY_COLOR = (0.16, 0.75, 0.16, 1.0)  # green; see _entry_category (currently unreachable, kept for record-shape compatibility)
_CATEGORY_COLORS = {
    "target_astrocyte": _TARGET_ASTROCYTE_COLOR,
    "astrocyte": _ASTROCYTE_CATEGORY_COLOR,
    "neuron": _NEURON_CATEGORY_COLOR,
    "dendrite": _DENDRITE_CATEGORY_COLOR,
}
_CATEGORY_LABELS = {
    "target_astrocyte": "target astrocyte",
    "astrocyte": "other astrocyte",
    "neuron": "neuron",
    "dendrite": "dendrite",
}


def _entry_category(entry: dict) -> str:
    """4-way bucket for an interaction_label_map.json entry; classification_
    system and cell_type=="astrocyte" never both apply to one entry. The
    "dendrite" fallback is currently unreachable but kept as a defensive default."""
    if entry.get("is_target_astrocyte"):
        return "target_astrocyte"
    if entry.get("classification_system") in ("excitatory_neuron", "inhibitory_neuron"):
        return "neuron"
    if entry.get("cell_type") == "astrocyte":
        return "astrocyte"
    return "dendrite"


def _build_category_color_lut(id_map: list, max_label: int) -> np.ndarray:
    """RGBA lookup table for label ids 0..max_label (index 0 = background,
    transparent). Shared by _label_overlay_rgba and save_label_legend so image
    and legend always agree."""
    lut = np.zeros((max_label + 1, 4))
    for entry in id_map:
        lut[entry["local_id"]] = _CATEGORY_COLORS[_entry_category(entry)]
    return lut


def _label_overlay_rgba(label_slice: np.ndarray, id_map: list, max_label: int):
    if max_label <= 0:
        return np.zeros((*label_slice.shape, 4))
    lut = _build_category_color_lut(id_map, max_label)
    return lut[np.clip(label_slice, 0, max_label)]


def save_label_legend(out_path: str, id_map: list, ncol: int = 1):
    """Up-to-4-swatch legend (target astrocyte, other astrocytes, neuron, and
    dendrite), each with its count."""
    import matplotlib
    from collections import Counter

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    counts = Counter(_entry_category(e) for e in id_map)
    handles = [
        mpatches.Patch(color=_CATEGORY_COLORS[cat], label=f"{_CATEGORY_LABELS[cat]} ({counts[cat]})")
        for cat in ("target_astrocyte", "astrocyte", "neuron", "dendrite")
        if counts.get(cat)
    ]

    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=ncol, fontsize=9, frameon=True,
              title=f"{len(id_map)} labeled cells")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_astrocyte_neuron_overlay_single(out_path: str, img_slice: np.ndarray, label_slice: np.ndarray, id_map: list, max_label: int):
    """Single-panel EM+label composite. figsize matches the array's exact
    pixel dims so one array pixel = one output pixel (no antialiased
    blending of the label color). vmin and vmax pinned to 0-255, not
    matplotlib's per-slice auto-stretch, so brightness stays consistent."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    ny, nx = img_slice.shape
    fig = plt.figure(figsize=(nx / dpi, ny / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_slice, cmap="gray", vmin=0, vmax=255, interpolation="none")
    ax.imshow(_label_overlay_rgba(label_slice, id_map, max_label), alpha=0.6, interpolation="none")
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_mask_overlay_single(out_path: str, img_slice: np.ndarray, mask_slice: np.ndarray):
    """Single-panel EM+astrocyte-mask composite; see
    _save_astrocyte_neuron_overlay_single for the figsize, vmin, and vmax reasoning."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgba = np.zeros((*mask_slice.shape, 4))
    rgba[mask_slice != 0] = _TARGET_ASTROCYTE_COLOR

    dpi = 100
    ny, nx = img_slice.shape
    fig = plt.figure(figsize=(nx / dpi, ny / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_slice, cmap="gray", vmin=0, vmax=255, interpolation="none")
    ax.imshow(rgba, alpha=0.5, interpolation="none")
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_astrocyte_neuron_overlay_stack(out_dir: str, image_path: str, label_path: str, max_label: int, downsample: int = 10, chunk_z: int = 10, resume: bool = True):
    """One EM+label overlay PNG per XY section under
    "astrocyte_neuron_overlays/" (fixed color LUT, one shared legend.png).
    Uses both madvise and posix_fadvise per chunk: madvise alone was
    insufficient against some hosts' low-memory guard. resume=True: same
    checkpointing as save_astrocyte_overlay_stack."""
    import matplotlib

    matplotlib.use("Agg")

    id_map = load_interaction_label_map(out_dir)
    if id_map is None:
        raise ValueError(f"No interaction_label_map.json under {out_dir} -- category coloring needs it; run save_interaction_label_map first.")
    overlay_dir = os.path.join(out_dir, "astrocyte_neuron_overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    save_label_legend(os.path.join(overlay_dir, "legend.png"), id_map)

    prog_path = _overlay_progress_path(overlay_dir)
    start_z = 0
    if resume:
        progress = load_progress(prog_path)
        if progress:
            if progress.get("completed"):
                return progress["n_slices"]
            start_z = progress.get("last_completed_z", -1) + 1

    img_arr = tifffile.memmap(image_path)
    label_arr = tifffile.memmap(label_path)
    nz, ny, nx = label_arr.shape

    img_bytes_per_plane = ny * nx * img_arr.dtype.itemsize
    label_bytes_per_plane = ny * nx * label_arr.dtype.itemsize
    img_mmap_len = len(img_arr._mmap)
    label_mmap_len = len(label_arr._mmap)
    page_size = mmap.PAGESIZE

    img_fd = os.open(image_path, os.O_RDONLY)
    label_fd = os.open(label_path, os.O_RDONLY)
    try:
        for z0 in range(0, nz, chunk_z):
            z1 = min(z0 + chunk_z, nz)
            if z1 <= start_z:
                continue
            img_chunk = np.array(img_arr[z0:z1])
            label_chunk = np.array(label_arr[z0:z1])

            for i, z in enumerate(range(z0, z1)):
                if z < start_z:
                    continue
                img_slice = img_chunk[i]
                label_slice = label_chunk[i]
                if downsample > 1:
                    img_slice = img_slice[::downsample, ::downsample]
                    label_slice = label_slice[::downsample, ::downsample]
                out_path = os.path.join(overlay_dir, f"section_z{z:05d}.png")
                _save_astrocyte_neuron_overlay_single(out_path, img_slice, label_slice, id_map, max_label)
                save_progress(prog_path, {"last_completed_z": z, "completed": False})

            for arr, fd, mmap_len, bytes_per_plane in (
                (img_arr, img_fd, img_mmap_len, img_bytes_per_plane),
                (label_arr, label_fd, label_mmap_len, label_bytes_per_plane),
            ):
                byte_start = z0 * bytes_per_plane
                byte_end = min(z1 * bytes_per_plane, mmap_len)
                aligned_start = (byte_start // page_size) * page_size
                if byte_end > aligned_start:
                    arr._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)
                    os.posix_fadvise(fd, aligned_start, byte_end - aligned_start, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(img_fd)
        os.close(label_fd)

    save_progress(prog_path, {"last_completed_z": nz - 1, "completed": True, "n_slices": nz})
    return nz


_CONTACT_DENDRITE_COLOR = np.array([51, 204, 51], dtype=np.float32)   # green
_CONTACT_ASTROCYTE_COLOR = np.array([255, 204, 0], dtype=np.float32)  # gold or yellow
_CONTACT_ALPHA = 0.55


def save_contact_filtered_overlay_stack(out_dir: str, image_path: str, mask_path: str, label_path: str, downsample: int = 10, chunk_z: int = 10, threshold_nm: float = 40.0, native_xy_resolution_nm: float = 8.0, resume: bool = True):
    """QC-only overlay: 2D XY distance transform on the DOWNSAMPLED mask
    (not contact_filter.py's real 3D native-resolution check); fine for
    eyeballing, not a source of truth. Target=gold; dendrite pixels within
    threshold_nm of the mask=green; else plain EM (raw uint8 RGB via PIL).
    resume=True: same checkpointing as the other overlay-stack functions."""
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    id_map = load_interaction_label_map(out_dir)
    if id_map is None:
        raise ValueError(f"No interaction_label_map.json under {out_dir} -- dendrite filtering needs it; run save_interaction_label_map first.")
    max_local_id = max(e["local_id"] for e in id_map)
    dendrite_lut = np.zeros(max_local_id + 1, dtype=bool)
    for e in id_map:
        if _entry_category(e) == "dendrite":
            dendrite_lut[e["local_id"]] = True

    overlay_dir = os.path.join(out_dir, f"contact_filtered_overlays_{int(threshold_nm)}nm_ds10")
    os.makedirs(overlay_dir, exist_ok=True)

    prog_path = _overlay_progress_path(overlay_dir)
    start_z = 0
    if resume:
        progress = load_progress(prog_path)
        if progress:
            if progress.get("completed"):
                return progress["n_slices"]
            start_z = progress.get("last_completed_z", -1) + 1

    xy_resolution_nm = np.array([native_xy_resolution_nm * downsample] * 2)

    img_arr = tifffile.memmap(image_path)
    mask_arr = tifffile.memmap(mask_path)
    label_arr = tifffile.memmap(label_path)
    nz, ny, nx = label_arr.shape

    img_bytes_per_plane = ny * nx * img_arr.dtype.itemsize
    mask_bytes_per_plane = ny * nx * mask_arr.dtype.itemsize
    label_bytes_per_plane = ny * nx * label_arr.dtype.itemsize
    img_mmap_len = len(img_arr._mmap)
    mask_mmap_len = len(mask_arr._mmap)
    label_mmap_len = len(label_arr._mmap)
    page_size = mmap.PAGESIZE

    for z0 in range(0, nz, chunk_z):
        z1 = min(z0 + chunk_z, nz)
        if z1 <= start_z:
            continue
        img_chunk = np.array(img_arr[z0:z1])
        mask_chunk = np.array(mask_arr[z0:z1])
        label_chunk = np.array(label_arr[z0:z1])

        for i, z in enumerate(range(z0, z1)):
            if z < start_z:
                continue
            em_slice = img_chunk[i][::downsample, ::downsample]
            astro_mask = mask_chunk[i][::downsample, ::downsample].astype(bool)
            label_slice = label_chunk[i][::downsample, ::downsample]

            dist_nm = distance_transform_edt(~astro_mask, sampling=xy_resolution_nm)
            is_dendrite = dendrite_lut[np.clip(label_slice, 0, max_local_id)]
            keep_dendrite = is_dendrite & (dist_nm <= threshold_nm)

            rgb = np.stack([em_slice] * 3, axis=-1).astype(np.float32)
            rgb[astro_mask] = rgb[astro_mask] * (1 - _CONTACT_ALPHA) + _CONTACT_ASTROCYTE_COLOR * _CONTACT_ALPHA
            rgb[keep_dendrite] = rgb[keep_dendrite] * (1 - _CONTACT_ALPHA) + _CONTACT_DENDRITE_COLOR * _CONTACT_ALPHA
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            out_path = os.path.join(overlay_dir, f"section_z{z:05d}.png")
            Image.fromarray(rgb).save(out_path, optimize=False)
            save_progress(prog_path, {"last_completed_z": z, "completed": False})

        for arr, mmap_len, bytes_per_plane in (
            (img_arr, img_mmap_len, img_bytes_per_plane),
            (mask_arr, mask_mmap_len, mask_bytes_per_plane),
            (label_arr, label_mmap_len, label_bytes_per_plane),
        ):
            byte_start = z0 * bytes_per_plane
            byte_end = min(z1 * bytes_per_plane, mmap_len)
            aligned_start = (byte_start // page_size) * page_size
            if byte_end > aligned_start:
                arr._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)

    save_progress(prog_path, {"last_completed_z": nz - 1, "completed": True, "n_slices": nz})
    return nz


# Fixed colors per compartment tag (few distinct tags; a small fixed legend beats a hashed one)
SYNAPSE_TAG_COLORS = {
    "soma": "#e41a1c",
    "shaft": "#377eb8",
    "spine": "#4daf4a",
    "soma_spine": "#984ea3",
    "axon_bouton": "#ff7f00",
    "orphan": "#999999",
    "other": "#999999",
}
SYNAPSE_TAG_DEFAULT_COLOR = "#999999"


def save_synapse_marker_overlay(out_dir: str, image_path: str, synapses: list, downsample: int = 10, chunk_z: int = 10, z_tolerance: int = 2):
    """One EM+synapse scatter-dot overlay PNG per XY section under
    "synapse_overlays/" (+ legend.png); synapses are points, not regions.
    `synapses`: dicts with local ZYX voxel coords + compartment_tag. Each
    is drawn on every Z section within z_tolerance of its own z."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    overlay_dir = os.path.join(out_dir, "synapse_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    tags_present = sorted({s["compartment_tag"] for s in synapses}) if synapses else []
    if tags_present:
        fig, ax = plt.subplots(figsize=(0.1, 0.1))
        ax.axis("off")
        handles = [
            mpatches.Patch(color=SYNAPSE_TAG_COLORS.get(tag, SYNAPSE_TAG_DEFAULT_COLOR), label=tag)
            for tag in tags_present
        ]
        ax.legend(handles=handles, loc="center", fontsize=7, frameon=True, title=f"{len(synapses)} synapses")
        fig.savefig(os.path.join(overlay_dir, "legend.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    img_arr = tifffile.memmap(image_path)
    nz, ny, nx = img_arr.shape

    img_bytes_per_plane = ny * nx * img_arr.dtype.itemsize
    img_mmap_len = len(img_arr._mmap)
    page_size = mmap.PAGESIZE

    for z0 in range(0, nz, chunk_z):
        z1 = min(z0 + chunk_z, nz)
        img_chunk = np.array(img_arr[z0:z1])

        for i, z in enumerate(range(z0, z1)):
            nearby = [s for s in synapses if abs(s["z_vx"] - z) <= z_tolerance]
            img_slice = img_chunk[i]
            if downsample > 1:
                img_slice = img_slice[::downsample, ::downsample]

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_slice, cmap="gray")
            if nearby:
                xs = [s["x_vx"] / downsample for s in nearby]
                ys = [s["y_vx"] / downsample for s in nearby]
                colors = [SYNAPSE_TAG_COLORS.get(s["compartment_tag"], SYNAPSE_TAG_DEFAULT_COLOR) for s in nearby]
                ax.scatter(xs, ys, c=colors, s=12, edgecolors="white", linewidths=0.3)
            ax.axis("off")
            fig.savefig(os.path.join(overlay_dir, f"section_z{z:05d}.png"), dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        byte_start = z0 * img_bytes_per_plane
        byte_end = min(z1 * img_bytes_per_plane, img_mmap_len)
        aligned_start = (byte_start // page_size) * page_size
        if byte_end > aligned_start:
            img_arr._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)

    return nz
