# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""Classifies cells near an astrocyte's bbox as astrocyte or neuron via three
layers: soma proximity, synapse tags (alone, NOT sufficient for "neuron"),
and bbox-segmentation classification (the only one catching axon-astrocyte
contact), each with a loose view and a strict (proofread) view. Real contact
is decided downstream by contact_filter.py; this module only classifies identity."""
from __future__ import annotations

import logging
import time

import pandas as pd

from .volume_extractor import retry_with_backoff

logger = logging.getLogger(__name__)

_QUERY_BATCH_SIZE = 5000
_INTER_BATCH_DELAY_S = 0.5  # throttling on top of retry_with_backoff: the CAVE backend has been observed to refuse connections under a rapid undelayed request burst

NUCLEUS_TABLE = "nucleus_detection_v0"
PRIMARY_CELLTYPE_TABLE = "aibs_metamodel_celltypes_v661"  # whole-dataset classifier; not 100% nucleus coverage
FALLBACK_NONNEURONAL_TABLE = "aibs_column_nonneuronal_ref"  # manual non-neuronal typing, "Minnie Column" region only

SYNAPSE_TABLE = "synapses_pni_2"
VORTEX_COMPARTMENT_TABLE = "vortex_compartment_targets"  # manual tag in {soma, spine, shaft, soma_spine, orphan, other}
SYNAPSE_STRUCTURE_TABLE = "synapse_target_structure"  # model prediction, value 0-6

# synapse_target_structure's numeric value -> the tag vocabulary vortex_compartment_targets uses natively.
# Values 4 and 5 (axonal bouton or other axonal structure) both coarsen to "axon_bouton".
_STRUCTURE_VALUE_TO_TAG = {
    0: "spine", 1: "spine", 2: "shaft", 3: "spine",
    4: "axon_bouton", 5: "axon_bouton", 6: "soma",
}

# Only these three postsynaptic tags count as a dendrite hit. "soma" (axo-somatic) and
# "axon_bouton" (axo-axonic) are structurally distinct compartments, not dendrite subtypes.
COMPARTMENT_TAG_CATEGORY = {
    "spine": "dendrite",
    "shaft": "dendrite",
    "soma_spine": "dendrite",
}

CELL_TYPE_CATEGORY = {"astrocyte": "astrocyte"}  # only astrocyte maps; every other soma-proximity cell_type is context only

NEURON_CLASSIFICATION_SYSTEMS = ("excitatory_neuron", "inhibitory_neuron")

NEURON_PROOFREADING_TABLE = "proofreading_status_and_strategy"  # only general-purpose neuron proofreading table in this materialization

ALL_CATEGORIES = ("astrocyte", "neuron")


def default_categories() -> dict:
    return {c: True for c in ALL_CATEGORIES}


def cell_category(cell_type: str | None, proofread_clean: bool | None, require_astrocyte_proofread: bool = False) -> str | None:
    """Category for a soma-proximity cell. require_astrocyte_proofread=True
    additionally requires proofread_clean is True (manually verified on the
    same table the target astrocyte list is drawn from)."""
    cat = CELL_TYPE_CATEGORY.get(cell_type)
    if cat == "astrocyte" and require_astrocyte_proofread and proofread_clean is not True:
        return None
    return cat


def synapse_post_category(compartment_tag: str | None) -> str | None:
    """Category for a synapse's postsynaptic side, from its compartment tag."""
    return COMPARTMENT_TAG_CATEGORY.get(compartment_tag)


def bbox_min_max_nm_from_metadata(metadata: dict) -> tuple[list[float], list[float]]:
    """mesh_bbox_nm is only populated on an astrocyte's first extraction run;
    a resumed run re-saves metadata.json with it None. voxel_bbox *
    em_resolution_nm reconstructs the same physical bbox either way."""
    mesh_bbox = metadata.get("mesh_bbox_nm") or {}
    if mesh_bbox.get("min") is not None and mesh_bbox.get("max") is not None:
        return mesh_bbox["min"], mesh_bbox["max"]

    voxel_bbox = metadata["voxel_bbox"]
    res = metadata["em_resolution_nm"]
    bbox_min_nm = [voxel_bbox["min"][i] * res[i] for i in range(3)]
    bbox_max_nm = [voxel_bbox["max"][i] * res[i] for i in range(3)]
    return bbox_min_nm, bbox_max_nm


def _nm_to_table_voxels(bbox_min_nm, bbox_max_nm, table_voxel_resolution_nm):
    return (
        [bbox_min_nm[i] / table_voxel_resolution_nm[i] for i in range(3)],
        [bbox_max_nm[i] / table_voxel_resolution_nm[i] for i in range(3)],
    )


def _query_table_in_batches(cave_client, table_name: str, materialization_version: int, column: str, values, batch_size: int = _QUERY_BATCH_SIZE) -> pd.DataFrame:
    """filter_in_dict query chunked into batches (a real astrocyte bbox can
    imply 100k+ ids; see interaction_query module docstring), each through
    retry_with_backoff with a small delay between batches."""
    values = list(values)
    if not values:
        return pd.DataFrame()

    n_batches = -(-len(values) // batch_size)
    frames = []
    for i in range(0, len(values), batch_size):
        batch = values[i : i + batch_size]
        batch_num = i // batch_size + 1
        if n_batches > 1:
            logger.info("querying %s: batch %d/%d (%d ids)", table_name, batch_num, n_batches, len(batch))
        df = retry_with_backoff(
            lambda: cave_client.materialize.query_table(
                table_name, materialization_version=materialization_version, filter_in_dict={column: batch},
            ),
            description=f"query_table({table_name}) batch {batch_num}/{n_batches}",
        )
        if not df.empty:
            frames.append(df)
        if _INTER_BATCH_DELAY_S:
            time.sleep(_INTER_BATCH_DELAY_S)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --- Layer 1: soma proximity ---

def nucleus_table_voxel_resolution_nm(cave_client, materialization_version: int) -> list[float]:
    meta = cave_client.materialize.get_table_metadata(NUCLEUS_TABLE, version=materialization_version)
    return meta["voxel_resolution"]


def query_real_cells_in_bbox(cave_client, materialization_version: int, bbox_min_nm, bbox_max_nm, table_voxel_resolution_nm=None) -> pd.DataFrame:
    """Nuclei whose centroid falls inside the bbox (server-side spatial query)."""
    if table_voxel_resolution_nm is None:
        table_voxel_resolution_nm = nucleus_table_voxel_resolution_nm(cave_client, materialization_version)
    bbox_vx_min, bbox_vx_max = _nm_to_table_voxels(bbox_min_nm, bbox_max_nm, table_voxel_resolution_nm)
    return retry_with_backoff(
        lambda: cave_client.materialize.query_table(
            NUCLEUS_TABLE,
            materialization_version=materialization_version,
            filter_spatial_dict={"pt_position": [bbox_vx_min, bbox_vx_max]},
            split_positions=True,
        ),
        description="query_table(nucleus_detection_v0) spatial query",
    )


def annotate_cell_types(cave_client, materialization_version: int, nucleus_ids) -> pd.DataFrame:
    """Cell type per nucleus id: PRIMARY_CELLTYPE_TABLE first, FALLBACK_NONNEURONAL_TABLE
    for whatever it didn't cover. Indexed by nucleus id; a missing index means type unknown."""
    nucleus_ids = list(int(i) for i in nucleus_ids)
    if not nucleus_ids:
        return pd.DataFrame(columns=["classification_system", "cell_type", "source"]).set_index(pd.Index([], name="target_id"))

    primary = _query_table_in_batches(cave_client, PRIMARY_CELLTYPE_TABLE, materialization_version, "target_id", nucleus_ids)
    if primary.empty:
        primary = pd.DataFrame(columns=["target_id", "classification_system", "cell_type", "source"])
    else:
        primary = primary[["target_id", "classification_system", "cell_type"]].copy()
        primary["source"] = PRIMARY_CELLTYPE_TABLE
        # a target_id can return >1 row (versioning duplicate, identical fields); keep first
        primary = primary.drop_duplicates(subset="target_id", keep="first")

    remaining = [i for i in nucleus_ids if i not in set(primary["target_id"])]
    if remaining:
        fallback = _query_table_in_batches(cave_client, FALLBACK_NONNEURONAL_TABLE, materialization_version, "target_id", remaining)
        if not fallback.empty:
            fallback = fallback[["target_id", "cell_type"]].copy()
            fallback["classification_system"] = "nonneuron"
            fallback["source"] = FALLBACK_NONNEURONAL_TABLE
            fallback = fallback.drop_duplicates(subset="target_id", keep="first")
            primary = pd.concat([primary, fallback], ignore_index=True)

    return primary.set_index("target_id")


def find_cells_in_bbox(cave_client, materialization_version: int, root_id: int, bbox_min_nm, bbox_max_nm, clean_astrocyte_root_ids: set | None = None) -> list:
    """Cells whose soma sits in the bbox, typed via annotate_cell_types.
    clean_astrocyte_root_ids tags each astrocyte-typed hit's proofread_clean;
    None skips it (stays None)."""
    table_res = nucleus_table_voxel_resolution_nm(cave_client, materialization_version)
    nuclei_df = query_real_cells_in_bbox(cave_client, materialization_version, bbox_min_nm, bbox_max_nm, table_voxel_resolution_nm=table_res)
    types_df = annotate_cell_types(cave_client, materialization_version, nuclei_df["id"].tolist())

    cells_in_bbox = []
    for _, row in nuclei_df.iterrows():
        nucleus_id = int(row["id"])
        partner_root_id = int(row["pt_root_id"])
        type_row = types_df.loc[nucleus_id] if nucleus_id in types_df.index else None
        cell_type = type_row["cell_type"] if type_row is not None else None
        proofread_clean = None
        if cell_type == "astrocyte" and clean_astrocyte_root_ids is not None:
            proofread_clean = partner_root_id in clean_astrocyte_root_ids
        cells_in_bbox.append({
            "nucleus_id": nucleus_id,
            "root_id": partner_root_id,
            "is_target_astrocyte": partner_root_id == int(root_id),
            "volume_um3": float(row["volume"]) if "volume" in row else None,
            "position_nm": [
                float(row["pt_position_x"]) * table_res[0],
                float(row["pt_position_y"]) * table_res[1],
                float(row["pt_position_z"]) * table_res[2],
            ] if "pt_position_x" in row else None,
            "classification_system": type_row["classification_system"] if type_row is not None else None,
            "cell_type": cell_type,
            "cell_type_source": type_row["source"] if type_row is not None else None,
            "category": cell_category(cell_type, proofread_clean),
            "proofread_clean": proofread_clean,
        })
    return cells_in_bbox


# --- Layer 2: synapse compartment ---

def synapse_table_voxel_resolution_nm(cave_client, materialization_version: int) -> list[float]:
    meta = cave_client.materialize.get_table_metadata(SYNAPSE_TABLE, version=materialization_version)
    return meta["voxel_resolution"]


def query_synapses_in_bbox(cave_client, materialization_version: int, bbox_min_nm, bbox_max_nm, table_voxel_resolution_nm=None) -> pd.DataFrame:
    """Synapses whose center point falls inside the bbox (server-side spatial query)."""
    if table_voxel_resolution_nm is None:
        table_voxel_resolution_nm = synapse_table_voxel_resolution_nm(cave_client, materialization_version)
    bbox_vx_min, bbox_vx_max = _nm_to_table_voxels(bbox_min_nm, bbox_max_nm, table_voxel_resolution_nm)
    return retry_with_backoff(
        lambda: cave_client.materialize.query_table(
            SYNAPSE_TABLE,
            materialization_version=materialization_version,
            filter_spatial_dict={"ctr_pt_position": [bbox_vx_min, bbox_vx_max]},
            split_positions=True,
        ),
        description="query_table(synapses_pni_2) spatial query",
    )


def annotate_synapse_compartments(cave_client, materialization_version: int, synapse_ids) -> pd.DataFrame:
    """Compartment tag per synapse id: VORTEX_COMPARTMENT_TABLE (manual) first,
    SYNAPSE_STRUCTURE_TABLE (model, mapped via _STRUCTURE_VALUE_TO_TAG) as fallback."""
    synapse_ids = list(int(i) for i in synapse_ids)
    if not synapse_ids:
        return pd.DataFrame(columns=["compartment_tag", "source"]).set_index(pd.Index([], name="target_id"))

    primary = _query_table_in_batches(cave_client, VORTEX_COMPARTMENT_TABLE, materialization_version, "target_id", synapse_ids)
    if primary.empty:
        primary = pd.DataFrame(columns=["target_id", "compartment_tag", "source"])
    else:
        primary = primary[["target_id", "tag"]].rename(columns={"tag": "compartment_tag"}).copy()
        primary["source"] = VORTEX_COMPARTMENT_TABLE

    remaining = [i for i in synapse_ids if i not in set(primary["target_id"])]
    if remaining:
        structure = _query_table_in_batches(cave_client, SYNAPSE_STRUCTURE_TABLE, materialization_version, "target_id", remaining)
        if not structure.empty:
            structure = structure[["target_id", "value"]].copy()
            structure["compartment_tag"] = structure["value"].map(_STRUCTURE_VALUE_TO_TAG)
            structure = structure.dropna(subset=["compartment_tag"])
            structure["source"] = SYNAPSE_STRUCTURE_TABLE
            structure = structure.drop(columns=["value"])
            primary = pd.concat([primary, structure], ignore_index=True)

    return primary.set_index("target_id")


def find_synapses_in_bbox(cave_client, materialization_version: int, root_id: int, bbox_min_nm, bbox_max_nm) -> list:
    """Synapses in the bbox with compartment tag and postsynaptic category.
    pre_root_id is recorded but never categorized (always an axon terminal)."""
    table_res = synapse_table_voxel_resolution_nm(cave_client, materialization_version)
    syn_df = query_synapses_in_bbox(cave_client, materialization_version, bbox_min_nm, bbox_max_nm, table_voxel_resolution_nm=table_res)
    if syn_df.empty:
        return []

    comp_df = annotate_synapse_compartments(cave_client, materialization_version, syn_df["id"].tolist())

    synapses = []
    for _, row in syn_df.iterrows():
        synapse_id = int(row["id"])
        comp_row = comp_df.loc[synapse_id] if synapse_id in comp_df.index else None
        tag = comp_row["compartment_tag"] if comp_row is not None else None
        source = comp_row["source"] if comp_row is not None else None
        pre_root_id = int(row["pre_pt_root_id"])
        post_root_id = int(row["post_pt_root_id"])

        synapses.append({
            "synapse_id": synapse_id,
            "pre_root_id": pre_root_id,
            "pre_category": None,
            "post_root_id": post_root_id,
            "post_category": synapse_post_category(tag),
            "compartment_tag": tag,
            "compartment_source": source,
            "position_nm": [
                float(row["ctr_pt_position_x"]) * table_res[0],
                float(row["ctr_pt_position_y"]) * table_res[1],
                float(row["ctr_pt_position_z"]) * table_res[2],
            ],
        })
    return synapses


# --- Layer 3: bbox segmentation enumeration ---
# Layers 1-2 only see a neuron's tissue if its own soma sits in the bbox or a synapse
# lands on it; an axon or an untagged dendrite fragment is invisible to both. This
# layer classifies every root_id volume_extractor.enumerate_root_ids_in_bbox found
# with voxels in the bbox, regardless of soma or synapse presence.


def classify_root_ids_by_celltype(cave_client, materialization_version: int, root_ids) -> pd.DataFrame:
    """Cell-type lookup for arbitrary root_ids via root_id -> NUCLEUS_TABLE ->
    annotate_cell_types. Absent from the index = no detected nucleus at all
    (genuinely unknown, not "unclassified"; see root_id_category). Nuclei
    disagreeing on type (a real soma-soma merge) get reported ambiguous
    rather than an arbitrary pick."""
    root_ids = sorted(set(int(r) for r in root_ids))
    empty_cols = ["nucleus_id", "classification_system", "cell_type", "cell_type_source", "ambiguous_multi_nucleus"]
    if not root_ids:
        return pd.DataFrame(columns=empty_cols).set_index(pd.Index([], name="root_id"))

    nuclei = _query_table_in_batches(cave_client, NUCLEUS_TABLE, materialization_version, "pt_root_id", root_ids)
    if nuclei.empty:
        return pd.DataFrame(columns=empty_cols).set_index(pd.Index([], name="root_id"))

    types_df = annotate_cell_types(cave_client, materialization_version, nuclei["id"].tolist())

    records = []
    for root_id, group in nuclei.groupby("pt_root_id"):
        nucleus_ids = [int(x) for x in group["id"]]
        type_rows = [types_df.loc[nid] if nid in types_df.index else None for nid in nucleus_ids]
        distinct_types = {
            (tr["classification_system"], tr["cell_type"]) if tr is not None else (None, None)
            for tr in type_rows
        }
        ambiguous = len(nucleus_ids) > 1 and len(distinct_types) > 1
        nucleus_id = nucleus_ids[0]
        type_row = None if ambiguous else type_rows[0]
        records.append({
            "root_id": int(root_id),
            "nucleus_id": nucleus_id,
            "classification_system": type_row["classification_system"] if type_row is not None else None,
            "cell_type": type_row["cell_type"] if type_row is not None else None,
            "cell_type_source": type_row["source"] if type_row is not None else None,
            "ambiguous_multi_nucleus": ambiguous,
        })
    return pd.DataFrame.from_records(records, columns=["root_id"] + empty_cols).set_index("root_id")


def root_id_category(classification_system: str | None, cell_type: str | None) -> str | None:
    """Category for a root_id classified by classify_root_ids_by_celltype.
    Neuron keys off classification_system (stable and coarse vs. the open-ended cell_type
    subtype list); astrocyte keys off cell_type via CELL_TYPE_CATEGORY. Every other
    nonneuron cell_type (oligo, microglia, pericyte, OPC) stays uncategorized."""
    if classification_system in NEURON_CLASSIFICATION_SYSTEMS:
        return "neuron"
    return CELL_TYPE_CATEGORY.get(cell_type)


def find_bbox_segmentation_cells(cave_client, materialization_version: int, root_id: int, candidate_root_ids) -> list:
    """classify_root_ids_by_celltype reshaped into find_cells_in_bbox's record
    shape. candidate_root_ids is volume_extractor.enumerate_root_ids_in_bbox's
    output (every root_id with >=min_voxels in the bbox segmentation)."""
    root_id = int(root_id)
    types_df = classify_root_ids_by_celltype(cave_client, materialization_version, candidate_root_ids)

    cells = []
    for rid in sorted(int(r) for r in set(candidate_root_ids)):
        type_row = types_df.loc[rid] if rid in types_df.index else None
        classification_system = type_row["classification_system"] if type_row is not None else None
        cell_type = type_row["cell_type"] if type_row is not None else None
        cells.append({
            "root_id": rid,
            "is_target_astrocyte": rid == root_id,
            "nucleus_id": int(type_row["nucleus_id"]) if type_row is not None and pd.notna(type_row["nucleus_id"]) else None,
            "classification_system": classification_system,
            "cell_type": cell_type,
            "cell_type_source": type_row["cell_type_source"] if type_row is not None else None,
            "category": root_id_category(classification_system, cell_type),
            "proofread_clean": None,  # this layer never checks the astrocyte proofreading table; present for record-shape parity with cells_in_bbox
        })
    return cells


def query_neuron_proofreading_status(cave_client, materialization_version: int, root_ids) -> pd.DataFrame:
    """Neuron proofreading status from NEURON_PROOFREADING_TABLE, indexed by root_id
    with columns status_axon, status_dendrite, strategy_axon, strategy_dendrite. A
    root_id absent from the index was never proofread as a neuron (genuinely unknown,
    not False)."""
    root_ids = sorted(set(int(r) for r in root_ids))
    empty_cols = ["status_axon", "status_dendrite", "strategy_axon", "strategy_dendrite"]
    if not root_ids:
        return pd.DataFrame(columns=empty_cols).set_index(pd.Index([], name="root_id"))

    df = _query_table_in_batches(cave_client, NEURON_PROOFREADING_TABLE, materialization_version, "pt_root_id", root_ids)
    if df.empty:
        return pd.DataFrame(columns=empty_cols).set_index(pd.Index([], name="root_id"))

    df = df.drop_duplicates(subset="pt_root_id", keep="first")
    df = df[["pt_root_id"] + empty_cols].copy()
    df["pt_root_id"] = df["pt_root_id"].astype(int)
    return df.rename(columns={"pt_root_id": "root_id"}).set_index("root_id")


def neuron_proofread_root_ids(proofreading_df: pd.DataFrame, require_axon: bool = True) -> set:
    """Strict-view root_id set: status_dendrite True always required, status_axon True
    also required by default (require_axon=True) since this layer doesn't know which
    compartment actually touches the target astrocyte."""
    if proofreading_df.empty:
        return set()
    mask = proofreading_df["status_dendrite"] == True  # noqa: E712
    if require_axon:
        mask = mask & (proofreading_df["status_axon"] == True)  # noqa: E712
    return set(int(r) for r in proofreading_df.index[mask])


# --- Combined result ---

def compute_interacting_root_ids(
    root_id: int, cells_in_bbox: list, synapses: list, categories: dict | None = None,
    require_astrocyte_proofread: bool = False, bbox_segmentation_cells: list | None = None,
    require_neuron_proofread: bool = False, neuron_proofread_root_ids: set | None = None,
) -> list:
    """Deduped, category-filtered interacting root_ids from already-computed
    layers; category recomputed from raw fields (not
    cached) to stay current. require_*_proofread narrows to the strict view
    (call twice for both). A dendrite tag alone does NOT grant "neuron":
    only bbox_segmentation_cells' classification does (a tag-only rule let an oligodendrocyte through)."""
    if categories is None:
        categories = default_categories()
    root_id = int(root_id)
    neuron_proofread_root_ids = neuron_proofread_root_ids or set()

    interacting_root_ids = set()
    for cell in cells_in_bbox:
        if cell["is_target_astrocyte"]:
            continue
        cat = cell_category(cell["cell_type"], cell.get("proofread_clean"), require_astrocyte_proofread=require_astrocyte_proofread)
        if cat is not None and categories.get(cat, True):
            interacting_root_ids.add(cell["root_id"])

    for cell in bbox_segmentation_cells or []:
        if cell["is_target_astrocyte"]:
            continue
        cat = cell["category"]
        if cat == "astrocyte" and require_astrocyte_proofread:
            continue
        if cat == "neuron" and require_neuron_proofread and cell["root_id"] not in neuron_proofread_root_ids:
            continue
        if cat is not None and categories.get(cat, True):
            interacting_root_ids.add(cell["root_id"])

    return sorted(interacting_root_ids)


def find_interactions(cave_client, materialization_version: int, root_id: int, bbox_min_nm, bbox_max_nm, categories: dict | None = None, clean_astrocyte_root_ids: set | None = None) -> dict:
    """Layers 1-2 (soma proximity + synapse compartment) only, a preview
    main.py's cmd_find_interactions overwrites after adding layer 3 (bbox
    segmentation) and neuron-proofreading via a second
    compute_interacting_root_ids call. interacting_root_ids_proofread is
    always a subset of interacting_root_ids."""
    if categories is None:
        categories = default_categories()

    cells_in_bbox = find_cells_in_bbox(cave_client, materialization_version, root_id, bbox_min_nm, bbox_max_nm, clean_astrocyte_root_ids=clean_astrocyte_root_ids)
    synapses = find_synapses_in_bbox(cave_client, materialization_version, root_id, bbox_min_nm, bbox_max_nm) if categories.get("neuron", True) else []

    interacting_root_ids = compute_interacting_root_ids(root_id, cells_in_bbox, synapses, categories, require_astrocyte_proofread=False, require_neuron_proofread=False)
    interacting_root_ids_proofread = compute_interacting_root_ids(root_id, cells_in_bbox, synapses, categories, require_astrocyte_proofread=True, require_neuron_proofread=True)

    compartment_tag_counts: dict[str, int] = {}
    for syn in synapses:
        tag = syn["compartment_tag"] or "unclassified"
        compartment_tag_counts[tag] = compartment_tag_counts.get(tag, 0) + 1

    return {
        "root_id": int(root_id),
        "bbox_min_nm": bbox_min_nm,
        "bbox_max_nm": bbox_max_nm,
        "categories": categories,
        "n_cells_in_bbox": len(cells_in_bbox),
        "cells_in_bbox": cells_in_bbox,
        "n_synapses_in_bbox": len(synapses),
        "synapses_in_bbox": synapses,
        "compartment_tag_counts": compartment_tag_counts,
        "n_interacting_root_ids": len(interacting_root_ids),
        "interacting_root_ids": sorted(interacting_root_ids),
        "n_interacting_root_ids_proofread": len(interacting_root_ids_proofread),
        "interacting_root_ids_proofread": sorted(interacting_root_ids_proofread),
    }
