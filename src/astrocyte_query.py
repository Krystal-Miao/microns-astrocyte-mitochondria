# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""Query proofread astrocyte root_ids from the CAVE materialization table,
via the generic query_table API (not the tutorial's tables.<name>(...)
shortcut) with the table name verified against get_tables() first, so a
future rename fails loudly instead of silently returning nothing."""
from __future__ import annotations

import pandas as pd


def _find_table_candidates(available_tables, configured_name: str):
    if configured_name in available_tables:
        return [configured_name]
    keywords = ("astro", "glia", "proofread")
    return [t for t in available_tables if any(k in t.lower() for k in keywords)]


def query_clean_astrocytes(cave_client, materialization_version: int, table_name: str, status_value: str = "clean") -> pd.DataFrame:
    """Rows from the astrocyte proofreading-status table matching
    status_value (default 'clean'); has a 'pt_root_id' column."""
    available_tables = cave_client.materialize.get_tables(version=materialization_version)

    if table_name not in available_tables:
        candidates = _find_table_candidates(available_tables, table_name)
        raise ValueError(
            f"Configured proofreading_table '{table_name}' was not found in "
            f"materialization version {materialization_version}. "
            f"Tables that look related: {candidates if candidates else '(none found)'}. "
            f"Full table list: {available_tables}. "
            "Update config.yaml's `proofreading_table` to the current table name."
        )

    df = cave_client.materialize.query_table(
        table_name,
        materialization_version=materialization_version,
        split_positions=True,
    )

    if "status" not in df.columns:
        raise ValueError(
            f"Table '{table_name}' does not have a 'status' column (columns={list(df.columns)}); "
            "the schema for this table may have changed -- inspect it and update this function."
        )
    if "pt_root_id" not in df.columns:
        raise ValueError(
            f"Table '{table_name}' does not have a 'pt_root_id' column (columns={list(df.columns)}); "
            "the schema for this table may have changed -- inspect it and update this function."
        )

    clean_df = df[df["status"] == status_value].reset_index(drop=True)
    return clean_df


def resolve_root_ids(manual_root_ids, use_query: bool, clean_df: pd.DataFrame | None) -> list:
    """Combine manually-supplied root_ids with query results (if requested),
    de-duplicated, preserving manual ids first.
    """
    root_ids = [int(r) for r in manual_root_ids]
    if use_query:
        if clean_df is None:
            raise ValueError("use_query=True but no query result DataFrame was provided")
        for rid in clean_df["pt_root_id"].tolist():
            rid = int(rid)
            if rid not in root_ids:
                root_ids.append(rid)
    seen = set()
    deduped = []
    for rid in root_ids:
        if rid not in seen:
            seen.add(rid)
            deduped.append(rid)
    return deduped
