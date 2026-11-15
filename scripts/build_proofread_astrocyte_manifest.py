"""Query pinned v1300 proofread astrocytes and summarize their mesh extents."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.astrocyte_query import query_clean_astrocytes  # noqa: E402
from src.microns_client import MicronsClient, load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    client = MicronsClient(config)
    table = query_clean_astrocytes(
        client.cave, client.materialization_version,
        config["proofreading_table"], config["proofreading_status_value"],
    )
    cells = []
    for row in table.itertuples(index=False):
        root_id = int(row.pt_root_id)
        vertices = client.get_mesh_vertices_nm(root_id)
        minimum = np.min(vertices, axis=0)
        maximum = np.max(vertices, axis=0)
        cells.append({
            "root_id": str(root_id),
            "status": str(row.status),
            "valid_id": str(row.valid_id) if hasattr(row, "valid_id") else None,
            "mesh_vertices": int(len(vertices)),
            "mesh_bbox_nm": {"min": minimum.tolist(), "max": maximum.tolist()},
            "mesh_extent_um_xyz": ((maximum - minimum) / 1000).tolist(),
        })
        print(root_id, cells[-1]["mesh_extent_um_xyz"], flush=True)
    # Cell-level splits avoid leakage between crops of the same connected instance.
    root_ids = [cell["root_id"] for cell in cells]
    folds = []
    for index, test in enumerate(root_ids):
        validation = root_ids[(index + 1) % len(root_ids)]
        train = [root for root in root_ids if root not in {test, validation}]
        folds.append({"fold": index + 1, "train_root_ids": train,
                      "validation_root_ids": [validation], "test_root_ids": [test]})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": client.dataset,
        "materialization_version": client.materialization_version,
        "proofreading_table": config["proofreading_table"],
        "proofreading_status": config["proofreading_status_value"],
        "n_clean_cells": len(cells),
        "cells": cells,
        "suggested_leave_one_cell_out_folds": folds,
        "label_semantics": "Each positive mask is one proofread astrocyte instance, not an exhaustive any-astrocyte class mask.",
        "background_warning": "Other or unclassified astrocytes can occur outside the selected root mask and must not automatically be treated as confirmed negative class labels.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(cells)} cells to {args.out}")


if __name__ == "__main__":
    main()
