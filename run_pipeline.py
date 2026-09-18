# Confidential - Do not distribute without permission.
# Copyright (c) QuIN Lab, University of Waterloo. All rights reserved.
# Author: Bowen Deng

"""Chains extract -> find-interactions -> extract-interaction-labels ->
reconstruct-local-meshes, stopping on first failure.
    python run_pipeline.py --data-dir /path/to/output --root-id 864691136057675736
"""
from __future__ import annotations

import argparse
import subprocess
import sys

STEPS = ["extract", "find-interactions", "extract-interaction-labels", "reconstruct-local-meshes"]


def build_step_cmd(step: str, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", "src.main"]
    if args.config:
        cmd += ["--config", args.config]
    cmd += [step, "--data-dir", args.data_dir]

    if args.root_id:
        for rid in args.root_id:
            cmd += ["--root-id", str(rid)]
    elif step == "extract":
        cmd += ["--all-clean"]
    else:
        cmd += ["--all-fetched"]

    if step in ("extract", "extract-interaction-labels") and args.mode:
        cmd += ["--mode", args.mode]
    if step == "reconstruct-local-meshes" and args.downsample is not None:
        cmd += ["--downsample", str(args.downsample)]
    return cmd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: src.main's own default, config.yaml)")
    parser.add_argument("--data-dir", required=True, help="Root output directory, passed through to every step")
    parser.add_argument("--root-id", type=int, action="append", default=None, help="Root ID to run the pipeline for (repeatable)")
    parser.add_argument("--all-clean", action="store_true", help="Run for every clean proofread astrocyte instead of specific --root-id values")
    parser.add_argument("--mode", choices=["full_bbox", "chunked"], default=None, help="Extraction/relabeling mode override, passed to extract and extract-interaction-labels")
    parser.add_argument("--downsample", type=int, default=None, help="Y/X downsample override for reconstruct-local-meshes")
    args = parser.parse_args(argv)

    if not args.root_id and not args.all_clean:
        parser.error("Provide --root-id (repeatable) or --all-clean.")

    completed = []
    for i, step in enumerate(STEPS, start=1):
        cmd = build_step_cmd(step, args)
        print(f"\n=== [{i}/{len(STEPS)}] {step} ===", flush=True)
        print("$ " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"\nPipeline stopped: '{step}' exited with code {result.returncode}. "
                f"Completed steps: {completed or '(none)'}. Fix the error above, then re-run this "
                f"same command -- every step here is safe to re-run (extract reuses existing "
                f"image.tif/mask.tif, the rest key off runs/<pipeline_date>/).",
                file=sys.stderr,
            )
            return result.returncode
        completed.append(step)

    print(f"\n=== Pipeline complete: {', '.join(completed)} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
