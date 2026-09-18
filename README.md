# MICrONS Astrocyte Toolkit (v1300)

Extracts 3D EM subvolumes and binary astrocyte masks, and finds real cells
and synapse-compartment interactions near them, for proofread astrocytes in the
[MICrONS](https://www.microns-explorer.org/) `minnie65_public` dataset.

### Proofread astrocytes (v1300)

12 available (`python -m src.main list`); `864691136057675736` is the one
verified end-to-end so far.

```
864691136057675736  <- verified
864691135234576473
864691135772920395
864691135970105957
864691135384083674
864691135852189639
864691135955003656
864691135527301979
864691135995876778
864691135164125997
864691134886294010
864691136298097435
```

Official tutorial for these same 12 astrocytes (query + mesh render via PyVista): [Astrocyte Meshes](https://tutorial.microns-explorer.org/examples/astrocyte_meshes.html)

### Contact threshold

`contact_filter.threshold_nm` (config.yaml) defaults to 40nm, the "direct
contact" distance from Connectomic analysis of astrocyte-synapse interactions
in the cerebral cortex (bioRxiv, 2025, `10.1101/2025.02.20.639274`).
`scripts/threshold_sensitivity.py` re-sweeps this against the full-shape
volume if needed.

`contact_filter` modifies `interaction_labels.tif` in place. Never re-run it
with a *larger* threshold on an already-filtered file — filter from an
unfiltered, full-shape volume instead (`scripts/build_full_mask_labels.py`).

## Environment

Two dedicated conda envs, not shared with other projects:

```bash
# main env -- everything except mesh rendering
conda create -n microns-astrocyte-toolkit python=3.10
conda env config vars set PYTHONNOUSERSITE=1 -n microns-astrocyte-toolkit
conda run -n microns-astrocyte-toolkit python -m pip install -r requirements.txt

# mesh rendering -- pyvista/vtk conflict with the main env's caveclient stack
conda create -n microns-astrocyte-toolkit-pyvista python=3.10
conda env config vars set PYTHONNOUSERSITE=1 -n microns-astrocyte-toolkit-pyvista
conda run -n microns-astrocyte-toolkit-pyvista python -m pip install pyvista numpy tifffile scikit-image
```

CAVE auth token, one-time per machine:
```python
from caveclient import CAVEclient
client = CAVEclient()
client.auth.get_new_token()   # opens a URL, log in, copy the token
client.auth.save_token(token="PASTE_TOKEN_HERE")
```
Verify: `python -m src.main list` should print the astrocyte table with no auth error.

## File structure

```
config.yaml               config
run_pipeline.py           runs all four CLI steps in order

src/
  microns_client.py       CAVEclient + CloudVolume wrapper
  volume_extractor.py     mesh -> bbox -> EM+seg download -> mask
  coordinate_utils.py     coordinate conversions
  astrocyte_query.py      proofread astrocyte root_ids
  interaction_query.py    candidate neighbor cells + synapses
  contact_filter.py       distance-based contact filter
  io_utils.py             volume/overlay/label/QC I/O
  mesh_export.py          per-cell .ply export
  main.py                 CLI entry point

scripts/
  build_strict_overlay.py            proofread-subset label map
  build_full_mask_labels.py          full per-cell label volumes
  build_full_mask_overlays.py        2D overlays, full-mask
  build_full_mask_meshes.py          3D meshes, full-mask
  render_contact_meshes.py           3D render, contact meshes
  render_full_mask_meshes.py         3D render, full-mask
  render_full_masks_within_200nm.py  3D render, full-mask + contact filter
  threshold_sensitivity.py           40-80nm threshold sweep
```

## Usage

Every command except `list` requires `--data-dir <path>`. Data lands under
`<data-dir>/astro_<root_id>/`; `extract` reuses an existing `image.tif`/`mask.tif`
there instead of re-downloading.

**One command, all four steps** (stops on first failure, safe to re-run):
```bash
python run_pipeline.py --data-dir /path/to/output --root-id 864691136057675736
```

**Or step by step:**
```bash
python -m src.main list
python -m src.main extract --data-dir /path/to/output --root-id 864691136057675736
python -m src.main find-interactions --data-dir /path/to/output --root-id 864691136057675736
python -m src.main extract-interaction-labels --data-dir /path/to/output --root-id 864691136057675736
python -m src.main reconstruct-local-meshes --data-dir /path/to/output --root-id 864691136057675736
```

**Auxiliary scripts** (`--data-dir <astro_dir>` or `--run-dir <runs/date>`), in order:
```bash
python scripts/build_strict_overlay.py --data-dir /path/to/output/astro_864691136057675736
python scripts/build_full_mask_labels.py --data-dir /path/to/output/astro_864691136057675736
python scripts/build_full_mask_overlays.py --data-dir /path/to/output/astro_864691136057675736
python scripts/threshold_sensitivity.py --data-dir /path/to/output/astro_864691136057675736
```

