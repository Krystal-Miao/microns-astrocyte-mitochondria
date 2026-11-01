# Raw 2D U-Net baseline

This is a small, randomly initialized CNN U-Net for a **single-cell pilot**. It predicts the provided target-cell mask from raw EM, not an exhaustive annotation of all astrocytes. Other astrocytes may be marked background in these masks, affecting both training and scores. Anatomical label completeness remains unverified.

## Environment

Use the project-local, isolated `.venv-unet` (Python 3.10). No shared site packages are required.

```bash
source .venv-unet/bin/activate
python -m pip install -r requirements-unet.txt
python -m unittest discover -s validation -p 'test_astro_unet.py'
```

## Data and split

The three existing sample volumes each contain 128 × 512 × 512 voxels (ZYX) at 8 × 8 × 40 nm (XYZ). The loader checks dimensions, binary masks, metadata axes/spacing, and spatial overlap. It records source SHA-256 hashes. Constant image planes are excluded automatically: currently sample_01 Z=9,124 and sample_02 Z=108. Their causes are not established; original arrays are unchanged.

Each outer fold holds out one complete sample for testing. In the other two samples, Z=0–95 is training, Z=96–111 is an unused 16-section guard band, and Z=112–127 is validation. Constant slices are removed from each partition. The guard band reduces immediate adjacency but does not establish statistical independence within this cell. The final training step selects the test checkpoint (`last.pt`), at threshold 0.5. A separate `best.pt` records the highest validation Dice; ties retain the earliest checkpoint. It is not used for final testing.

## Model and training

Three encoder levels (16,32,64 channels), a 128-channel bottleneck, and three decoder levels with concatenated skip connections. Each block has two padded 3×3 convolutions and ReLU; downsampling is 2×2 max pooling, upsampling is bilinear. A 1×1 head outputs a single logit per pixel. This is a small modern U-Net variant, not an exact reproduction of the original paper or AstroSeg-vEM's MiT-B2 architecture.

Raw input is converted from uint8 to float32 and divided by 255. No denoising, CLAHE, percentile clipping or resampling is applied. Uniformly sampled 256×256 crops retain native resolution. Image and label receive identical 90-degree rotations and reflections. Batch size is 2, Adam learning rate 0.001, and loss is equally weighted BCEWithLogits plus per-image soft Dice loss (smoothing constant 1). Background-only crops are allowed. Validation/test inference uses full 512×512 slices without augmentation or postprocessing.

The default 400 optimizer steps are an initial engineering budget, **not evidence of convergence**. Seeds are recorded, but bitwise equality across devices is not guaranteed. Do not tune this budget based on test scores. Repeated-seed comparisons remain future work.

## Commands

Every run requires a new output directory, protecting previous results. Outputs under `samples/` remain local and ignored by Git.

```bash
python scripts/train_astro_unet.py --audit-only --out samples/unet_audit_01
python scripts/train_astro_unet.py --smoke --steps 400 --out samples/unet_smoke_new
python scripts/train_astro_unet.py --fold 3 --steps 400 --out samples/unet_raw_fold3_seed42
python scripts/train_astro_unet.py --fold all --steps 400 --seed 42 --out samples/unet_raw_all_seed42
```

The smoke check repeatedly trains on one fixed sample_01 patch and checks finite, declining loss. It is a development diagnostic, not a segmentation benchmark or proof of successful generalization. It must not be used to tune methods against fold 1's held-out sample. Run all three folds only after fixing the protocol.

## Outputs and interpretation

- `audit.json`, `config.json`: source checks, exclusions, software versions, settings and limitations.
- `fold_N/split.json`: exact sample/Z membership of each partition.
- `last.pt`, `best.pt`, `history.json`: final weights used for testing, validation-best weights for reference, and training/validation history.
- `training_monitor.json`: fixed 8 evenly spaced valid slices per training volume used for training diagnostics; these are not the entire training set.
- `result.json`, `test_slice_metrics.csv`: test-volume and per-slice counts and Dice/IoU.
- `*_prediction_valid_slices.tif` and `*_prediction_z.json`: binary predictions and explicit mapping back to original Z indices. Excluded planes are omitted, so output Z indices must not be treated as source Z indices.

Primary scores aggregate TP/FP/FN over all valid voxels of a held-out volume, rather than averaging slice scores. Empty prediction plus empty reference is assigned 1; nonempty reference plus empty prediction is 0. Report folds separately, along with seed variability when available. The three volumes are not independent biological replicates.

## References

- [U-Net original paper](https://arxiv.org/abs/1505.04597): encoder/decoder and skip-connection architecture.
- [AstroSeg-vEM](https://github.com/QuIN-Lab/AstroSeg-vEM): related astrocyte workflow; this code does not copy its model or parameters.

## Initial local validation (2026-10-29)

Dependencies were installed into `.venv-unet`; `requirements-unet-lock.txt` records the resolved versions for this Python/macOS environment. Recreate the same package versions with `python -m pip install -r requirements-unet-lock.txt` (platform compatibility still applies).

Five automated checks pass: known-overlap metric values, output dimensions and finite gradients, volume separation and guard-band rules, paired image/mask augmentation, and prediction TIFF/source-Z mapping. The data audit found no overlapping sample bounding boxes. Valid train/validation/test slice counts are 192/32/126 for fold 1, 191/31/127 for fold 2, and 191/31/128 for fold 3.

The first 100-step fixed-patch diagnostic decreased loss from 1.3004 to 1.1978, but did not meet the predeclared 20% reduction check. This is retained in `samples/unet_smoke_01/smoke.json`; it is not a passing run. A longer diagnostic is checked before any complete cross-validation run. No test-volume performance is claimed by these implementation checks.

The 400-step diagnostic completed on CPU and passed the unchanged loss-reduction criterion: 1.3004 → 0.9284 (28.6% reduction). Loss fluctuated, so this demonstrates finite optimization with an overall decrease, not convergence or a successful full overfit. Results are in `samples/unet_smoke_02/smoke.json`. Full-volume training and held-out Dice/IoU have not yet been run. Source hashes were rechecked and unchanged.


## Final-step evaluation update

New runs evaluate `last.pt` at exactly `--steps` (400 by default). Existing runs are unchanged; their final-step weights were not saved, so rerunning is necessary. Every 100 steps and at the final step, logs/history include Dice, mean per-slice loss, predicted foreground fraction, and labeled foreground fraction for both the fixed training monitor and complete validation split. These evaluations run without gradients or augmentation and do not change sampling RNG state. The usual printed `loss` remains current minibatch training loss; the monitor loss is a separate full-slice diagnostic. No test scores are used during training.

```bash
python -u scripts/train_astro_unet.py --fold all --steps 400 --seed 42 --out samples/unet_raw_last_seed42_run1 2>&1 | tee samples/unet_raw_last_seed42_run1.log
```

Regression validation includes a synthetic run where step 100 is validation-best but step 101 is final: final testing must load step 101 and report its validation score. This does not claim biological performance.

## Fixed-patch diagnostic

Run `python -u scripts/diagnose_astro_unet.py --steps 1000 --out samples/unet_patch_diagnostic_new`.
This reuses the earlier smoke patch (sample_01, Z=47, Y=256:512, X=128:384),
which belongs to training in folds 2/3. No validation/test scoring is performed.
Because development uses sample_01, subsequent fold 1 results must not be described
as completely untouched by development. This is a memorization diagnostic, not a benchmark.
The baseline architecture, Adam LR=0.001, raw /255 scaling and BCE+Dice loss are
unchanged; a single fixed patch is used without augmentation for 1000 updates.
The final Dice target is 0.95 at threshold 0.5; failing it is reported rather than
silently extending the run or tuning the threshold.

`history.csv` and `history.json` record post-update measurements every 50 steps,
including step 0, Dice/IoU, loss, predicted/label foreground fractions, probability
range and mean probability on each reference class. Gradient L2 norm and maximum
output-head weight update help detect broken optimization. Step 0 has no update.
`config.json` records coordinates, source hashes, software and code hashes.
`result.json`, `last.pt`, input/mask/prediction/probability TIFFs and RGB overlay
PPMs are saved. The source TIFFs remain unchanged. Overlays aid visual inspection;
matching dimensions and metadata alone do not prove anatomical label correctness.

### Fixed-patch result (2026-10-29)

`samples/unet_patch_diagnostic_01` completed 1000 CPU steps in 170 seconds.
Initial loss 1.3004 / Dice 0; final loss 0.9124 / Dice 0.6371 / IoU 0.4674.
Maximum logged Dice was 0.7915 at step 900, followed by a drop to 0.4661 at
950 and partial recovery at 1000. Final predicted foreground was 48.19% versus
30.63% labeled. Foreground/background mean probabilities were 0.5772/0.2508.
Nonzero finite gradients and head updates, together with improved Dice, show
learning is occurring; they do not establish correctness of every implementation detail.
The 0.95 final Dice target was **not met**. Visual inspection of the final overlay
shows a large approximately rectangular false-positive area instead of accurate
label boundaries. Matching spatial metadata was checked, but anatomical label
accuracy is not independently established. Source hashes were unchanged.
This does not diagnose the exact cause of the multi-volume all-background result.
A subsequent controlled learning-rate comparison on this same development patch
would test optimization stability; no learning-rate change was made in this run.
Eight automated checks pass, including the distinction between below-threshold
probability separation and zero hard Dice. PNG copies of the two PPM overlays
were generated locally for viewing.

## Learning-rate control experiment

The diagnostic now accepts `--learning-rate` (default 0.001) and saves probability
TIFFs plus PNG overlays in `snapshots/` every 50 steps, including the initial model.
PNG reference and final overlays are also saved at the run root. Saving snapshots
adds no random draws and does not alter optimization. Compare CPU runs with seed
42, width 16, the identical fixed patch, loss, normalization, and 1000 steps:

```bash
python -u scripts/diagnose_astro_unet.py --device cpu --learning-rate 0.001 --out samples/unet_patch_lr001_01
python -u scripts/diagnose_astro_unet.py --device cpu --learning-rate 0.0003 --out samples/unet_patch_lr0003_01
```

Keep the baseline training script unchanged until this development experiment is
interpreted. This is one seed and one memorized patch, not validation/test evidence.

Both controlled runs completed: LR 0.001 final Dice/IoU = 0.6371/0.4674,
loss = 0.9124; LR 0.0003 final Dice/IoU = 0.9968/0.9935, loss = 0.0127.
The latter predicted foreground fraction is 30.66% versus 30.63% reference.
The original-LR rerun exactly reproduces the previous logged Dice and loss at
all 21 observation steps. Shared configuration/source/code hashes were checked.
The smaller-LR final overlay closely follows the reference and no longer has the
large rectangular false-positive area. This supports sensitivity to optimization
settings on this patch, not a proven padding bug or generalization improvement.
Intermediate side-by-side overlays and metrics are available in
`samples/unet_patch_lr_comparison_01/index.html`; raw results in each run folder.
No full-volume training, loss, sampling or preprocessing configuration was changed.

## Full 512 × 512 training-slice diagnostic

`--full-slice` uses all of sample_01 Z=47 instead of the earlier 256 × 256 crop.
It initializes a fresh model with seed 42; no previous checkpoint is loaded.

```bash
python -u scripts/diagnose_astro_unet.py --full-slice --device cpu --learning-rate 0.0003 --steps 1000 --out samples/unet_slice512_lr0003_01
```

All reported scores are on that same training slice. Other slices/volumes are
not evaluated. Foreground occupancy changes from 30.63% in the crop to 16.66%
in the full slice, so this is not purely an input-size comparison on identical
content. Snapshot PNGs/probability TIFFs and history are saved every 50 steps.

Full-slice run completed in 839 seconds on CPU. Initial loss 1.4075; final loss
0.9677, Dice 0.002281, IoU 0.001142. Maximum logged Dice was 0.7021 at step 700,
followed by substantial regression. Final predicted foreground fraction 0.0591%
versus 16.6649% reference. Thus the final model is nearly all background and did
not pass the memorization target. This does not establish that 512 inputs are
inherently worse; content, occupancy and optimization behavior differ. All source
hashes were rechecked unchanged. The run folder `index.html` shows loss/Dice/IoU
curves and a step slider for intermediate prediction/reference overlays.

## sample_01 random-crop development run

`train_astro_single_volume.py` reads only sample_01. Valid Z<96 form training
(95 slices; Z=9 excluded), Z>=112 validation (15 slices; Z=124 excluded), and
Z=96..111 is an unused guard band. These same-cell blocks are not independent
biological samples. Eight fixed evenly spaced training slices are monitoring
only, not the entire training set. Full 512-square slices are scored at threshold
0.5 every 100 updates, without augmentation. No sample_02/03 test is performed.

```bash
python -u scripts/train_astro_single_volume.py --out samples/unet_s01_random256_lr0003_01
```

A fresh width-16 U-Net uses seed42, Adam LR0.0003, BCE+soft Dice, 1000 steps and
batch size2. Each crop selects a valid training slice uniformly and an XY origin
uniformly from 0..256, then applies paired rot90 and random horizontal reflection.
`sampling.jsonl` records coordinates, augmentation and foreground occupancy for
all 2000 draws. `metrics.csv`/`history.json` record training-monitor and validation
loss/Dice/IoU/foreground fractions; `last.pt` is the final model, `best.pt` retains
the earliest validation-Dice maximum for reference. Fixed previews show training
Z=0,47 and validation Z=112,127 every100 steps. `result.json` includes sampling
statistics and explicitly reports no test evaluation. Inputs are hash checked.

The run completed in 441 seconds (7.4 minutes). Final training-monitor loss
1.1961 and validation loss1.1782; both Dice/IoU and predicted foreground fractions
were zero at threshold0.5. This is an unsuccessful initial segmentation run,
not convergence. Of 2000 crops, 1.2% were background-only and average foreground
occupancy was8.53%; all95 training slices appeared. Thus exclusively empty crops
cannot explain the result. Full-slice monitoring also differs from patch-shaped
training inputs; learning/threshold behavior should be diagnosed before assigning
a single cause. Source hashes were unchanged. Nine checks passed including
recorded-coordinate reconstruction and unchanged RNG behavior when trace logging
is enabled. `samples/unet_s01_random256_lr0003_01/index.html` provides curves and
fixed training/validation previews. No extra training budget or parameter tuning
was performed after observing these results.

## Frozen-model context check (2026-10-30)

`python scripts/compare_astro_context.py --out samples/unet_s01_context_check_01`
loads the single-volume run's final checkpoint, verifies source hashes and chooses
20 evenly spaced sampling-record indices before inference. No optimization or
threshold tuning occurs. It compares native 256 crops, matching regions cropped
from full512 inference, and exact recorded rotation/reflection replay restored to
native coordinates. All three pooled Dice/IoU values are zero (TP=FP=0,
FN=108077). Native crop foreground/background mean probabilities are
0.132818/0.132856; full-context values are0.130730/0.130629. Maximum probabilities
are0.34976 and0.14790 respectively. Exact augmented replay also yields zero Dice.
Mean absolute crop/full probability difference is0.002706. This argues against
input-size change alone explaining the failure; the final model has little mean
probability separation on these sampled training regions. It does not identify
the optimization cause or prove the distributions have no discriminative signal.
Overlapping sampled pixels are counted repeatedly, so this is a training diagnostic,
not an independent validation score. All8 augmentation inverse cases and metric
sanity checks passed. Model/source hashes remain unchanged. HTML, per-region CSV,
probability TIFFs and selection/provenance JSON are saved in the output folder.

## Four fixed patches (2026-10-30)

`diagnose_astro_four_patches.py --out samples/unet_four_patches_lr0003_01`
selects four evenly spaced prior training draw indices (0,666,1332,1999) before
inference, from Z=8,2,64,89. Source hash and training membership are checked.
Native crops have foreground fractions5.15%,1.67%,6.77%,0.88%. A fresh seed42
width16 model, CPU, LR0.0003, BCE+soft Dice and batch2 alternates pairs [0,1]
and [2,3], without augmentation for1000 updates (500 exposures per crop).
Every50 steps, each full256 crop is scored/saved individually without gradients.
All four final Dice scores exceed0.95:0.9720,0.9663,0.9663,0.9735. Pooled
Dice0.9688/IoU0.9394, mean loss0.0422; runtime364seconds. Source hashes unchanged.
This demonstrates simultaneous memorization of four sparse-target examples,
not generalization. Compared with random-crop training, diversity, repetition,
augmentation and ordering differ, so it does not isolate the causal factor.
The run folder contains config, history, per-patch CSV, final weights, snapshots
and an HTML curve/preview viewer. Existing nine regression checks passed.

## Paired-augmentation control (2026-10-30)

`python scripts/diagnose_astro_four_patches.py --augment --out samples/unet_four_patches_aug_lr0003_01`
adds independently drawn rot90/reflection per patch exposure using a separate
NumPy RNG seed42. Image/mask transform identically. Crop pairs and optimizer/model
settings remain unchanged; no warm-start. `augmentation.json` records all2000
transforms. Eight paired transform/inverse cases and nine regression checks pass.
Shared configuration fields match the earlier no-augmentation run; all four
step0 metric/probability records match exactly, supporting identical initialization.

After1000 updates (366seconds), augmented mean training loss1.1199 and all four
native-orientation Dice/IoU scores0, versus unaugmented pooled Dice0.9688,
IoU0.9394 and mean loss0.0422. Additional frozen inference on all8 orientations
for all4 patches (32 cases) also gives Dice0. Source hashes are unchanged.
The experiment shows this augmentation setting makes fitting harder under this
seed/budget/optimizer, not that augmentation generally harms generalization.
It also does not establish the cause of all previous random-crop failures.
` samples/unet_four_patches_aug_comparison_01/index.html` compares curves and
snapshots. Next isolate rotation vs reflection, or first establish a multi-patch
unaugmented baseline; avoid interpreting memorization scores as validation.

## Random-crop augmentation ablation (2026-10-30)

Command (fresh initialization, not warm-start):
```sh
.venv-unet/bin/python -u scripts/train_astro_single_volume.py \
  --out samples/unet_s01_random256_noaug_01 \
  --replay-without-augmentation samples/unet_s01_random256_lr0003_01
```

Replays all 2000 original training crops in exactly the same order, disabling
rotation/flip. Seed42, Adam LR0.0003, batch2, width16, CPU4 threads, 1000
updates and monitor/validation split are unchanged. Replay checks source hashes,
training membership, coordinates, foreground fractions, model script hash,
PyTorch version and core configuration. Regression checks cover native image/mask
reconstruction and unchanged sampling metadata; all nine checks pass.
No sample_02/03 loading or test evaluation; source hashes unchanged.

Final whole-slice training monitor loss1.199455, Dice/IoU0, predicted foreground0%;
validation loss1.186939, Dice/IoU0, predicted foreground0%; elapsed433.18seconds.
Augmentation-ON baseline validation loss1.178204, Dice/IoU0. Therefore disabling
augmentation alone did not resolve all-background whole-slice inference for this
seed and budget. This does not measure native-crop fitting directly and does not
establish which remaining factor is causal.

Final weights: samples/unet_s01_random256_noaug_01/last.pt.
Reproducible comparison builder:
` .venv-unet/bin/python scripts/report_astro_sampling_ablation.py `
Output: samples/unet_s01_random256_aug_comparison_01/index.html.
Report asserts matching XYZ/step/size/foreground for every draw and no transforms
in the new run; plots monitor/validation curves and matched Z/step overlays.

## Native-crop check after augmentation ablation (2026-10-30)

Frozen last.pt step1000 from unet_s01_random256_noaug_01 was evaluated using
compare_astro_context.py on the same 20 evenly spaced training draws as the
earlier ON context check. All XYZ, step and draw indices match. Source and
checkpoint hashes are unchanged; no optimization or new training.
Output: samples/unet_s01_noaug_context_check_01/index.html.

Native256 pooled Dice/IoU0, TP/FP0, FN108077; probability range0.07218–0.37578;
mean foreground probability0.15000 vs background0.14883. Matched512 inference
crops also Dice/IoU0, probability range0.10009–0.18369; foreground mean0.14767
vs background0.14755. Mean absolute context difference0.007486.
Replay has identity transforms and exactly matches native predictions.
All 20 sampled crops fail to produce foreground at threshold0.5 even at the
training input size. Thus whole-slice context alone does not explain this
failure. Mean probabilities suggest weak separation but are not a ranking metric.
These are overlapping training draws, not an exhaustive training-set evaluation
or independent validation. Nine regression checks pass.

## Sixteen fixed patches, matched learning rate (2026-10-30)

Command:
```sh
.venv-unet/bin/python -u scripts/diagnose_astro_four_patches.py \
  --patch-count 16 --steps 1000 --extend-if-needed 4000 \
  --out samples/unet_sixteen_patches_lr0003_01
```

Generalized the fixed-patch diagnostic (default remains four) to cycle consecutive
pairs of an even patch count. Selection uses 16 evenly spaced draw indices
[0,133,266,399,533,666,799,932,1066,1199,1332,1465,1599,1732,1865,1999].
The previous four crops are included at positions0,5,10,15; their complete step0
metric/probability records exactly match the four-patch baseline. There are15
distinct source Z values; repeated Z64 uses different XY crops. All are training
records and contain foreground/background, with no output-based selection.
Fresh seed42, CPU4 threads, width16, Adam LR0.0003, batch2, no augmentation.
Source hashes remain unchanged.

At step1000 all16 Dice scores were0, so the predeclared continuation ran to4000
without resetting model or optimizer. Each patch received exactly125 exposures
at1000 and500 at4000. Both step_1000.pt and last.pt retain optimizer/model states.

| Step | Mean training loss | Pooled Dice | Pooled IoU |
| --- | --- | --- | --- |
| 1000 | 1.196430 | 0 | 0 |
| 3000 | 0.800992 | 0.537161 | 0.367204 |
| 4000 | 0.084904 | 0.954663 | 0.913258 |

Total elapsed1589.44seconds (26.49minutes). Final individual Dice range
0.898622–0.975939; 9/16 reach0.95, so the all-patches diagnostic criterion is
NOT met despite pooled Dice exceeding0.95. Lowest patch is index13.
Longer training substantially improved fitting of these fixed crops at unchanged
LR; this does not establish that random-crop training will converge similarly,
nor does it measure generalization. Pooled counts can include overlapping pixels.

The logging denominator remains the initial1000 during continuation; recorded
step values, checkpoint metadata and actual schedule correctly reach4000.
Nine existing regression tests pass; both checkpoint steps and every snapshot
image referenced by the report were verified.
Report builder:
` .venv-unet/bin/python scripts/report_astro_fixed_patches.py --run samples/unet_sixteen_patches_lr0003_01 `
Viewer: samples/unet_sixteen_patches_lr0003_01/index.html; per-patch metrics.csv.

## Shared held-out-Z validation for 1/4/16-patch models (2026-10-30)

The three final checkpoints were evaluated without parameter updates on the
same within-sample validation block: sample_01 nonconstant slices Z>=112
(15 slices; constant Z124 excluded). Threshold is0.5. `full_512` uses one
forward pass per full slice. `tiled_256` uses nine overlapping256x256 tiles at
XY starts0,128,256 and averages probability in overlaps before scoring the
complete512x512 reference mask. All checkpoints and source files are hashed;
the three run configs reference identical source hashes.

| Training set / final step | Inference | Validation loss | Pooled Dice | Pooled IoU | Predicted foreground |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 fixed patch / 1000 | full512 | 7.2008 | 0.1390 | 0.0747 | 55.32% |
| 1 fixed patch / 1000 | tiled256 | 4.8969 | 0.1301 | 0.0696 | 44.25% |
| 4 fixed patches / 1000 | full512 | 2.1279 | 0.1204 | 0.0640 | 5.81% |
| 4 fixed patches / 1000 | tiled256 | 1.9826 | 0.1082 | 0.0572 | 4.53% |
| 16 fixed patches / 4000 | full512 | 1.7817 | 0.0761 | 0.0395 | 5.64% |
| 16 fixed patches / 4000 | tiled256 | 1.6637 | 0.0651 | 0.0336 | 4.04% |

The labeled validation foreground fraction is6.57%. Every checkpoint has low
validation overlap despite strong fitting on its fixed training crops. The
16-patch model's pooled training Dice0.9547 does not transfer to this held-out
Z block (full512 Dice0.0761). The 1-patch model's superficially highest Dice is
driven by severe overprediction (55.32% foreground) and has the worst loss; it
is not the best calibrated model. Tiled inference is slightly worse than
full-slice inference for all three, so input size/context does not explain the
generalization failure. These are development comparisons within one astrocyte,
not an independent test or biological replication.

Reproducible evaluator: `scripts/evaluate_fixed_patch_checkpoints.py`.
Viewer: `samples/unet_fixed_patch_shared_validation_01/index.html`.
`slice_metrics.csv` contains all90 per-slice/mode/model score rows; all90
prediction/reference overlays and checkpoint steps were verified. Existing
nine regression tests pass.

## Shared-validation threshold and ranking diagnostics (2026-10-30)

`analyze_validation_thresholds.py` reruns frozen inference for the same three
checkpoints and held-out Z block. It computes exact non-interpolated Average
Precision (equal scores grouped), probability distributions, and hard metrics
over thresholds0, a logarithmic grid from1e-8 to0.1, then0.1–1.0 by0.01.
The logarithmic range is necessary because the16-patch model assigns most
pixels probabilities below0.01. Best thresholds are chosen and scored on this
same validation set, so they are diagnostic values, not test performance.

| Model | Mode | AP | Dice @0.5 | Best threshold | Best Dice |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 patch | full512 | 0.0827 | 0.1390 | 1.0 | 0.1441 |
| 1 patch | tiled256 | 0.0761 | 0.1301 | 0.24 | 0.1352 |
| 4 patches | full512 | 0.0929 | 0.1204 | 1e-8 | 0.1413 |
| 4 patches | tiled256 | 0.0893 | 0.1082 | 1e-8 | 0.1411 |
| 16 patches | full512 | 0.0748 | 0.0761 | 0.000200 | 0.1292 |
| 16 patches | tiled256 | 0.0785 | 0.0651 | 0.000631 | 0.1370 |

Foreground prevalence is0.0657, also the expected AP of an uninformative random
ranking. All AP values are only slightly above this baseline, and even
validation-selected thresholds yield Dice only0.129–0.144. An all-foreground
prediction itself has Dice0.1233. Therefore threshold calibration does not
explain the poor validation performance: foreground and background rankings
overlap heavily. The4-patch model has the highest AP, while the16-patch model's
extra memorization does not improve ranking on held-out Z.

Viewer: `samples/unet_fixed_patch_threshold_diagnostics_02/index.html`.
`threshold_metrics.csv`, `results.json`, and hashed probability TIFF volumes
make the curves reproducible. Three focused AP/threshold tests were added; all
12 astro regression tests pass.
