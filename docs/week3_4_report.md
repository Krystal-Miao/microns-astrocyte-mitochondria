# Week 3–4 Report — EM Denoising Pilot

Updated: 2026-10-18

## Summary and scope

This pilot compares non-local means and bilateral filtering on three EM crops from one astrocyte, examining the balance between smoothing and visible structural detail. Sample 01 was used to explore settings; H2 and C2 were then compared with fixed parameters on samples 02 and 03.

## Literature inspiration and experimental decisions

DenoisEM motivated an interactive workflow: select small regions, compare parameter settings, and inspect both denoising and structural effects. Its evaluation also supports considering task performance alongside image appearance. Here, this became a limited pilot with boundary, texture, and relatively uniform regions rather than immediate processing of the entire volume. This is an adaptation, not a replication of its benchmark. [Roels et al., DenoisEM, 2020](https://www.nature.com/articles/s41467-020-14529-0)

Noise2Self provides a framework for evaluation without clean targets under specific statistical assumptions. We have not established those assumptions for these images or implemented a masked, self-supervised evaluation. Consequently, similarity to the noisy input and residual magnitude are not treated as measures of denoising accuracy. [Batson and Royer, Noise2Self, 2019](https://proceedings.mlr.press/v97/batson19a.html)

The accompanying literature notes also cover BM3D, total variation, anisotropic diffusion, Noise2Void, and structured-noise approaches. These remain possible future comparisons; they were not run in this experiment. Results from nanoparticle TEM or tomography are adjacent evidence, not direct validation for this serial-section biological EM dataset.

## Data and quality control

- Dataset: MICrONS `minnie65_public`, materialization 1300; astrocyte ID `864691136057675736`.
- Three downloaded crops, each 512 × 512 × 128 voxels in XYZ, with downloaded spacing 8 × 8 × 40 nm. TIFF arrays use ZYX order.
- All crops belong to one astrocyte. They are not independent biological replicates.
- Evaluation ROIs are 192 × 192 pixels in XY. Each boundary, texture, or uniform candidate is 32 × 32 pixels, corresponding to 0.256 × 0.256 µm.
- The astrocyte mask indicates cell membership; it is not a mitochondrion or contact label.
- Sample 02 source Z=108 is an all-zero slice of unresolved origin. It was excluded from display-window estimation and this comparison, and was not interpolated or modified.
- Candidate locations were proposed on raw images. Uniform candidates were selected using low central-slice intensity SD and high mask occupancy. They may contain biological structure and are not pure-noise patches.
- Candidate regions may overlap. Neighboring sections contain related but different anatomy.

The existing candidate manifest remains a record of proposed locations. Method review does not automatically constitute anatomical confirmation of those candidates.

## Experimental protocol

Sample 01 was used for exploratory parameter comparison at Z=46, 47, and 48. Five conditions were inspected: raw, NLM H2, NLM H3, bilateral C2, and bilateral C4. H2 and C2 were retained for fixed-parameter comparisons on sample 02 at Z=104–106 and sample 03 at Z=17–19.

Samples 02 and 03 are additional evaluation crops, not a blinded test set: their raw images had already been explored, and all crops come from the same cell.

Each selected full 512 × 512 plane was processed independently in 2D, using float32 intensities divided by 255. The evaluation ROI was then cropped. The mask was not used to zero the image. No cross-section denoising, resampling, or CLAHE was used in this comparison.

| Setting | Parameters |
| --- | --- |
| NLM H2 / H3 | h=2/255 or 3/255; patch_size=5; patch_distance=6; fast_mode=True; sigma=0; preserve_range=True; channel_axis=None |
| Bilateral C2 / C4 | sigma_color=2/255 or 4/255; sigma_spatial=1 pixel; win_size=7; bins=10000; mode=reflect; cval=0; channel_axis=None |

NLM sigma=0 is an algorithm configuration, not a measured absence of noise. These presets are not matched denoising strengths and have not been shown to be optimal.

## Display and diagnostic controls

Each ROI used one raw-derived 1st–99th percentile display window, fixed across methods and neighboring sections: sample 01 [112, 160], sample 02 [115, 162], and sample 03 [112, 156]. Windows differ across samples. Pixel magnification uses nearest-neighbor display.

Residuals are raw minus processed output in original 0–255 intensity units. Each run uses a common symmetric residual scale across its panels and sections; scales can differ between runs. Residuals show changed content, not proven noise.

Boundary profiles sample 25 pixels along the proposed line. They are intensity profiles, not fitted edge-width measurements or measurements of membrane thickness. The line must be anatomically appropriate before interpreting it as a boundary diagnostic.

## Quantitative observations: relatively uniform patches

The table reports pixel intensity SD within the same candidate patch and section. Values are descriptive and use original 0–255 intensity units. Lower SD indicates less variation, which may reflect both noise suppression and loss of structure.

| Sample | Source Z | Raw SD | H2 SD | C2 SD |
| --- | --- | --- | --- | --- |
| 01 | 46 | 4.367 | 3.596 | 3.891 |
| 01 | 47 | 3.335 | 2.313 | 2.667 |
| 01 | 48 | 5.097 | 4.448 | 4.498 |
| 02 | 104 | 5.968 | 5.438 | 5.536 |
| 02 | 105 | 4.436 | 3.713 | 3.974 |
| 02 | 106 | 8.340 | 8.009 | 7.902 |
| 03 | 17 | 4.549 | 3.681 | 3.806 |
| 03 | 18 | 4.164 | 3.342 | 3.537 |
| 03 | 19 | 5.411 | 4.733 | 4.647 |

H2 has lower patch SD than C2 in seven of the nine section/patch combinations. The exceptions are sample 02 Z=106 and sample 03 Z=19. This does not establish a universal ranking, statistical significance, or biological fidelity. These nine observations are correlated and are not nine independent replicates.

No clean-reference PSNR/SSIM, validated SNR estimate, or segmentation accuracy was established in this round.

## Visual review

### Sample 01

I initially preferred C2 as the best overall compromise. H2 looked slightly more uniform in the uniform region, but the difference was modest. I also felt that H2 might suppress more noise, although I could only confirm stronger apparent smoothing. The H2/C2 residual differences looked small, and I observed similar behavior at Z=46. I retained both methods for further comparison.

### Sample 02

I found that H2 and C2 had different strengths, with no clear winner. In the boundary, texture, and uniform comparisons at Z=105, both outputs retained the prominent dark boundary band. H2 looked smoother, while C2 retained more visible local variation. I could not determine boundary accuracy from these images alone.

### Sample 03

In the Z=18 comparisons, I could still recognize the prominent horizontal dark band in both outputs. H2 appeared to smooth the surrounding texture more strongly, while C2 retained more local variation. C2 also showed locally stronger residuals, so its less-smoothed appearance did not mean smaller changes everywhere. I did not identify a clear overall winner or establish which removed features were noise.

I have completed the neighboring-section review for samples 02 and 03, checked that the proposed profile lines cross the intended boundaries, and compared boundary profiles and residual patterns alongside image appearance. This completes the visual and profile review for the current pilot.

## Current conclusion and limitations

Retain H2 and C2 as two plausible candidates. H2 provides stronger apparent smoothing in many inspected regions; C2 provides a more conservative visual compromise in some regions. Neither is validated as the best preprocessing method.

The main limitations are the lack of a clean reference, a validated noise model, confirmed mitochondrion labels, expert anatomical validation, and independent biological samples. Uniform-region selection favors low initial variation, and overlapping patches and adjacent sections limit independence. Preserved visible contrast alone does not prove preservation of fine biological detail.

## Deliverables and reproducibility

Local artifact root:
`samples/astro_864691136057675736_run1/preprocessing/`

- `tuning_sample01_01/index.html`: five-condition exploratory comparison.
- `evaluation_sample02_01/index.html`: fixed H2/C2 comparison.
- `evaluation_sample03_01/index.html`: fixed H2/C2 comparison.
- Each run includes comparison, residual, and profile figures; float32 image/residual TIFFs; `diagnostics.csv`; `profiles.csv`; and `report.json` recording provenance and parameters.
- Generator: `scripts/run_denoising_tuning.py`.
- Experimental history: `docs/preprocessing_pilot.md`.

The three current result viewers and the generator's viewer text were converted to English. This presentation update did not rerun denoising or change TIFF, PNG, CSV, or JSON artifacts. Image references were checked. Source-integrity and numerical output checks from the pilot are retained in the run records.

These are local artifacts; listing them here does not mean the image data have been uploaded to Notion or published to GitHub.

## Next steps

1. **Evaluate downsampling before larger-field analysis.** I will create a lower-resolution copy of one raw sample, starting with 2× downsampling in XY while keeping Z unchanged (8 × 8 × 40 nm to 16 × 16 × 40 nm; 512 × 512 × 128 to 256 × 256 × 128 voxels in XYZ). I will retain the original and compare file size and the visibility of boundaries and fine structures to assess whether the storage reduction is acceptable. **Deliverable:** an original-versus-downsampled comparison and a file-size summary.
2. **Compare H2 and C2 at a larger field of view.** I will generate raw/H2/C2 comparisons for the full 512 × 512 planes at the reviewed Z positions, using the existing parameters. This will show whether the differences seen in small patches affect the appearance of larger structures. **Deliverable:** a comparison viewer for all three samples.
3. **Measure changes at selected boundaries.** I will mark a small set of clearly visible boundaries on the raw images, then compare their positions and intensity transitions in the H2 and C2 outputs. I will flag boundaries that cannot be located reliably. **Deliverable:** annotated examples and a table of boundary displacement and contrast changes. These annotations will serve as consistency references, not clean ground truth.
4. **Make a provisional preprocessing decision.** I will combine the completed patch review with the larger-field comparisons and boundary measurements to choose a default setting for the next pilot, or explicitly retain both if the evidence remains inconclusive. **Deliverable:** a short decision record with the selected parameters, supporting examples, and unresolved cases.

## Related records

- [Week 1–2 Report](https://app.notion.com/p/3ef86ce34b2880429674dacf72c2e5ec)
- [EM Denoising — Methods, Evidence & Detail Preservation](https://app.notion.com/p/3fd86ce34b2881fa9f6ccbf4edd6e433)
