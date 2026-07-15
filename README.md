# Vairons

A desktop tool for quantitative analysis of multi-channel fluorescence microscopy
images. It segments **nuclei**, optionally **cytoplasm** (whole cell minus
nucleus), and any number of **stain / marker** channels, then extracts per-cell
morphology, intensity, texture and stain features and writes tidy Excel summaries
with per-sample statistics and hierarchical aggregation. **2-D images and 3-D
z-stacks** are both supported (dimensionality is auto-detected per sample).

Segmentation **method is independent of channel role**: each detection (nuclei,
cytoplasm, each stain) can be segmented by **intensity thresholding**, **Cellpose**
(Cellpose-SAM / `cpsam`), or a **pre-segmented label image**, chosen per detection
with its own parameters. The nuclear channel is **optional** — with only stain
channels the tool runs in *particle mode* (one row per stain particle). An optional
**outlier-removal** tier adds a robustly-cleaned copy of the per-cell results.

---

## Contents

- [Architecture](#architecture)
- [Requirements & installation](#requirements--installation)
- [Input data layout](#input-data-layout)
- [Running the tool](#running-the-tool)
- [Channel roles](#channel-roles)
- [Segmentation methods](#segmentation-methods)
- [Outputs](#outputs)
- [Output columns reference](#output-columns-reference)
- [Batch processing & aggregation](#batch-processing--aggregation)
- [Batch QC report](#batch-qc-report)
- [Scientific notes & caveats](#scientific-notes--caveats)
- [Configuration reference](#configuration-reference)

---

## Architecture

Three modules, no framework:

| File | Responsibility |
|------|----------------|
| `processing.py` | Pure image-analysis algorithms — segmentation (threshold & Cellpose), feature extraction (morphology, intensity, GLCM texture), stain quantification, spatial statistics, per-image QC. No I/O, no UI. |
| `io_manager.py` | Orchestration & I/O — channel discovery, per-sample pipelines (`run_analysis` for nucleus-based cells, `run_particle_analysis` for nucleus-free particle mode), DataFrame/Excel construction, the cell filter, MAD outlier removal, and hierarchical aggregation. Holds `DEFAULT_CONFIG` / `DEFAULT_STAIN_CONFIG`. |
| `ui.py` | PyQt6 GUI — channel-role assignment, parameter panels, live single-image preview with mask overlays, and the batch runner. Entry point. |

---

## Requirements & installation

- **Python 3.13+** (validated on 3.14).
- See [`requirements.txt`](requirements.txt). Key constraint: **Cellpose ≥ 4.2, < 5**
  (the code targets the Cellpose-SAM 4.x API and will not run on Cellpose 3.x).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Cross-platform:** the tool runs on **Windows and macOS** (and Linux). The GUI uses
the Qt *Fusion* style with a dark palette, so it looks the same on every platform.

**GPU (optional but recommended for Cellpose):** the app **auto-detects the device**
(CUDA / MPS / CPU) and **uses the GPU automatically when one is present** (falling
back to CPU otherwise); the detected device is shown in the status bar.
- **macOS (Apple Silicon):** the default `torch` wheel already includes **MPS**
  (Metal) acceleration — no extra install needed.
- **Windows / Linux (NVIDIA):** the default `torch` wheel is CPU-only; install a
  CUDA build of PyTorch from <https://pytorch.org> for GPU acceleration.

The first Cellpose run downloads the model weights (a few hundred MB).

---

## Input data layout

Each **sample** is a folder containing one single-channel `.tif`/`.tiff` file
**per channel** (the extension is matched **case-insensitively** — `.tif`, `.tiff`,
`.TIF`, `.TIFF` all work). Images may be **2-D** `(Y,X)` or **3-D z-stacks** `(Z,Y,X)`;
dimensionality is **auto-detected per sample** from the TIFF metadata (the
`Z` axis in `series[0].axes`). The filename stem identifies the channel and is how
roles are assigned, so **channel filenames must be consistent across all samples**.

For 3-D stacks the **voxel depth (z-step)** and unit are read from the file
metadata (ImageJ `spacing` / OME `PhysicalSizeZ`); the lateral pixel size stays
user-set. Files bundling multiple channels/timepoints (`C`/`T` axes) are
**auto-split** into separate virtual channels (`<stem>_c0`, `<stem>_c1`, …), so a
manual pre-split is no longer required.

**Beyond TIFF.** ND2 (Nikon), CZI (Zeiss) and LIF (Leica) files are also read when
the optional [`aicsimageio`](https://github.com/AllenCellModeling/aicsimageio)
package is installed (`pip install aicsimageio`, plus its per-format reader); with a
plain TIFF-only environment nothing changes. `supported_image_extensions()` reports
what the current install can read.

**Calibration safety.** Embedded pixel sizes are trusted only when they carry a real
length **unit** (µm / nm / mm / …) and are converted accordingly; a bare TIFF
resolution in inches or with no unit (a print-DPI default) is treated as
**uncalibrated** so it can't be mistaken for a microscope pixel size. Precedence is
**your configured pixel size → file metadata → a 1 µm/px fallback**, and if a sample
falls through to that fallback the run prints a loud warning and records it. Every
sample's `qc_meta.json` carries a `calibration` block (`pixel_size_um`, `source`,
`calibrated`) so the provenance of every µm²/µm³ value is auditable.

```
experiment_root/
├── condition_A/
│   ├── sample_001/
│   │   ├── DAPI.tif          # nuclear channel
│   │   ├── membrane.tif      # (optional) for cytoplasm
│   │   └── marker.tif        # a stain
│   └── sample_002/
│       ├── DAPI.tif
│       ├── membrane.tif
│       └── marker.tif
└── condition_B/
    └── ...
```

- Images are expected to be single-channel grayscale (multi-channel `C`/`T`
  acquisitions are auto-split on load, as above).
- 8-bit or 16-bit are both supported; intensity thresholds are applied to the
  **raw** pixel values (the GUI hover read-out shows raw intensities).
- For 3-D stacks the preview shows one z-plane at a time (a **Z slider** scrolls
  the stack); segmentation and features are computed on the full volume.

**Sample organization (`input_mode`).** Two layouts are supported, selectable in
the GUI or via `config['input_mode']`:

- `folder_per_sample` (the default and the layout shown above) — each folder is a
  sample; the files (or `C`/`T` splits) inside it are its channels.
- `file_per_sample` — each **multi-channel image file** is a sample, and its bundled
  channels are exposed under bare positional names (`c0`, `c1`, …); output goes to
  `<file_parent>/<filestem>/Analysis/`.
- `auto` prefers `folder_per_sample` and falls back to `file_per_sample` only when
  no folder offers the primary channel.

---

## Running the tool

```bash
python ui.py
```

Typical workflow:

1. **Test Image → Browse** a single sample folder — or **drag & drop** a channel
   folder (or a multi-channel image) anywhere onto the window. Channels load and a
   preview appears. Dropping several loose channel files loads their parent folder.
2. In the **Channels** grid (top of the middle panel) each channel gets a row:
   assign its **role**, adjust display **Contrast**, toggle the channel **Show**
   layer, and (after Apply) toggle its **Mask** and set the mask **Opacity**.
   Assigning a role reveals its settings tab in the left panel.
3. Tune parameters in the left panel. The **XY pixel size** (and, for z-stacks, the
   **Z size**) lives in the top **Setup** section and is auto-filled from the image
   metadata when present (editable — override it if the metadata is missing or
   wrong). Per-detection settings live in **tabs** below Setup (**Nuclear**,
   **Cytoplasm**, one per stain — each appears once its role is assigned); choose
   **Method: Threshold, Cellpose, Pre-segmented or None** per detection. The
   **Analysis Filter** and **Outlier Removal** controls are in the right panel; the
   **Sample-level features** and **Selected Cell Features** appear below them. Click
   **Apply on Current Image** to preview, and click a cell to inspect its features.
   **Save Current Image Result** writes just this image's `Analysis/` output (masks +
   `Feature_summary.xlsx`), exactly as Run Batch would.
4. **Input Folder → Browse** the experiment root, then **Run Batch**. Every
   sample folder under the root that contains the **primary channel** (the nuclear
   channel, or — in particle mode — the first stain) is processed, and aggregated
   summaries are generated at each folder level.

You can also run a batch programmatically:

```python
from io_manager import DEFAULT_CONFIG, run_batch
config = DEFAULT_CONFIG.copy()
config['channel_roles'] = {'DAPI': 'nuclear', 'marker': 'stain'}
run_batch('path/to/experiment_root', config)
```

---

## Channel roles

| Role | Meaning |
|------|---------|
| `nuclear` | The channel used to segment nuclei. Optional — see *Particle mode* below. |
| `membrane` | A membrane channel used as the Cellpose input for cytoplasm. |
| `cytoplasm` | A cytoplasm channel used as the Cellpose input for cytoplasm. |
| `stain` | A marker channel to segment and quantify per cell. Any number allowed. |
| `none` | Ignored for segmentation (still reported as a per-cell mean intensity). |

Cytoplasm = the whole-cell mask (Cellpose or nuclei-seeded watershed) **minus**
the nuclei, so nuclear and cytoplasmic measurements are disjoint (no double
counting).

**Particle mode (no nuclear channel).** If **no** channel is assigned `nuclear`,
the tool analyses each `stain` channel's segmented objects as **standalone
particles**: the output has one row per particle (morphology + intensity, a
`stain` column identifying the channel) and per-stain sample-level summaries
(particle count, total/mean area, whole-image coverage, alignment). Cytoplasm and
the analysis filter are disabled (both require nuclei). In the live preview the
first stain's particles are click-inspectable; the batch run covers every stain.
At least one `nuclear` **or** `stain` channel is required.

---

## Segmentation methods

Each detection exposes a **Method** selector. The default is **threshold** for
nuclei and stains and **Cellpose** for cytoplasm; nuclei and cytoplasm additionally
offer **pre-segmented** (load label images). Every detection also offers **`none`**
— skip that segmentation: a `none` stain stays a measured channel (its per-cell
mean is still reported, but no `stain_<s>_*` columns), `none` nuclei drop the sample
to particle mode, and `none` cytoplasm disables cytoplasm.

**Threshold** (nuclei & stains)
- Pixels in `[threshold_min, threshold_max]` → binary mask → fill holes → label.
- Objects smaller than `min_object_size` (px) are dropped; nuclei within
  `edge_exclusion_margin` of the border are dropped.
- Nuclei: a heuristic doublet score (circularity, solidity, eccentricity, size)
  flags likely doublets, which are split by a distance-transform watershed.
- Stains can output a **binary** mask or **labeled** instances (with the same
  doublet-splitting option).

**Cellpose** (nuclei, stains, cytoplasm)
- Runs the selected model (`cpsam` by default) and returns instance labels
  directly, so no doublet/watershed step is needed.
- `min_object_size` and (for nuclei) `edge_exclusion_margin` still apply.
- Each detection keeps **its own** Cellpose parameters (model, diameter,
  cellprob/flow thresholds, niter, min size, max size fraction, invert). The
  **GPU is used automatically** when one is detected (no toggle). Models are
  cached per `(model_type, gpu)` and reused across detections and across the batch.
- **Max Size Fraction** (default `0.4`) is a Cellpose cut-off: any object covering
  more than this fraction of the image is *discarded*. Raise it towards `1.0` for
  cropped fields or sparse, very large cells — otherwise they disappear and no
  amount of cellprob/flow tuning brings them back.
- **Niter** `0` means *auto*: Cellpose scales its 200 dynamics iterations by
  `30/diameter`. Only pin a value if you know you need more steps.

**3-D / volumetric analysis.** When a sample is a z-stack, segmentation runs in 3-D
automatically: Cellpose uses `do_3D=True` with the voxel **anisotropy** (z/xy) from
the metadata, and threshold segmentation labels 3-D connected components (the 2-D
doublet/watershed split is disabled — Cellpose separates touching nuclei natively).
Features become volumetric: **area → volume (µm³)**, **perimeter → surface area
(µm², marching cubes)**, **circularity → sphericity**, **eccentricity →
elongation/flatness**; nuclear alignment uses a **3-D nematic order parameter**,
texture is a **per-slice GLCM** averaged over z, and stain spatial descriptors use
the nucleus's principal axes. **Anisotropy-corrected shape:** the principal axes
(major/minor axis length, elongation, flatness) and the alignment directors are
computed from voxel coordinates **scaled to physical µm** before eigen-decomposition,
so they are correct when the z-step differs from the xy pixel size (rather than
biased by the voxel aspect ratio). The 2-D outputs and column names are unchanged;
3-D samples emit the `*_um3` / volumetric columns, and mixed batches are supported.

**Cytoplasm** can also be **Cellpose** *or* **Threshold**. Because cytoplasm
must be split *per cell*, the threshold method runs a **nuclei-seeded watershed**:
it floods outward from each nucleus over a thresholded foreground and assigns each
territory to its nucleus, then subtracts the nucleus. The flooding landscape
adapts to the **Source**: a *membrane* channel uses its bright ridges as cell
boundaries; a *cytoplasm* channel partitions the thresholded fill by proximity to
the nearest nucleus. Resulting cytoplasm ids match their nucleus id (Cellpose is
the default cytoplasm method).

**Pre-segmented masks** are exposed as a third **Method** (alongside Threshold and
Cellpose) for the nuclear and cytoplasm detections: pick *Pre-segmented* and give
the label-image **stem** (e.g. `nuclear_labels`, `cytoplasm_labels`), loaded from
the sample folder or its `Analysis/` subfolder. For cytoplasm, the **Source** still
selects which channel's intensity is measured inside the loaded mask, so a source
must be chosen.

---

## Outputs

For each sample, results are written to `<sample>/Analysis/`:

```
Analysis/
├── nuclear_labeled.tif / nuclear_binary.tif        (cell mode)
├── cytoplasm_labeled.tif / cytoplasm_binary.tif    (if cytoplasm enabled)
├── stain_<name>_binary.tif  or  stain_<name>_labeled.tif
├── Feature_summary.xlsx
└── qc_meta.json                                    (config + counts sidecar for the QC report)
```

In **particle mode** (no nuclear channel) only the `stain_<name>_*` masks,
`Feature_summary.xlsx`, and `qc_meta.json` are written. For 3-D samples the label
TIFFs are saved as ImageJ stacks carrying the voxel calibration.

`Feature_summary.xlsx` has up to six sheets — three progressively-cleaned cell
tiers (**all cells → filtered → outlier-removed**), each paired with a
sample-level sheet:

| Sheet | Unit | Contents |
|-------|------|----------|
| `All_cells` | one row per nucleus | morphology / intensity / texture / per-channel means (nucleus `N_*`, cytoplasm `C_*`) and per-stain quantities; a `MEAN` row. |
| `Sample_level` | one metric per row | sample summaries: cell count, nuclear alignment, and per-stain area / coverage / particle-count / spatial means **with SEM**, plus per-image QC. |
| `Filtered_cells` | one row per kept cell | `All_cells` restricted to the analysis filter. |
| `Filtered_sample_level` | one metric per row | sample summaries recomputed on the filtered subset. |
| `Outlier_removed` | one row per kept cell | `Filtered_cells` with statistical outliers removed (see below). Present only when outlier removal is enabled. |
| `Outlier_removed_sample_level` | one metric per row | sample summaries recomputed on the outlier-removed subset. |

**Outlier removal.** When enabled (default), a third tier drops cells flagged as
outliers by **one metric, with one vote per channel**.

- **The metric is the same for every channel, whatever its role**: `log10` of the
  object's size — `area_um2` in 2-D, `volume_um3` in 3-D. Each channel screens its
  own objects: one nucleus per cell, one cytoplasm per cell, many particles per cell
  for a stain. Intensity, texture and per-channel means are **excluded** — they
  reflect biology, so genuinely high-expressing cells are **not** removed.
- **The test is a two-sided robust z**, `0.6745·(x − median) / MAD`, against
  `outlier_mad_threshold` (default `4`). The upper tail catches merged blobs and
  doublet nuclei; the lower tail catches fragments.
- **Each channel casts exactly one vote per cell**: `nuclear`/`cytoplasm` if that
  cell's own object is an outlier, a stain if the cell owns **≥1** outlier particle.
  A cell is removed when **any** channel votes.

**Why `log10(size)` and not the raw area.** The `0.6745` factor calibrates the
robust z against a *normal* distribution — that is what makes a threshold of 4 or 5
mean anything. Stain-particle sizes are strongly right-skewed (skewness ≈ 5), and on
the raw scale the calibration collapses: `|z| > 5` fires on ~**13 %** of perfectly
ordinary puncta, because there it merely means "area > 7.6 × the median". Worse,
since size > 0 the raw lower tail is unreachable (`z → −0.76` as area → 0), so a
two-sided test is silently one-sided. `log10` restores both the calibration and the
lower tail, and it is free for near-symmetric objects such as nuclei.

**Why size alone, and one metric per channel.** Shape is unresolvable at particle
scale — `solidity`, `eccentricity` and `convexity_defects` all have MAD = 0 on puncta,
and `4πA/P²` exceeds 1 for small digital objects. Screening several columns and OR-ing
them also gives a channel more chances to flag, so more metrics silently means more
removals; one metric per channel makes every channel weigh in equally.

A channel whose MAD is 0 (constant size) is skipped, and non-positive or missing sizes
never flag. Stain particles are labelled **once over the whole image** and assigned to
the cell owning most of their pixels, so a punctum straddling the nucleus/cytoplasm
border counts as one particle, not two fragments.

The QC sidecar records `outlier_screen` (metric, statistic, sidedness, threshold,
aggregation), `outlier_cells_flagged_by_channel` (each channel's votes) and
`outlier_size_distribution_by_channel` (n, median, MAD on the log scale, robust-z
range, flag rate) — so an over-firing channel is visible at a glance. Toggle the tier
and tune the threshold in the **Outlier Removal** panel.

In **particle mode** (no nuclear channel) the cell sheets are replaced by
`All_particles` (one row per stain particle) + `Sample_level`, plus the
`Outlier_removed` tier. The same screen runs per stain channel; since there are no
cells, a flagged particle drops itself. There is no `Filtered_*` tier.

**Statistics convention:** SEM is reported **only** on the sample-level sheets
(replication unit = the sample). The cell-level sheets carry a `MEAN` row only,
because individual cells are not independent replicates.

---

## Output columns reference

**Per-cell (`All_cells` / `Filtered_cells`)** — prefix `N_` = nucleus,
`C_` = cytoplasm:

| Column | Meaning |
|--------|---------|
| `cell_id` | Nucleus label id. |
| `N_area_um2`, `N_perimeter_um` | Calibrated area / perimeter (µm², µm). |
| `N_circularity`, `N_solidity`, `N_eccentricity`, `N_convexity_defects` | Shape descriptors. |
| `N_major_axis_length_um`, `N_minor_axis_length_um` | Fitted-ellipse major / minor axis lengths (µm). |
| `N_intensity_mean`, `N_intensity_sd` | Mean / SD of the nuclear channel within the nucleus. |
| `N_texture_*` | GLCM contrast / homogeneity / energy / correlation. |
| `N_<channel>_mean` | Mean intensity of each *other* channel within the nucleus. |
| `C_*` | The same set, measured on the cytoplasm (present only if cytoplasm enabled). `C_texture_*` is omitted when cytoplasm is derived from a **membrane** channel, where boundary signal makes texture meaningless. |
| `nuclei_in_cell` | Number of nuclei sharing this cell's cytoplasm territory (`1` = mononucleated, `≥2` = polynucleated). Blank when the cell has no cytoplasm (nucleation cannot be assessed without a cell boundary). Only the Cellpose cytoplasm method can yield `≥2`; the nuclei-seeded watershed always gives one nucleus per territory. |
| `stain_<s>_area_nuc_um2` / `_cyto_um2` / `_cell_um2` | Absolute stain area in the nucleus / cytoplasm / whole cell (µm²). `_cyto` is blank when the cell has no cytoplasm; **when cytoplasm is enabled but a cell has none, all whole-cell `_cell_*` measurements are also blank** (an incomplete cell cannot be measured as a whole cell). |
| `stain_<s>_coverage_fraction_nuc` / `_cyto` / `_cell` | Fraction of the nucleus / cytoplasm / whole cell covered by stain *s* (0–1). |
| `stain_<s>_particle_count_cell` | Number of stain particles in the whole cell. |
| `stain_<s>_particle_avg_area_nuc_um2` / `_cell_um2` | Mean stain-particle area within the nucleus / over the whole cell (µm²). |
| `stain_<s>_cell_stain_area_px` | Whole-cell stain area (px); used internally by the filter. |
| `stain_<s>_nuc_has_stain` | Nucleus overlaps stain *s* (any pixel). |
| `stain_<s>_cell_has_stain` | Whole-cell stain area ≥ the **mean area of one stain-*s* particle in that image** (the filter criterion). Falls back to "any whole-cell stain pixel" only if the image has no particles. |

**Sample-level metrics** (`metric` / `value`):

| Metric | Meaning |
|--------|---------|
| `cell_count` | Number of nuclei. |
| `nuclear_alignment` | Axial alignment order parameter of nuclear orientations (0 = random, 1 = aligned). |
| `mononucleated_cell_count` / `polynucleated_cell_count` | Number of unique cytoplasm territories holding exactly one / two-or-more nuclei. Emitted only when cytoplasm is enabled. |
| `polynucleated_cell_fraction` | `polynucleated_cell_count / (mono + poly)` — fraction of cells that are polynucleated. |
| `stain_<s>_positive_fraction_cell` | Fraction of cells scored stain-*s*-positive (0–1) — the per-cell `stain_<s>_cell_has_stain` flag averaged over the sample. Answers "% marker-positive cells"; dimensionless, so the name is unchanged in 3-D. |
| `stain_<s>_area_cell_um2_mean` / `_sem` | Mean ± SEM of whole-cell stain area (µm²) across cells. |
| `stain_<s>_particle_count_cell_mean` / `_sem` | Mean ± SEM of stain-particle count per whole cell. |
| `stain_<s>_particle_avg_area_nuc_um2_mean` / `_cell_um2_mean` (+ `_sem`) | Mean ± SEM of nuclear / whole-cell stain-particle area across cells. |
| `stain_<s>_coverage_fraction_cell_mean` / `_sem` | Mean ± SEM of whole-cell stain coverage across cells. |
| `stain_<s>_alignment_mean` / `_sem` | Mean ± SEM of within-nucleus stain-particle alignment. |
| `stain_<s>_periodicity_major_mean` / `_minor_*` | Mean ± SEM of stain banding frequency along the nuclear axes. |
| `qc_<channel>_intensity_mean` | Mean raw intensity of the channel. |
| `qc_<channel>_pct_saturated` | % pixels at the dtype maximum. |
| `qc_<channel>_pct_in_threshold_band` | % pixels inside the segmentation threshold band (threshold method only). |

In **3-D** samples the `area`/`um2` tokens above become `volume`/`um3`
(e.g. `stain_<s>_volume_cell_um3_mean`).

**Particle mode (`All_particles`)** — one row per stain particle:

| Column | Meaning |
|--------|---------|
| `cell_id` | Running particle id (unique across the sheet). |
| `stain` | The stain channel the particle belongs to. |
| `label_id` | The particle's instance id within that stain's mask. |
| `area_um2` / `volume_um3`, `perimeter_um` / `surface_area_um2`, `circularity` / `sphericity`, `solidity`, `eccentricity` / `elongation` / `flatness`, `convexity_defects` | Particle morphology (2-D / 3-D). |
| `intensity_mean`, `intensity_sd`, `texture_*` | Intensity + GLCM texture on the stain channel. |
| `<channel>_mean` | Mean intensity of each *other* channel within the particle. |

The particle `Sample_level` sheet reports, per stain: `stain_<s>_particle_count`,
`stain_<s>_area_total_um2`, `stain_<s>_area_mean_um2` (+ `_sem`),
`stain_<s>_coverage_fraction_image`, and `stain_<s>_alignment` (3-D: `volume`/`um3`).

---

## Batch processing & aggregation

`Run Batch` walks the input root, processes every folder that contains the
**primary channel** (nuclear, or the first stain in particle mode), then builds an
`Aggregated_summary.xlsx` at **each** folder level that has downstream samples.
Aggregated sheets:

- `All_cells` / `Filtered_cells` / `Outlier_removed` (or `All_particles` /
  `Outlier_removed` in particle mode) — cells/particles pooled across samples (with
  a `sample_name` column and a `MEAN` row).
- `Sample_level` / `Filtered_sample_level` / `Outlier_removed_sample_level` — one
  row per sample, plus `MEAN` and `SEM` rows computed **across samples** (the
  statistically correct level).

---

## Batch QC report

`Run Batch` **automatically** generates a single QC PDF (`Batch_QC_report.pdf`)
at the batch root once processing finishes. Generation is best-effort: if it
fails (e.g. `matplotlib` is missing) the analysis outputs are unaffected and a
warning is printed. You can also (re)generate it manually:

```bash
python qc_report.py <batch_root> [-o report.pdf]
```

`qc_report.py` is independent of the analysis modules (it does **not** load
Cellpose/torch). It reads each sample's `Feature_summary.xlsx`, the per-sample
`Analysis/qc_meta.json` sidecar (segmentation counts + config snapshot written
during the batch), and the raw channel TIFFs, and emits **`Batch_QC_report.pdf`**
covering: run summary (configuration incl. the Cellpose model & parameters used,
the set of image dimensions present, and the environment); per-channel
acquisition QC (mean intensity, % saturated, % in threshold band, overlaid
intensity histograms); segmentation QC (cell counts, nuclear
area/circularity/solidity distributions, doublet/split counts, filtered fraction,
cytoplasm-mapping completeness, alignment); **depth (z) QC** for 3-D samples
(per-plane channel intensity, stain coverage and nuclear density vs depth — the
whole-mount attenuation/penetration checks — plus the % of nuclei clipped by the
first/last plane); **outlier-removal QC** (cells removed per sample, the removal rate
with samples that are robust-z outliers of the batch rate highlighted, and a heatmap
of how each channel voted — channels overlap, so the rows need not sum to the
removals); and
per-marker QC (coverage, area and particle count per cell). It
presents the numbers for **you** to judge — the only automated cue is that light
batch-relative highlight on the outlier-rate panel. Requires `matplotlib`. (The
report is tuned to the cell-mode sheets; particle-mode batches produce a reduced
report — chiefly the acquisition-QC pages.)

---

## Scientific notes & caveats

- **Fixed global thresholds.** Threshold-based segmentation uses one cut-off
  across the whole batch. If acquisition is not perfectly standardised, stain
  coverage can drift for technical (not biological) reasons. Use the
  `qc_<channel>_pct_in_threshold_band` and `pct_saturated` QC columns to check
  cross-sample comparability before interpreting differences.
- **Statistics.** Use the **sample-level** SEM, not the pooled cell counts —
  cells within a sample are not independent replicates.
- **Poly-nucleated cells share cytoplasm.** With the **Cellpose** cytoplasm method a
  single whole-cell region can contain two (or more) nuclei. Each nucleus is still one
  row, but they all map to the **same** cytoplasm, so their `C_*` and stain `_cyto_*`
  values are **identical (double-counted)** — the cytoplasm is not split between them.
  The `nuclei_in_cell` column and the `mononucleated_cell_count` / `polynucleated_cell_count`
  sample-level metrics flag this so such cells can be identified or filtered. The
  nuclei-seeded watershed cytoplasm method never produces this (one nucleus per territory).
- **The analysis filter** keeps a cell whose whole-cell stain area reaches the
  **mean area of one particle in that image** (computed per image, per stain, over
  all its particles — recorded as `filter_threshold_per_stain` in `qc_meta.json`).
  Because the threshold is data-derived, it adapts to each image's particle size
  rather than a fixed pixel count. These answer different questions and need not agree.
- **Outlier removal is a QC filter, not a statistical test.** The `Outlier_removed`
  tier drops a cell when **any channel** flags it: the nucleus or cytoplasm is a
  two-sided robust-z outlier on `log10(size)`, or the cell owns ≥1 stain particle that
  is, within that channel's own size distribution — all at `outlier_mad_threshold`
  (default 4). It targets segmentation artifacts
  (debris, doublets, mis-segmentations, speckle/blobs) in any channel; it
  deliberately excludes intensity, per-channel means and texture so genuinely
  high-expressing cells are retained. Because every particle's shape is tested,
  this is more sensitive than a size-only rule — raise the threshold if it removes
  too aggressively. Report which tier you analysed.
- **Circularity** is computed from the digital perimeter and is systematically
  below 1 even for perfect disks — compare relatively, not absolutely.
- **Cellpose diameter / doublet splitting:** for `cpsam`, leave Diameter at `0`.
  That means *no rescaling*, not auto-estimation — Cellpose 4.x dropped the size
  model, and `cpsam` is scale-invariant, so `0` is the right setting. A non-zero
  value rescales the image by `30/diameter` before segmentation. The
  threshold-mode `watershed_min_distance` default (80 px) may be too large to
  split typical touching nuclei — tune to your nucleus size. In 3-D the
  doublet/watershed split is disabled (Cellpose separates touching nuclei).
- **Objects vanishing at low magnification:** check **Max Size Fraction**. A cell
  covering >40 % of the frame is dropped by Cellpose at the default `0.4`.

---

## Configuration reference

Defaults live in `io_manager.DEFAULT_CONFIG` and `io_manager.DEFAULT_STAIN_CONFIG`.

**Nuclear / global**

| Key | Default | Notes |
|-----|---------|-------|
| `num_threads` | `None` | CPU worker threads for the parallel feature-extraction / stain-quantification loops. `None` → auto = detected logical cores − 1; `1` → sequential. Cellpose is unaffected (GPU / its own threads). The GUI no longer exposes it (always emits `None`) and shows the resolved count in the status bar; still settable programmatically. |
| *(GLCM texture)* | *always on* | Per-object GLCM (Haralick) texture is **always** computed (no longer configurable — the toggle was removed). It is the slowest per-object feature (~14× the rest) and is unreliable on small objects, but the `N_texture_*` / `C_texture_*` / `texture_*` columns are always populated. The only exception is **membrane-derived cytoplasm**, where `C_texture_*` is skipped (boundary signal makes it meaningless). |
| `pixel_size_um` | `0.0613414` | Lateral calibration (µm/px). |
| `z_size_um` | `None` | 3-D voxel depth (µm/plane). `None` → read from the TIFF metadata, else falls back to `pixel_size_um`. Ignored for 2-D. |
| `input_mode` | `'auto'` | `'folder_per_sample'`, `'file_per_sample'`, or `'auto'`. |
| `nuclear_method` | `'threshold'` | `'threshold'`, `'cellpose'`, `'presegmented'`, or `'none'` (no nuclear segmentation → particle mode). |
| `nuclear_cellpose` | `{…}` | Cellpose params used when method = cellpose. |
| `threshold_min` / `threshold_max` | `18` / `255` | Intensity band (raw values). |
| `min_object_size` | `2500` | Minimum nucleus area (px). The GUI no longer applies a nuclear size filter (emits `0`); still settable programmatically. |
| `edge_exclusion_margin` | `5` | Drop objects (nuclei **and** stain particles) within N px of the lateral border. Global control in the *Analysis Filter* panel. |
| `connectivity` | `2` | Labeling connectivity (threshold mode). |
| `doublet_threshold` | `0.7` | Doublet score cut-off (threshold mode). |
| `watershed_min_distance` | `80` | Min peak separation for doublet splitting (px). |
| `cytoplasm_source` | `'none'` | `'none'`, `'membrane'`, or `'channel'`. |
| `cytoplasm_method` | `'cellpose'` | `'cellpose'`, `'threshold'` (nuclei-seeded watershed), `'presegmented'`, or `'none'`. |
| `cytoplasm_threshold` | `{…}` | Threshold-method params (`threshold_min/max`, `min_object_size`). |
| `cellpose_*` | — | Cytoplasm Cellpose parameters (`model_type`, `diameter`, `cellprob_threshold`, `flow_threshold`, `niter`, `min_size`, `max_size_fraction`, `invert`). |
| `filter_stain` / `filter_condition` | `None` / `'contains'` | Analysis filter. |
| `remove_outliers` | `True` | Emit the `Outlier_removed` tier (drop cells any channel flags, on the filtered set). |
| `outlier_mad_threshold` | `4.0` | Two-sided robust-z cut-off on `log10(size)`, applied per channel. Not comparable to the old raw-size threshold of 5. |
| `use_presegmented_nuclear` / `_cytoplasm` (+ `_stem`) | `False` | Load label images instead of segmenting. In the GUI these are set by choosing the *Pre-segmented* method (nuclear / cytoplasm tab). |

**Per-stain (`DEFAULT_STAIN_CONFIG`)**

| Key | Default | Notes |
|-----|---------|-------|
| `method` | `'threshold'` | `'threshold'`, `'cellpose'`, or `'none'` (measured channel only — no stain columns). |
| `cellpose` | `{…}` | Cellpose params for this stain. |
| `threshold_min` / `threshold_max` | `18` / `255` | Intensity band. |
| `min_object_size` | `100` | Minimum particle area (px) at **segmentation** (particles below this are dropped). No longer the filter cut-off — the analysis filter now uses the per-image mean particle area. The GUI no longer exposes it (emits `0`; segmentation size is controlled by the Cellpose *Min Size*); still settable programmatically. |
| `output_type` | `'binary'` | `'binary'` or `'labeled'` (threshold mode). |
| `doublet_threshold` / `watershed_min_distance` | `0.7` / `80` | Used when `output_type = 'labeled'`. |

---

## License

Released under the [MIT License](LICENSE) — © 2026 Matthias Blanc.

This tool builds on [Cellpose](https://github.com/MouseLand/cellpose),
[scikit-image](https://scikit-image.org/), [SciPy](https://scipy.org/),
[PyQt6](https://www.riverbankcomputing.com/software/pyqt/), and
[pandas](https://pandas.pydata.org/); please cite them where appropriate.
