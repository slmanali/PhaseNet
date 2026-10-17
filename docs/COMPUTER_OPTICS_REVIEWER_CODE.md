# Computer Optics revision: code, evidence and remaining manuscript work

## Run order

First complete the unchanged evaluation on `feature/complex-valued-network`,
following `COMPUTER_OPTICS_BASELINE.md`. Its commit is
`013bb7c7cd81115639369f90ca8f010eab4b04ca`. It adds orchestration and a JSON plan;
the existing evaluators, model and metric implementations remain unchanged.

Then use this branch for the corrected evaluation of **the same twelve saved
40-epoch models**, with the same data and resolution:

```bash
git fetch origin
git switch computer-optics/reviewer-code-corrections
git pull --ff-only origin computer-optics/reviewer-code-corrections

python -m tools.run_saved_evaluation \
  --plan experiments/computer_optics_eval_12_corrected.json --dry-run

python -m tools.run_saved_evaluation \
  --plan experiments/computer_optics_eval_12_corrected.json --device cuda:0

python -m tools.analyze_journal_evaluation
python -m tools.compare_evaluation_protocols
python -m tools.audit_journal_runs
```

No command above trains a network. Dataset paths and `--run-root` overrides work
as in the baseline guide. Run from the repository root in the existing Python
environment. The analysis tools use NumPy and Matplotlib, already project
dependencies. If checkpoints are not under `runs_seeded`, also update `run_root`
in the JSON used by `audit_journal_runs`.

The corrected results are written to `results/computer_optics/corrected_12models`.
The baseline remains under `legacy_12models`. The comparison refuses to pair
different checkpoint hashes, sample manifests or tile sizes. Keep both sets of
outputs. A change in the saved model can change all metrics; a correction to PCE
does not by itself improve the generated image.

The three seeds are 11, 22 and 33, for `phasenet-64`, `phasenet-93`,
`complex-full-64` and `complex-loss-matched-64`. DAVIS validation, UCF101,
Middlebury and the four SNU-FILM subsets give **84 evaluations**. These are
40-epoch journal results. They do not include the separate REEPE ten-epoch
component study or its seeds 101, 202 and 303.

## What the corrections measure

The real evaluator now averages PCE over three RGB channels (previously it divided
their sum by two). Its amplitude mask now uses the original pyramid amplitude,
restored from the normalization scale. Therefore the correction is not necessarily
just multiplication by two thirds. No trained weights, architecture, forward
prediction or historical loss definition has been replaced.

Checkpoint loading reads saved model settings, rejects conflicting overrides,
loads state dictionaries strictly, and stages full trusted checkpoints on the CPU.
LPIPS is explicitly in evaluation mode; publication scoring enforces batch size
one. The two training entry points only lose an unsupported scheduler `verbose`
argument for PyTorch 2.6 compatibility. Their objectives are unchanged.

For each image the corrected evaluator additionally saves:

- internal coefficient PCE and the phase error of the circular midpoint of the
  two endpoint phases, with the same target-amplitude mask;
- PCE after the final clamped RGB prediction is decomposed again;
- valid and total coefficient counts and their fraction;
- the actual unweighted phase-unit penalty for complex outputs, using the exact
  historical loss expression, and the lowpass imaginary-component error.

`phase_diagnostics.csv` contains image-level values. `phase_by_band.csv` separates
RGB channel, scale (coarse to fine) and orientation. It pools coefficient errors
within each band; it must not be averaged unweighted to reproduce the image-level
score. `diagnostics.json` states the aggregation: valid-coefficient mean within
each channel, then RGB mean, then equal image mean. A channel with no valid
coefficients contributes zero, matching the existing metric; counts expose that
case. The amplitude threshold is `1e-4`; the historical pyramid amplitude uses
`sqrt(real**2 + imag**2 + 1e-8)`. These are measures of phase agreement with the
target frame, not evidence of temporal consistency across an output video.

The unit-phase loss is measured **after** the model's stabilized normalization,
where the original training loss is applied. This tests its numerical size; it
does not establish that the term helped training. No earlier model is described
as having used a different loss.

## Uncertainty and protocol

`analysis/seed_summary.csv` reports means and sample standard deviations across
trained seeds. `paired_training_seeds.csv` records paired differences per seed.
`paired_video_intervals.csv` reports a separate 95% percentile bootstrap interval:
whole source videos are resampled, the same video resample is used for every
trained seed, and image weighting is preserved. This interval is conditional on
the saved training runs. It does not estimate the uncertainty of a new training
run. Missing seeds are recorded; a partial run must not be called a three-seed
result.

DAVIS sequence identities are verified from the dataset structure and official
disjoint train/validation lists. External triplet directory names are **not**
automatically assumed to identify independent source videos. External intervals
are withheld until grouping is verified. To provide a verified mapping, add
`"sequence_map": "path/to/mapping.csv"` to the relevant dataset entry. The CSV
requires `sample_key,sequence_id` and exactly one row per evaluated triplet;
derive the keys from the saved manifest and identify the original source video.
Do not assign a different video ID to overlapping triplets of one video. Keep
the same mapping in all plans being compared. Changing a mapping changes the
manifest hash; use a new output directory for that protocol.

The default protocol is 256 x 256 FP32 RGB, batch size one. DAVIS uses the existing
PIL resize; external datasets use their existing tensor resize. PSNR is computed
per image at data range one; SSIM uses the shared 11 x 11 Gaussian window
(sigma 1.5), zero padding and RGB averaging; AlexNet LPIPS receives [-1, 1]
inputs. Outputs are clamped to [0, 1]. There is no evaluation border crop.
Manifests record the actual file lists, counts, dimensions and preprocessing.
Their hashes cover paths, grouping and dimensions, **not image bytes**. No
automatic resizing or dataset truncation occurs after an out-of-memory error.

## Review comments and their status

The table paraphrases the supplied reviews. "Implemented" means the code is
available, not that the required scientific result has already been obtained.

| Comment | Where to address it | Code/evidence now available | Still required |
|---|---|---|---|
| Reviewer 1: explain the practical contribution despite stronger RIFE/FILM results | Text and evaluation | Common-protocol external scorer described below | State the scope as an architecture comparison; discuss measured limitations |
| Reviewer 1: explain pyramid bands, dimensions, information flow and the narrow real model | Text | Existing model source and recorded feature widths; audit exports settings and parameter counts | Correct the architecture description and diagram |
| Reviewer 1: independently processed color channels may produce chromatic artifacts | Text and qualitative evidence | Shared triplet manifests and per-channel phase diagnostics | Inspect identical RGB crops; discuss color failure cases without inventing a measured artifact rate |
| Reviewer 1: justify loss weights and the full objective when the matched loss performs similarly | Both | Saved weight values, matched three-seed comparisons, historical loss curves, measured unit-phase term | Explain how weights were selected; a sensitivity study would require new runs |
| Reviewer 1: report optimizer details | Text | Metadata audit exports Adam, learning rate, betas, weight decay and scheduler | Add the actual settings to the methods section |
| Reviewer 1: present external results and more convincing visual examples | Both | Matched-resolution RIFE/FILM scoring; existing image-export evaluators and common manifests | Run external references, select identical regions and revise the discussion/figures |
| Reviewer 2, point 1: representation and output-head changes confound a convolution-only claim | Both | Optional 40-epoch unrestricted-real convolution control plan, retaining complex representation and head | Train/evaluate this control, or limit the claim to the complete architecture; the twelve existing runs cannot isolate convolution |
| Reviewer 2, point 2: sufficient architecture, preprocessing and training detail | Both | Per-job manifests, metadata/checkpoint audit and strict configuration-aware loading | Describe pyramid, sharing, normalization, activations, highpass, lowpass, initialization and checkpoint selection accurately |
| Reviewer 2, point 3: repeated seeds, paired uncertainty and training curves | Both | Three-seed summaries, source-video bootstrap, separate seed variability, log audit/plots | Run evaluation; verify external source-video groups; discuss incomplete convergence evidence |
| Reviewer 2, point 4: possible redundancy of the unit-phase penalty | Both | Exact post-normalization term measured without changing the historical objective | Report its observed size and remove unsupported causal claims |
| Reviewer 2, point 5: exact dataset protocol and fair external references | Both | Actual triplet lists/counts/dimensions; common float-input external scorer with repository/checkpoint provenance | Run references with the same protocol and report versions, training-data differences and all preprocessing |
| Reviewer 2, point 6: reconstructed-image PCE, mask coverage, aggregation and circular midpoint | Both | Implemented in `journal_diagnostics.py`; per-image and per-band exports | Report the new values and definitions; avoid interpreting PCE as temporal consistency |
| Reviewer 2, point 7: wording, DAVIS validation/test distinction and references | Text | DAVIS output is explicitly labelled validation | Revise the manuscript and bibliography |
| Reviewer 2, point 8: desirable runtime/memory information | Optional evaluation and text | Repaired profiler separates network/full-pipeline timing and allocator peaks | GPU measurements are pending; keep the REEPE numerical cost table separate as requested |
| Editor: citation block and automatic numbering | Manuscript source | No Python change applies | Fix the DOCX/TEX citation block and automatic equations/figure/table numbering |

### Facts to preserve in the methods section

- The code uses pyramid height 12, four orientations and a scale factor of sqrt(2):
  ten oriented levels plus highpass and lowpass. Eleven learned blocks process
  the lowpass followed by the ten oriented levels. Blocks have separate weights;
  the same model weights are reused for the three color channels.
- Complex features use full-covariance complex batch normalization and separate
  real/imaginary LeakyReLU (slope 0.2); the prediction head uses componentwise
  tanh. These changes are not isolated by comparing only parameter-matched models.
- Alpha and beta are learned real scalars initialized to 0.5, with no interval
  constraint. The audit reports their final values from each checkpoint.
- The real reconstruction inserts a zero highpass. The complex reconstruction
  uses the average endpoint highpass. This is an additional architecture/pipeline
  difference; these saved models must not be presented as a convolution-only test.
- The lowpass target has zero imaginary part. Complex lowpass features can still
  affect later levels; image reconstruction uses the real lowpass component.
- Training uses RGB channel processing, 256 x 256 resizing, no random spatial
  augmentation in these entry points, Adam (betas 0.9/0.999, weight decay 1e-5),
  initial learning rate 8e-5 and ReduceLROnPlateau (factor 0.5, patience 2) on
  training loss. The recorded checkpoint selection is the final completed epoch,
  not the best validation checkpoint. Use each run's metadata as the authority.
- Full versus loss-matched weights differ for image (0.08/0.05), phase
  (0.25/0.2), gradient (0.12/0.1), imaginary residual (0.05/0) and phase-unit
  (0.01/0) terms. A small difference between those two configurations is not a
  separate ablation of every loss term.

`audit_journal_runs` exports the twelve metadata records, checkpoint identities,
parameter counts and final alpha/beta values. Its explicit legacy log mapping is
in `experiments/computer_optics_log_map.json`. The historical filenames do not
embed a cryptographically verified run identity; retain the original files and
check the mapping when preparing the response. `--logs-only` can inspect logs
without loading checkpoints.

The exported curves are labelled smoothed, rounded **logged minibatches**, not
epoch means or validation curves. Complex logs recorded the last processed RGB
channel every ten steps. A printed `0.000` does not prove an exactly zero term.
Equal training duration and similar smoothed curves do not establish convergence.

## Optional evidence that needs additional runs

The reviewers' single convolution control is already supported by the existing
architecture. A dedicated plan keeps it separate from the REEPE ten-epoch study:

```bash
python -m tools.run_experiments \
  --plan experiments/computer_optics_representation_control.json \
  --run-root runs_journal_controls --seeds 11 22 33 --dry-run
```

This is only a preview. Remove `--dry-run` only when choosing to perform the extra
training. It uses 40 epochs and the loss-matched settings; all convolutional
operators, including prediction-head convolutions, become unrestricted real
maps, initialized from the complex block matrices. Complex normalization,
activation, phase representation and phase handling remain fixed. The real map
has more free parameters; report that explicitly. It is not a fully real network.
Evaluate its completed checkpoints with the same corrected protocol before using
it as evidence. Add its configuration and checkpoint root to a separate copy of
the evaluation plan; do not silently mix it into the twelve-run audit.

External references can be rescored without PNG rounding of resized endpoints:

```bash
python -m tools.evaluate_journal_external \
  --model rife --repo /path/to/ECCV2022-RIFE --checkpoint /path/to/rife/train_log \
  --dataset-path SNU-FILM --image-size 256 --device cuda:0

python -m tools.evaluate_journal_external \
  --model film --repo /path/to/frame-interpolation --checkpoint /path/to/film/saved_model \
  --dataset-path SNU-FILM --image-size 256 --device cpu
```

These paths are placeholders for the existing upstream installations and weights.
The wrappers use the project's existing backend classes. RIFE pads to a multiple
of 32 and crops back; FILM retains its upstream alignment/block handling. The
scorer records the upstream commit, dirty state and checkpoint file hashes. Its
scores use the same SNU-FILM float tensors and shared PSNR/SSIM/LPIPS functions.
It does not retrain either reference or make their training sets comparable to
DAVIS-only training. Native-resolution scores must be kept in a separate table.

The profiler remains available for the optional runtime comment. It counts only
the actual Conv2d MAC/FLOP subtotal (two FLOPs per MAC), not all FFT, normalization,
activation or pointwise operations. Timings include the full pipeline separately
from the network. GPU memory means PyTorch allocator memory, not total system or
driver memory. No cost measurements are launched by the journal quality plan.

## Verification and limits

Focused CPU tests cover the 84-job plan, checkpoint completion checks, scalar
aggregation, real/complex synthetic evaluation with phase diagnostics, mask and
phase wrapping, paired video bootstrap, float input preservation in external
adapters, and before/after checkpoint matching. Synthetic tests substitute a
small LPIPS stub to avoid weight downloads; they do not validate perceptual
quality. Historical log export has been exercised on all twelve committed logs.
Full dataset evaluation, real upstream RIFE/FILM inference and CUDA measurements
remain to be run on the research machine. No benchmark scores are supplied here.
