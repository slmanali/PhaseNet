# REEPE: component analysis and computational cost

This branch implements two experiments for the proposed REEPE paper. The first
asks how individual operations affect quality after a fixed training budget. The
second measures the computational cost of the twelve existing trained models.
The twelve GPU cost profiles were committed on 19 October 2026. The component
training plan now runs each configuration once; its quality results are pending.

## Keep the papers separate

| Collection | Models | Results | Destination |
|---|---|---|---|
| Computer Optics revision | Existing four configurations, seeds 11, 22, 33, 40 epochs | Journal quality comparisons and reviewer responses | Existing journal workflow |
| REEPE component experiment | Five fresh configurations, one run each, seed 101, 10 epochs | PSNR, SSIM, LPIPS, internal PCE | `runs_reepe_10ep/`, `results_reepe/quality_10ep/` |
| REEPE cost experiment | The twelve existing 40-epoch checkpoints | Parameters, convolution MACs/FLOPs, latency, allocator memory | `results_reepe/cost_40ep/` |

The REEPE quality reference is also trained from scratch for ten epochs. Do not
copy a journal quality row into the REEPE table or use a 40-epoch checkpoint as
the reference for a ten-epoch ablation. Sharing the dataset and method description
does not make the numerical results identical; disclose the relationship between
the papers and cite the journal work when available. State that the cost experiment
reuses trained checkpoints. If the same cost table is added to the journal, it will
no longer satisfy the intended separation of numerical results.

## Component experiment

The plan is `experiments/reepe_10epoch_ablations.json`: **5 runs in total**.
It fixes seed 101 internally; no `--seeds` argument is needed. A fixed seed
controls initialization and sampling without adding training runs. The original
three-seed plan remains in `experiments/reepe_10epoch_ablations_3seeds.json`.
It uses the same run directories so completed seed-101 runs can be retained if
the additional seeds are trained later. The model settings have not changed.
All use feature width 64, the existing `complex-loss-matched-64` loss weights,
DAVIS training sequences, 256 x 256 inputs, batch size 2, Adam, and the same
ReduceLROnPlateau rule. The scheduler responds to each run's training loss, so its
actual learning-rate trajectory can differ between variants. It is logged.
Evaluation uses the final completed tenth epoch, without best-test selection.

| Configuration | Single change from the reference | Interpretation |
|---|---|---|
| `reepe10-reference` | None | Complete loss-matched complex model |
| `reepe10-conv-unrestricted` | Replace complex convolutions with unrestricted real block convolutions | Tests the complex weight-sharing constraint; parameter count increases |
| `reepe10-bn-diagonal` | Set the covariance cross term to zero during whitening | Tests covariance mixing; centering and complex affine parameters remain |
| `reepe10-activation-modrelu` | Replace feature LeakyReLU with modReLU | Tests a phase-preserving feature nonlinearity; adds one bias per feature |
| `reepe10-phase-off` | Set learned angular correction scale to zero | Tests the contribution of the learned phase correction |

Convolution replacement includes feature blocks and prediction heads. Its real
block matrix starts as `[[A, -B], [B, A]]`, matching the complex convolution at
initialization; training can then break those ties. This is not an equal-parameter
comparison. The separate-real convolution already supported by the repository is
an optional control and is not included in these five runs.

The diagonal normalization retains the reference implementation's numerical
stabilization and affine transformation. It does not replace the entire block
with two ordinary BatchNorm layers. modReLU replaces only feature activations;
prediction-head componentwise tanh remains unchanged. Its zero-initialized bias
does not make its initial function equal to LeakyReLU.

For each seed, compatible weights and buffers are copied from an identically
initialized reference; the data-loader generator is independent of model size.
The phase-off variant retains the network computation. It removes the learned
angular correction, not the circular midpoint representation, and it is not a
pruned model for demonstrating a speed improvement. Oriented prediction branches
can consequently lose their useful gradient. Report this as a removal test.

Run from the repository root in the existing CUDA-enabled environment:

```bash
git fetch origin
git switch Component_Analysis_Computational_Cost
git pull --ff-only origin Component_Analysis_Computational_Cost

# Inspect all five commands without training.
python -m tools.run_experiments --plan experiments/reepe_10epoch_ablations.json --dry-run

# Train sequentially; completed runs are checked and skipped on a later invocation.
python -m tools.run_experiments --plan experiments/reepe_10epoch_ablations.json
```

The plan has the training-machine DAVIS path. Override it with `--dataset-path`
if the checkout is elsewhere. To begin with one run, append
`--configs reepe10-reference`. Repeating the same command resumes an
interrupted run from the last complete epoch. It does not resume from the middle
of an epoch. A missing checkpoint stops the launcher rather than overwriting the
run. Each run saves `metadata.json`, `training.log`, `epoch_metrics.jsonl`, a full
resumable checkpoint, and final weights. Keep the metadata beside the weights.

Evaluate the new checkpoints on DAVIS validation:

```bash
python -m tools.evaluate_reepe_ablations \
  --dataset davis \
  --dataset-path DAVIS-data/DAVIS/JPEGImages/480p \
  --device cuda:0
```

For an external motion-stratified check:

```bash
python -m tools.evaluate_reepe_ablations \
  --dataset snufilm --dataset-path SNU-FILM \
  --snu-mode all --image-size 256 --device cuda:0
```

Use `--configs` to evaluate completed variants while other runs are pending.
The default evaluator reads the same one-seed plan, so it does not expect seeds
202 or 303. For the optional repeated study, pass
`--plan experiments/reepe_10epoch_ablations_3seeds.json` to both training and
evaluation; `--seeds` can then select its completed subsets. The evaluator also supports
`--dataset ucf101` and `--dataset middlebury`; native resolution is optional for
external datasets and produces a separate output directory. Do not mix resized
and native-resolution rows. The default study uses resized 256 x 256 inputs.
LPIPS may download its pretrained AlexNet weights on the first evaluation.

Outputs include per-triplet `samples.csv`, per-sequence `sequences.csv`, run
`summary.json`, and `aggregates.csv/json`. With one run, each aggregate mean is
that run's dataset score, and sample standard deviation is **null**, not zero.
Report the score as a single-run result without a training-run error bar.
If the optional three-seed study is completed, the aggregate is the mean and
sample standard deviation of per-run dataset means, not the standard deviation
across frames. Per-triplet rows contain the seed, frame identifiers,
checkpoint SHA-256, dimensions and protocol identifier. The manifest hash covers
ordered identifiers; it is not a hash of the image bytes. DAVIS grouping uses
sequence directories; external frame directories are only proxies until their
mapping to original source videos is verified. Do not treat adjacent triplets
as independent observations in confidence intervals.

Evaluation restores normalization, activation, convolution and correction scales
from saved configuration. This matters because a phase scale is not contained in
a bare tensor state dictionary. The evaluator rejects journal checkpoints and
checkpoints whose saved configuration differs from the frozen ten-epoch plan.

## Computational measurements of the twelve saved models

`experiments/reepe_profile_12models.json` selects `phasenet-64`, `phasenet-93`,
`complex-full-64`, and `complex-loss-matched-64`, with seeds 11, 22 and 33.
It requires completed 40-epoch metadata and saved weights. It never starts training.

```bash
python -m tools.profile_reepe_models --dry-run
python -m tools.profile_reepe_models --device cuda:0
```

Each checkpoint runs in its own subprocess, sequentially. The default measurement
is one RGB midpoint, 256 x 256, FP32, no tiling, no compilation, TF32 disabled,
strict deterministic algorithms, 30 warmup calls and 100 timed calls. Keep the
laptop on the same power profile and power source and avoid other GPU workloads.
These settings characterize this implementation; they are not a claim about the
fastest achievable deployment. Synthetic endpoints are used consistently for
cost measurement; no image-quality scores are computed by this tool.

| Measurement | Definition |
|---|---|
| Parameters | Stored trainable and total scalar parameters |
| `conv_macs` | Actual real Conv2d multiply-accumulate calls, including reused kernels, all RGB channels and tile halos when enabled |
| `conv_flops` | Twice `conv_macs`, with one multiplication plus one addition per MAC |
| Network latency | Cached endpoint coefficients to predicted coefficients; CUDA-event timing |
| Complete interpolation latency | Endpoint pyramid construction, input preparation, network, reconstruction and clamping; synchronized wall-clock timing |
| GPU memory | PyTorch peak allocated and reserved memory in MiB; absolute and incremental allocated peaks are both recorded |

**The arithmetic values are convolution subtotals, not total network or pipeline
FLOPs.** FFTs, normalization, activation, interpolation, trigonometric and other
elementwise operations, and convolution bias additions are excluded. Do not label
the paper's columns simply "total FLOPs". Network and complete pipeline execute
the same network convolutions; separate duplicate convolution-count columns would
not measure pyramid arithmetic.

Latency includes neither model loading nor disk reads, transfers from the host,
or quality metrics. Both scopes retain model and input allocations. Network scope
also retains prepared coefficients; those caches are released before complete
pipeline measurement. GPU memory refers to the PyTorch allocator, not all memory
shown by `nvidia-smi`. Warmup precedes peak reset. Raw timing samples, device/software
information and checkpoint hashes are saved with every profile. CPU debug runs
report GPU memory as null and are marked unsuitable as GPU measurements.

See `results_reepe/cost_40ep/profiles.csv` for twelve rows and `profiles.json` for
the detailed records. OOM and other failures are explicit; no failed measurement
is converted to zero. Repeating the command skips matching completed measurements.
Use a new `--output-root` for another GPU, session, resolution or timing protocol.
Use `--overwrite` only when intentionally replacing a measurement. Do not pool
timing repetitions as independent trained models. The three seeds share an
architecture, so their parameter counts and convolution counts should agree.

To relate ablation quality directly to its cost, profile the five new checkpoints:

```bash
python -m tools.profile_reepe_models \
  --plan experiments/reepe_profile_5ablations.json --device cuda:0
```

The optional `reepe_profile_15ablations.json` plan is for the three-seed study.
The same single-checkpoint profiler can also measure an individual checkpoint:

```bash
python -m tools.benchmark_interpolation \
  --checkpoint runs_reepe_10ep/reepe10-reference/seed-101/checkpoint_last.pth \
  --expected-epochs 10 --study-id reepe_component_cost_10ep_v1 \
  --device cuda:0 \
  --output results_reepe/cost_10ep/reepe10-reference-seed101.json
```

## Evaluation corrections in this branch

The real-model PCE previously divided three channel errors by two; it now uses
their RGB mean. The real model's amplitude threshold is applied after restoring
raw pyramid amplitudes, matching the units used by the complex evaluator. Existing
PCE tables made with the old real evaluator must be regenerated before comparison;
do not silently combine old and corrected values. The complex evaluator's existing
RGB divisor was already correct.

Both evaluators require batch size one for per-image metrics. PSNR uses RGB MSE
on [0,1] images; SSIM uses an 11 x 11 Gaussian window with sigma 1.5 and zero
padding; LPIPS uses AlexNet with RGB inputs mapped to [-1,1]. PCE averages wrapped
absolute phase differences, in radians, over oriented coefficients whose target
raw amplitude exceeds 1e-4, then over RGB channels and images. PCE measures internal
coefficient agreement, not temporal consistency of a reconstructed video. It is
not a replacement for PSNR, SSIM or LPIPS.

## Claims the experiment can support

Ten epochs can support a controlled **fixed-budget component study**. They do not
establish convergence, and component rankings can change with longer training.
Inspect the recorded learning curves before explaining a difference as an
architectural advantage. One run per configuration does not estimate variation
between training runs, even when the configurations share a fixed seed. Report
the observed differences as a single-run comparison. If three seeds are later
completed, show individual runs and their mean and sample standard deviation.
One-at-a-time ablations identify conditional
effects in this model; they do not estimate interactions among components.

The unrestricted convolution is the most informative primary comparison for
weight sharing, but has more parameters. A separate-real convolution control
would provide a complementary test with the complex convolution's parameter
count. If epoch-ten curves remain steep, a later matched longer-budget experiment
on the reference and the most informative variant would strengthen the conclusion.
Neither optional extension is part of the five-run plan above.

The paper should include a component-quality table, learning curves, and a distinct
cost table for the twelve existing checkpoints. A quality/cost conclusion about
the component variants additionally requires profiling those variants, using the
same measurement procedure. A new perceptual metric is optional; it cannot repair
an unmatched training budget or an unsupported cost claim.
