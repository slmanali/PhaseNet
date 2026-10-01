# Seeded PhaseNet experiments and profiling

These tools create new output under `runs_seeded/`; they do not modify historical
checkpoints or results. Reproducibility is scoped to the recorded hardware,
driver, PyTorch and backend settings. Strict mode raises on an unsupported
nondeterministic operation. `--deterministic warn` is an explicitly best-effort
alternative, not a strict result.

## Experiment contract

`experiments/article_seeded.json` transcribes the four article commands and adds
the two convolution-only controls. All six use 40 epochs, batch size 2, Adam
(`betas=(0.9,0.999)`, weight decay `1e-5`), ReduceLROnPlateau (factor 0.5,
patience 2), 256 px preprocessing, the sequence-disjoint DAVIS training split,
and final-completed-epoch selection. The two controls use the loss-matched
objective. The complex reference is one of the six, not a seventh duplicate.

The loader generator is independent of model initialization and advances across
epochs. Checkpoints resume only after a completed epoch and include optimizer,
scheduler and all RNG states. Existing bare state dictionaries remain accepted
for inference, but cannot resume training. Runs refuse overwrite; the launcher
resumes an incomplete run and skips a completed one. Historical checkpoints have
unknown seeds and must not be placed into the seeded aggregate.

For paired initialization, both controls first construct the complex reference.
Non-convolution state/buffers are copied. Separate-real receives the matching A/B
kernels and biases. Unrestricted-real receives `[[A,-B],[B,A]]` and matching
biases. Normalization can still couple separate-real components; only feature and
prediction-head convolution operators are replaced.

## Laptop commands

```bash
git switch feature/reproducibility-ablation-profiling
python -m pip install -r requirements.txt

# Inspect exactly 18 fresh jobs (six configurations x seeds 11,22,33).
python tools/run_experiments.py --dry-run \
  --dataset-path /home/salman/Documents/GitHub/PhaseNet/DAVIS-data/DAVIS/JPEGImages/480p

# Run sequentially on one GPU. Add 44 55 after the initial campaign if desired.
python tools/run_experiments.py --seeds 11 22 33 --deterministic strict \
  --dataset-path /home/salman/Documents/GitHub/PhaseNet/DAVIS-data/DAVIS/JPEGImages/480p \
  --run-root runs_seeded
```

Evaluate each final state dictionary with the existing evaluator and the matching
`--feature-dim`; pass `--convolution` for complex controls (the evaluator exposes
this option). Store one JSON run record per dataset/protocol containing
`variant,dataset,resolution,protocol,seed,status,psnr,ssim,lpips,pce`. Keep SNU-FILM
easy/medium/hard/extreme as separate dataset keys.

```bash
python tools/aggregate_seeded_results.py runs_seeded_metrics/*.json \
  --expected-seeds 11 22 33 --output aggregate.json
python tools/aggregate_seeded_results.py runs_seeded_metrics/*.json \
  --paired complex-loss-matched-64 complex-separate-real-64 \
  --expected-seeds 11 22 33 --output paired.json
```

A one-run standard deviation is `null`, not zero. Failed/incomplete records are
listed explicitly; resolution/protocol keys are never pooled.

## Computational measurements

The benchmark accepts historical bare state dictionaries and new full
checkpoints. It measures batch-one RGB midpoint output, never decomposes a ground
truth frame, excludes disk I/O/metrics/checkpoint loading, and never silently
changes settings on OOM. Network scope caches both endpoint pyramids and prepared
coefficients, runs the predictor three times (RGB), and excludes reconstruction.
Full scope keeps input/output on the selected device and includes the two endpoint
pyramids, preparation, prediction, conversion, inverse pyramid and clamp.

```bash
python tools/benchmark_interpolation.py \
  --checkpoint runs_seeded/complex-loss-matched-64/seed-11/checkpoint_last.pth \
  --architecture complex --convolution complex --feature-dim 64 \
  --resolution 256 --precision fp32 --tile-size 256 --device cuda:0 \
  --deterministic strict --warmup 30 --repetitions 100 \
  --output benchmarks/complex-loss-matched-seed11.json
```

JSON and CSV identify PyTorch peak allocated/reserved memory (not `nvidia-smi`
total memory), latency statistics, unique parameters, and call-observed real
Conv2d MAC/FLOP subtotals. The FLOP convention is two operations per real MAC.
FFT pyramid work, normalization, activation, interpolation, trig and elementwise
work are explicitly excluded from that subtotal rather than assigned zero cost.
CUDA events time network device work; synchronized wall time captures the full
pipeline and any CPU work. CPU runs honestly leave CUDA memory fields null.

## Focused checks

```bash
pytest -q tests/test_reproducibility_ablation_profiling.py
pytest -q tests/test_complex_phasenet_safe_baseline.py tests/test_davis.py tests/test_steerable_pyramid.py
python -m py_compile train.py train_complex_safe_baseline.py utils/reproducibility.py \
  utils/inference.py tools/run_experiments.py tools/aggregate_seeded_results.py \
  tools/benchmark_interpolation.py
```

Full data-backed training/evaluation and CUDA timing are intentionally left for
the target laptop. Endpoint-only predictions use exactly the same coefficient
conversion, predictor, and reconstruction code as evaluation; focused tests use
a tolerance of `2e-5` where FFT roundoff is involved.
