# REEPE component paper after the journal evaluations

## Status checked on 19 October 2026

The committed journal result files contain 84 completed baseline evaluations on
`feature/complex-valued-network` (commit `e1a80a2`) and 84 corrected evaluations
on `computer-optics/reviewer-code-corrections` (commit `688d1e1`). Both select
four configurations, seeds 11/22/33, and seven dataset groups. Their counts are
DAVIS validation 1,336, UCF101 379, Middlebury 11, and SNU-FILM 310 triplets in
each of four subsets. These counts describe this dataset installation.

Those quality results belong to the Computer Optics revision. The REEPE branch
now contains all twelve completed GPU cost profiles in commit `fb0a58f`, measured
at 256 x 256, batch size one and FP32 on the RTX 3050 Laptop GPU. No profile
reports an out-of-memory failure. Ten-epoch ablation quality results are pending.
The workflow below produces its separate evidence. It does not repeat the
40-epoch training or copy journal quality rows into the conference paper.

## Run from the existing CUDA environment

```bash
git fetch origin
git switch Component_Analysis_Computational_Cost
git pull --ff-only origin Component_Analysis_Computational_Cost
source .venv/bin/activate
```

Use the repository root as the working directory. The training JSON retains the
existing DAVIS path. Pass `--dataset-path DAVIS-data/DAVIS/JPEGImages/480p` to
the training launcher if the absolute path differs on this checkout.

### 1. Existing-checkpoint profiling is complete

The twelve committed profiles provide the first REEPE cost table. No rerun is
needed for the current protocol. These are the commands used for this stage:

```bash
python -m tools.profile_reepe_models --dry-run
python -m tools.profile_reepe_models --device cuda:0
```

Keep the power source and laptop power profile fixed. Run profiling without a
concurrent training or evaluation process. The plan uses 256 x 256 RGB, batch
size one, FP32, 30 warmups and 100 timing repetitions. Outputs are in
`results_reepe/cost_40ep/`, with twelve detailed records in `profiles.json` and
one row per checkpoint in `profiles.csv`. CPU debug profiles do not substitute
for GPU measurements.

The reported FLOPs and MACs cover convolution operations only. They exclude
pyramid FFTs, normalization, activations, interpolation and elementwise work.
Use the headings **convolution GFLOPs** and **convolution GMACs**. Full-pipeline
latency includes those operations; GPU memory is PyTorch allocator memory.

### 2. Train the independent ten-epoch component study

```bash
python -m tools.run_experiments \
  --plan experiments/reepe_10epoch_ablations.json --dry-run

python -m tools.run_experiments \
  --plan experiments/reepe_10epoch_ablations.json
```

This is **5 training runs**: five configurations, each trained once. Seed 101
is fixed inside the plan for repeatable initialization and sampling, so no seed
argument is needed. The complete reference is also trained from scratch for ten
epochs. The four changes are unrestricted real convolution, diagonal covariance
normalization, modReLU feature activation and removal of learned phase correction.
Their exact scope is documented in `REEPE_COMPONENT_STUDY.md`.

The original three-seed design is preserved as
`experiments/reepe_10epoch_ablations_3seeds.json`. It selects seeds 101, 202 and
303 and the same run root. If used later, pass that plan to evaluation as well.
Completed matching seed-101 runs will be checked and skipped by the launcher.

The launcher runs jobs sequentially and resumes incomplete jobs from the last
completed epoch. It checks and skips completed matching runs. Keep the same run
root and plan when resuming. Each run writes its checkpoint, metadata,
`training.log` and `epoch_metrics.jsonl` under `runs_reepe_10ep/`.

Ten epochs define an equal-epoch comparison, not equal runtime or proven
convergence. Use the actual logged epoch durations to estimate remaining time.
Do not use the journal's 40-epoch loss-matched model as this study's quality
reference.

### 3. Evaluate the five new checkpoints

The main paper needs DAVIS validation and the four SNU-FILM difficulty subsets.
This gives 25 model/subset evaluations. UCF101 and Middlebury remain
available through the evaluator but are not needed to start this focused study.

```bash
python -m tools.evaluate_reepe_ablations \
  --dataset davis --dataset-path DAVIS-data/DAVIS/JPEGImages/480p \
  --device cuda:0

python -m tools.evaluate_reepe_ablations \
  --dataset snufilm --dataset-path SNU-FILM \
  --snu-mode all --image-size 256 --device cuda:0
```

Results go to `results_reepe/quality_10ep/`. Each dataset group contains
`per_seed.csv`, `aggregates.csv`, `aggregates.json`, and the per-checkpoint
sample/sequence results. The default evaluator reads the updated training plan
and expects only seed 101. Each aggregate mean is the single run's dataset
score; its sample standard deviation is null because only one training run is
available. Do not report this as zero training variation. The same command can
evaluate completed variants with `--configs`.

### 4. Optional direct quality-versus-cost comparison of the ablations

The twelve existing checkpoints describe the cost of the original architectures.
To relate a component change to its own quality and cost, profile its new
checkpoint as well. This plan uses the existing profiler and starts no training:

```bash
python -m tools.profile_reepe_models \
  --plan experiments/reepe_profile_5ablations.json --dry-run

python -m tools.profile_reepe_models \
  --plan experiments/reepe_profile_5ablations.json --device cuda:0
```

Outputs are separate under `results_reepe/cost_10ep/`. The phase-off model still
executes the network; it is not a pruned speed baseline. Unrestricted convolution
has more parameters, so its comparison tests the complex weight constraint and
must not be called parameter-matched.

The existing `reepe_profile_15ablations.json` is retained for the optional
three-seed study; it is not needed for the current five-checkpoint comparison.

## Material for the article

| Material | Source | Use |
|---|---|---|
| Five-variant quality table | Ten-epoch DAVIS evaluation, one run per variant | PSNR, SSIM, LPIPS and internal PCE |
| External motion comparison | Ten-epoch SNU-FILM evaluation | Keep Easy/Medium/Hard/Extreme separate |
| Training curves | `epoch_metrics.jsonl` from the five runs | Compare common loss components and final learning slopes |
| Original-model cost table | Twelve 40-epoch checkpoint profiles | Four architecture rows with all three checkpoint records retained |
| Component cost comparison | Optional five ten-epoch profiles | Relate each ablation to its own quality and cost |
| Visual examples | New ablation predictions on common triplets | Same crops and display range; include a failure case |

Do not pool the 100 timing repetitions as 100 independent trained models.
Report the timing distribution within a profile separately from variation across
the three checkpoint measurements. Use sequence-level pairing for uncertainty
on DAVIS. Such an interval concerns the evaluated sequences; it cannot measure
variation between training runs. External directories are proxies until original-video identities are
verified. The evaluator exports the required sample and sequence data; a paired
confidence interval still needs to be computed from these new results.

The manuscript should distinguish what was tested from what remains uncertain:
ten epochs, one training run per variant, one GPU, no component interactions, and larger parameter
count for unrestricted convolution. Do not infer that complex arithmetic is
universally better, or that a lower internal PCE proves temporal consistency.

## Supplied REEPE requirements

The supplied requirements specify 5–6 IEEE-template pages; a title of at most
15 words; a 75–150-word abstract with results and conclusions; 3–7 keywords;
introduction, problem statement, theory, results/discussion and conclusion;
a conclusion of at least 100 words; and 20–30 references in first-citation order.
No bracketed citation group may contain more than three sources; at most three
self-citations are allowed. There may be no more than six authors and no more
than three papers by one author in the issue. The stated similarity threshold
is 20%; that is a submission rule, not a guarantee from this workflow.

The author information requires affiliation, full postal address and email, with
academic degree/title as applicable. The template specifies A4, two-column body
text, numbered section styles, captions above tables and below figures, and no
page numbering. Its example copyright line contains a 2026 identifier; the final
REEPE-2027 line must come from the organizer and must not be guessed.

The earlier working article draft describes the original three-seed protocol.
Its methods must now state one run per configuration with fixed seed 101, and
remove across-training-run standard deviations and claims about consistency
across seeds. The completed twelve-checkpoint cost profiles still cover their
original seeds 11, 22 and 33. The final abstract, results/discussion and conclusion
should use these profiles and the new single-run ablation results when available.
Keep room for results and figures when completing the 5–6-page manuscript.

## Files to retain after running

Keep all of `results_reepe/` and, for every new run, `metadata.json`,
`epoch_metrics.jsonl`, `training.log` and the checkpoint files. Commit the result
and metadata files to this branch after each completed stage. Large checkpoint
files can remain on the research machine if their hashes and accessible backups
are retained; the paper tables require the result records, not a new training run.
