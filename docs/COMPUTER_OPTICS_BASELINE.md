# Evaluate the twelve completed models before corrections

`experiments/computer_optics_eval_12_legacy.json` evaluates the four existing
configurations with seeds 11, 22 and 33. No training is launched. The original
model, pyramid and metric files have not been changed on this branch. The plan
records their hashes and refuses to run if they differ.

The dataset selections are DAVIS **validation**, UCF101 interpolation triplets,
Middlebury folders with ground-truth frame11, and SNU-FILM Easy, Medium, Hard and
Extreme. This produces 84 model/seed/subset evaluations. All use 256 x 256 RGB
inputs, batch size one, FP32 and the existing dataset transforms. This is a
controlled reevaluation of the saved models; it reproduces an older table only
if that table used the same checkpoints, data and preprocessing.

```bash
git fetch origin
git switch feature/complex-valued-network
git pull --ff-only origin feature/complex-valued-network

python -m tools.run_saved_evaluation \
  --plan experiments/computer_optics_eval_12_legacy.json --dry-run

python -m tools.run_saved_evaluation \
  --plan experiments/computer_optics_eval_12_legacy.json --device cuda:0
```

The paths in the JSON are relative to the repository root. Override a path with,
for example, `--dataset-path davis=DAVIS-data/DAVIS/JPEGImages/480p`; repeat this
option for other datasets. Use `--run-root` if saved models are elsewhere.

To begin with DAVIS only, append `--datasets davis`. To evaluate one saved model,
append `--configs complex-loss-matched-64 --seeds 33`. The launcher is sequential;
each job writes its progress to its own `evaluation.log`. Matching completed jobs
are skipped when the same command is repeated. Failed jobs stop the launcher;
completed outputs are preserved. No partial dataset score is accepted.

Every model directory must contain completed 40-epoch `metadata.json` and
`model_final_state_dict.pth`. The exported weights are used because the unchanged
real-model loader cannot reliably open full RNG-containing checkpoints under
PyTorch 2.6. This is a file-format choice, not a weight or metric correction.

Results go to `results/computer_optics/legacy_12models/`. Each job saves a dataset
manifest, checkpoint hash, software/source information, per-triplet metrics and
its summary. Top-level `selected_runs.csv` and `aggregates.csv` contain the
selected jobs and mean/sample standard deviation across seeds. Run the full
command to collect all 84 rows; subset invocations summarize their own selection.
External frame-directory names are recorded as tentative groups. They are not
automatically treated as verified source videos for statistical inference.

The baseline deliberately preserves the old real-model PCE divisor and its
normalized-amplitude mask. Its PCE values are diagnostic legacy results, not the
corrected journal results. Changing the seed or checkpoint can change every metric;
correcting that PCE calculation changes PCE without changing the saved network.
The run does not test the ten-epoch REEPE ablations.

The reviewer corrections are prepared separately on
`computer-optics/reviewer-code-corrections`. Complete this baseline first. That
branch writes corrected results to a different directory, allowing explicit
before/after comparison without replacing these files.
