# Finish the interrupted loss-matched run

Work on `codex/implement-reproducibility-and-convolution-ablations`.
The immediate task is to finish `complex-loss-matched-64`, seed 33. Continue the
REEPE component experiments and remaining Computer Optics code work afterwards.

If training is already running on CUDA and has resumed at epoch 22 or later,
leave it running. An existing process does not need a branch switch to finish.
Apply repository updates when no training process is using this checkout.

## Why the remaining run is long

The committed metadata records 21 completed epochs out of 40. A recovery from
that checkpoint still performs 19 epochs, each with 1,010 batches (2,019 triplets,
batch size 2). Each batch processes RGB channels separately and performs pyramid
decomposition, network prediction, reconstruction, and backpropagation.

Autograd anomaly detection is enabled, the complex convolution executes four
real convolution calls, and the loop transfers debug predictions to CPU every
batch. These are sources of overhead; their individual contributions have not
been profiled. This recovery change preserves the existing training computation
and strict deterministic settings.

The two previous loss-matched jobs started about 38.5 and 39.0 hours apart. If
they ran consecutively with little delay, those intervals suggest about 18.3–18.5
hours for 19 epochs at the same throughput. These are estimates from start times,
not measured durations or the live remaining time. The progress bar's remaining
time describes the current epoch, not all remaining epochs.

## Update and recover

From the repository root, with the existing training environment active:

```bash
git fetch origin
git switch codex/implement-reproducibility-and-convolution-ablations
git pull --ff-only origin codex/implement-reproducibility-and-convolution-ablations

python -m tools.run_experiments \
  --plan experiments/computer_optics_revision.json \
  --configs complex-loss-matched-64 --seeds 33 \
  --run-root runs_seeded
```

The launcher selects one configuration and one seed. It automatically supplies
`--resume` when the run's full checkpoint exists. The training script prints
`RESUME`, the number of completed/remaining epochs, and the restored global step.
The target is 40 total epochs, not 40 more. Confirm that startup says `Using
device: cuda:0` (or another intended CUDA device). An unintended CPU run needs
environment diagnosis; there is no speed estimate for it in this note.

The required file is
`runs_seeded/complex-loss-matched-64/seed-33/checkpoint_last.pth`. It was absent
from the audited GitHub tree but may exist on the training laptop. If it is
missing, the launcher refuses to overwrite the interrupted run. Look for the
full checkpoint on that laptop before choosing a fresh run. A fresh run in a new
run root requires all 40 epochs; renaming a bare weights file does not restore
optimizer, scheduler, and random-generator state.

Do not reduce the epoch target or switch precision/deterministic settings merely
to obtain an earlier completion while reporting the result as the original
matched training protocol. Runtime optimizations can be checked separately after
this run. No other training job is launched by the command above.
