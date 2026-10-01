# PhaseNet: reproducibility, convolution ablations, and computational measurements

Implement this task in `slmanali/PhaseNet`, starting from `feature/complex-valued-network`. The branch inspected for this specification was at commit `bbc40ef2d5c5bc03c94ee616ea6ccc748721a678`. Inspect the current checkout and any repository instructions before editing, and account for changes since that commit. Work on a new branch and preserve local work and existing checkpoints.

The goal is to support defensible experiments for the article “Комплекснозначная нейронная сеть для фазовой интерполяции видеокадров.” I will run full training on my NVIDIA GeForce RTX 3050 Laptop GPU with 6 GB memory. Implement the changes and run focused tests and short smoke checks. Do not launch the full training campaign or claim that the modifications improve reconstruction quality before measurements exist.

## 1. Inspect and preserve the existing experiment

Start with these files:

- `training_testing_commands.txt`: commands for the four article models and their loss weights.
- `train.py` and `train_complex_safe_baseline.py`: the active training entry points in those commands.
- `net/phasenet.py`, `net/complex_phasenet_safe_baseline.py`, and `net/complex_nn/complex_layers.py`: actual architectures and complex operations.
- `test.py`, `test_complex_safe_baseline.py`, `utils/metrics.py`, and the dataset utilities: evaluation and result export.
- `steerable/SCFpyr_PyTorch.py`: pyramid decomposition and reconstruction.

Preserve the four existing configurations: PhaseNet-64, PhaseNet-93, ComplexPhaseNet-full-64, and ComplexPhaseNet-loss-matched-64. Create explicit experiment configurations from the saved commands; do not substitute CLI defaults, which differ from those commands. Retain 40 epochs, batch size 2, initial learning rate 8e-5, the existing optimizer/scheduler policy, preprocessing, losses, and sequence-disjoint DAVIS split unless a specific bug prevents execution. Document any necessary compatibility change separately.

Do not silently replace the existing model classes, input representations, reconstruction rules, or quality-metric definitions. Old state-dict checkpoints must remain loadable. Keep previous experimental results separate from the new seeded runs; an unknown historical seed cannot be recovered by editing checkpoint metadata.

## 2. Reproducibility and repeatable training

Add shared reproducibility utilities used by both active training entry points. Expose an explicit training seed, deterministic-mode option, and separate run/output directory. Seed Python, NumPy, PyTorch CPU, and CUDA before model construction. Seed any independent generators that are actually used.

Use an explicitly seeded DataLoader generator independent of model initialization, together with a top-level worker initialization function. For a given run seed, different architectures must receive the same sample order and any augmentation draws. Do not accidentally restart the same shuffle every epoch. Preserve dataset splits across seeds and record a hash of the ordered split/sample lists.

Use supported deterministic settings for the installed PyTorch/CUDA versions, including disabling cuDNN algorithm benchmarking in deterministic mode. Configure any required cuBLAS environment setting before CUDA initialization. If Python hash seeding is used, set it in the launcher before interpreter startup. A strict deterministic run must stop with a nonzero exit code if an unsupported nondeterministic operation is encountered; do not silently downgrade it to a reproducible run. Provide an explicitly labeled alternative if strict mode is unavailable. Reproducibility is scoped to the recorded software and hardware environment.

The current training loops catch arbitrary batch exceptions and continue. Replace this behavior for research runs with failure reporting and termination, including detection of nonfinite losses/gradients and zero-successful-batch epochs. Do not let skipped batches or an outer exception handler produce an apparently successful experiment.

Save per-run metadata: seed, configuration, architecture options, loss weights, optimizer/scheduler settings, input size, split hashes, software versions, GPU/driver details when available, precision/backend settings, commit and dirty-worktree status, completed epochs/steps, checkpoint identity, and run status. Prevent accidental overwriting of runs.

Save full resumable checkpoints containing model, optimizer, scheduler, epoch/step, configuration, and Python/NumPy/PyTorch/CUDA/DataLoader generator states; include scaler state if mixed precision is used. Support and test deterministic resume at epoch boundaries. State the resume boundary explicitly rather than promising exact mid-epoch resume. Retain compatibility with old bare state dictionaries for inference.

Prepare a configurable sequential launcher with a dry-run mode. Start with seeds 11, 22, and 33; allow adding 44 and 55. Each seed requires fresh model and optimizer initialization. Multiple evaluations of one checkpoint do not count as repeated training. Keep the checkpoint-selection rule fixed across variants and seeds; the existing final-epoch rule can be retained. Do not select runs using external-test performance.

Export every run's PSNR, SSIM, LPIPS, and PCE on the same datasets, with SNU-FILM subsets reported separately. Add an aggregator that computes per-dataset mean and sample standard deviation across independently trained seeds, reports the number of completed runs, and supports paired per-seed differences between variants. Keep seed variability separate from variation across frames. Do not pool different resolutions or protocols, silently omit failed runs, or report standard deviation as zero when only one run exists.

## 3. Isolate convolution within the complex architecture

The existing full-model comparison changes convolution, normalization, feature representation, and phase handling together. Implement additional controls inside `ComplexPhaseNetSafe` so that only the convolution operator changes. Preserve complex batch normalization, the existing componentwise activations, input/output representation, interpolation and phase rotation, prediction context, pyramid, widths, training objective, optimizer policy, and data order. This is a conditional ablation within the complex architecture.

Provide three explicit convolution choices with the same two-tensor input/output interface. The names below are proposed labels, not existing APIs. Here `*` denotes convolution; each formula also has one bias per output component.

1. **Complex convolution, existing reference:**
   `y_r = A*x_r - B*x_i + b_r`
   `y_i = B*x_r + A*x_i + b_i`

2. **Separate real convolutions:**
   `y_r = A*x_r + b_r`
   `y_i = B*x_i + b_i`
   Keep the same component widths. This has the same two learned kernel tensors and bias count as the complex operator, but fewer arithmetic operations. It removes cross-component interaction in the convolution; normalization may still couple the components. Do not call the entire model uncoupled or claim that this comparison alone establishes the superiority of complex algebra.

3. **Unrestricted real convolution on concatenated components:**
   Concatenate `[x_r, x_i]`, apply a real convolution from `2*C_in` to `2*C_out`, then split the result into two components. Equivalently, its four kernel blocks are independently trainable. At equal component width this has twice as many convolution weights, not twice as many total model parameters. Report the actual counts and do not label this control parameter-matched. Reducing width to match parameters would be an additional comparison with a changed width, not the primary single-operator ablation.

Apply the selected operator consistently to the feature convolutions and prediction-head convolutions. Preserve all other modules and the existing default behavior. Record the scope of replacement in configuration and checkpoints. Use the existing loss-matched objective for all three convolution variants so the complex reference can be reused from that experiment.

Control initialization explicitly. Copy matching non-convolution parameters and buffers across paired variants. For the unrestricted real control, provide initialization from the complex reference using the block matrix `[[A, -B], [B, A]]` and the same biases, so it starts with the same function but the four blocks can subsequently train independently. Record this initialization policy. Equal numerical seeds alone do not guarantee comparable initialization when parameter layouts differ.

Do not add new normalization or activation choices in this task. Existing whole-model baselines remain useful, but they answer a different question from this convolution ablation.

## 4. Computational measurements

Add a standalone benchmark that loads old and new checkpoints and writes machine-readable CSV/JSON. Measure a complete RGB midpoint prediction, with batch size 1 and a specified common resolution and precision. Start at 256 by 256; allow other explicit resolutions. Record tiling settings and any padding. The current complex model can use tiling, so an unnoticed difference in tiling would affect both memory and latency. Report out-of-memory cases without silently changing resolution, precision, device, or tiling.

Separate two scopes:

- **Network inference:** prepared pyramid inputs through the neural predictor, covering all three colour channels for one output RGB frame. State exactly what is cached and excluded.
- **Full interpolation:** two input RGB tensors through input-pyramid decomposition, coefficient preparation, prediction, output conversion, inverse-pyramid reconstruction, and output clamping. Define input/output device placement and whether host-device transfer is included. Exclude disk I/O, quality metrics, visualisation, and checkpoint loading.

Build an inference-only path accepting the two endpoint frames. Existing evaluation also decomposes the ground-truth middle frame and computes LPIPS/PCE; neither belongs in an inference benchmark. Verify that the new endpoint-only path reproduces the existing evaluator's predictions within a documented numerical tolerance.

Report:

- Unique total/trainable scalar parameter counts.
- MACs and FLOPs for the counted operations, with the convention of two real floating-point operations per real multiply-accumulate. Count calls, not merely unique weight tensors: the current `ComplexConv2d` calls its two internal real convolution modules four times per forward pass. Avoid counting both a wrapper and its child convolutions. Account for all three colour channels and any tile overlap.
- Coverage of the operation count. Pyramid FFTs, normalization, interpolation, trigonometric functions, and other operations must not silently become zero-cost operations. Either provide validated counting rules or report the supported subtotal and the excluded operations. `torch.profiler(with_flops=True)` alone is not a complete pipeline FLOP counter. Distinguish operation counts from FLOP/s.
- Latency in milliseconds per RGB output frame: warm up first, then collect independent timed calls and report mean, median, and p95. Make warm-up and repetition counts configurable; reasonable initial settings are 30 and 100. Use CUDA events for device timing and synchronized wall-clock timing for full-pipeline latency, including CPU work performed inside that pipeline. Keep detailed profiling/hooks out of latency measurements.
- Peak allocated and peak reserved GPU memory in MiB, clearly identified as PyTorch allocator measurements. Reset peak statistics immediately before the measured workload after warm-up and synchronization. Report the total peak including resident model/input tensors, rather than only an unlabeled incremental difference. Avoid retaining outputs across repetitions, and isolate benchmark cases so previous models do not inflate the result. These values are not total GPU usage as shown by nvidia-smi.

Use evaluation mode and inference mode. Record FP32/mixed-precision, TF32, deterministic, compilation, and cuDNN settings. Use the same policy across compared variants. Keep any faster nondeterministic measurements separate from deterministic ones. Report missing GPU measurements honestly if the execution environment has no CUDA device; do not estimate them from CPU timings.

## 5. Focused tests and deliverables

Add tests that establish:

- Two short CPU training runs with the same seed reproduce initialization, sample order, losses, and final parameters; a different seed changes initialization. Test the CUDA guarantee only where supported.
- Matching run seeds produce matching data order across different model widths and convolution choices.
- Epoch-boundary resume matches an uninterrupted run in the supported environment.
- The three operators have the expected shapes, finite gradients, and parameter counts; the initialized unrestricted real operator matches the complex operator's output within tolerance.
- Only the selected convolution modules change in an ablation; normalization, activation, phase construction, and non-convolution initialization remain aligned.
- Endpoint-only inference matches existing predictions and does not depend on a ground-truth middle frame.
- A hand-calculable small convolution example validates MAC/FLOP counts and catches shared-module call undercounting.
- The result aggregator handles incomplete runs and calculates sample standard deviation correctly.

Reuse existing tests and dependencies where practical. Run small smoke checks for the active real and complex training/evaluation paths and every new convolution choice. Avoid silently upgrading the entire environment to solve a local API incompatibility.

Deliver the implementation, explicit experiment configurations, benchmark and aggregation tools, a concise experiment guide, and copy-paste commands for my laptop. Use the existing dataset paths as defaults where available, with configurable alternatives. Provide a six-configuration, three-seed dry-run plan: the four existing article configurations plus the separate-real and unrestricted-real convolution controls under the loss-matched objective. This is 18 fresh training runs; the complex loss-matched reference is shared rather than trained twice. Make the launcher resumable and sequential on one GPU.

Finish with the changed files, tests actually run, any unresolved limitations, and exact next commands. Keep old results separate and do not edit the article to imply that the new experiments have already been completed.

## Primary references for implementation

- PyTorch reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch CUDA timing and memory semantics: https://docs.pytorch.org/docs/stable/notes/cuda.html
- PyTorch peak allocated memory: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html
- PyTorch profiler operation-count coverage: https://docs.pytorch.org/docs/stable/profiler.html

Check these against the repository's installed PyTorch version; do not introduce a newer API without compatibility handling.
