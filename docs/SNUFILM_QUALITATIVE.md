# SNU-FILM qualitative figures

The workflow has two stages so that selection is reproducible and figure styling
does not rerun a model. No manuscript files are modified.

## Inputs and prediction export

Expected dataset layout:

```text
SNU-FILM/eval_modes/test-{easy,medium,hard,extreme}.txt
SNU-FILM/test/...
```

Every official list line contains `input_1 ground_truth input_2`. Predictions use
`outputs/<model>_snufilm/<subset>/00000_pred.png`; indices are zero-based official
list positions. Generate all four sets and export the selected cases in one run:

```bash
python tools/export_snufilm_qualitative.py --snu-root /data/SNU-FILM \
  --checkpoint phasenet_default=/checkpoints/default.pth \
  --checkpoint phasenet_big=/checkpoints/big.pth \
  --checkpoint complex_loss_matched=/checkpoints/complex-matched.pth \
  --checkpoint complex_full=/checkpoints/complex-full.pth \
  --selection-metric lpips --metrics-json per_sample_metrics.json
```

Without `--checkpoint`, pass existing roots with repeated
`--pred-dir model=/path`. Model keys are `phasenet_default`, `phasenet_big`,
`complex_loss_matched`, and `complex_full`. Native resolution is the default;
pass `--image-size 256` only when every method was evaluated at 256×256.

`per_sample_metrics.json` has the shape
`{subset: {index: {model: {lpips, pce}}}}`. The exporter always computes PSNR,
global SSIM, and MSE from saved images. Use `--selection-metric mse` when LPIPS or
PCE sidecars are unavailable. Crop overrides have the shape
`{subset: {index: [[x0,y0,x1,y1], ...]}}`.

For each subset, the representative case minimizes distance to the median
PhaseNet-big minus ComplexPhaseNet error improvement. The illustrative-failure
case maximizes that improvement, breaking ties by mean model MSE (difficulty)
and official index. It cannot duplicate the representative case when another
case exists. Automatic crops maximize the sum of PhaseNet-big absolute GT error
and its difference from ComplexPhaseNet; the second crop is spatially suppressed
to avoid overlap. All rows use the exact same boxes and resolution.

## Figures

```bash
python tools/make_snufilm_qualitative_figure.py \
  --metadata qualitative_snufilm/metadata.json --kind internal \
  --output fig_snufilm_internal_qualitative
```

Optional external predictions follow
`outputs/external_snufilm/<method>/<subset>/00000_pred.png`. Missing predictions
are shown honestly as “not available”:

```bash
python tools/make_snufilm_qualitative_figure.py \
  --metadata qualitative_snufilm/metadata.json --kind external \
  --methods gt ifrnet rife cain complex_full \
  --output fig_snufilm_external_qualitative
```

Each command writes PNG, PDF, and SVG. The layout is an original full-frame plus
fixed-crop grid; IFRNet Figure 6 is a presentation reference only.

## Pretrained RIFE and FILM

Clone the upstream RIFE or FILM repository and download/extract its official
pretrained checkpoint. The runner imports the upstream implementation directly,
so this repository does not vendor model code or silently substitute weights.
Cloning either upstream repository downloads its code **but not its pretrained
weights**. Creating an empty checkpoint directory is not sufficient: download
the weights and extract the archive so that the directory passed to
`--checkpoint` is the archive's actual `train_log` (RIFE) or `saved_model`
(FILM) directory. The relative paths below assume the commands are run from the
PhaseNet repository root.

For RIFE, `--checkpoint` is the extracted `train_log` directory (RIFE 4.6 or a
newer release with the same `RIFE_HDv3.Model` API). Depending on the official
release, `RIFE_HDv3.py` may be in the checkout's `model` package or directly in
the extracted `train_log` beside `IFNet_HDv3.py` and `flownet.pkl`; the runner
supports both layouts. The similarly named `model.RIFE` is an older training
architecture and is not compatible with the published inference checkpoints:

First, clone the code. Then follow the upstream
[RIFE model download instructions](https://github.com/hzwer/ECCV2022-RIFE#model-list)
and extract the downloaded archive into `checkpoints/rife`. These are two
separate downloads: `git clone` does not create `train_log`.

```bash
mkdir -p opt checkpoints/rife
git clone https://github.com/hzwer/ECCV2022-RIFE.git "opt/RIFE"
```

After extracting the model archive, check its location without terminating the
current shell. The earlier `test ... || { exit 1; }` form was only a check; it
did not download the checkpoint, and `exit 1` can close terminals launched as a
task by an editor.

```bash
if test -d checkpoints/rife/train_log; then
  echo "RIFE checkpoint is ready"
else
  echo "RIFE checkpoint is missing: download and extract it into checkpoints/rife"
fi
```

Only run the evaluation after the check prints `RIFE checkpoint is ready`:

```bash
python tools/run_pretrained_snufilm.py --model rife \
  --repo "opt/RIFE" --checkpoint "checkpoints/rife/train_log" \
  --snu-root "SNU-FILM" --snu-mode all
```

For FILM, install this repository's optional TensorFlow dependency in the same
Python environment that runs the evaluation script, then point at the extracted
FILM SavedModel directory. In particular, cloning the upstream repository does
not install TensorFlow:

```bash
python -m pip install -r requirements-film.txt
mkdir -p opt checkpoints/film
git clone https://github.com/google-research/frame-interpolation.git "opt/FILM"
# Download and extract FILM's pretrained model as directed by its README.
python tools/run_pretrained_snufilm.py --model film \
  --repo "opt/FILM" \
  --checkpoint "checkpoints/film/film_net/Style/saved_model" \
  --snu-root "SNU-FILM" --snu-mode all
```

Before launching a long evaluation, verify all three input directories:

```bash
test -d "opt/RIFE" && \
test -d "checkpoints/rife/train_log" && \
test -d "SNU-FILM"
```

Outputs are written as
`outputs/external_snufilm/{rife,film}/<mode>/00000_pred.png`, ready for the
external qualitative figure command above. Use `--sample-indices 36,38,39` for
a quick subset, `--overwrite` to replace existing images, and `--device cpu`
for RIFE without CUDA. Both adapters preserve native SNU-FILM resolution; RIFE
pads only for inference and crops the result back to the input dimensions.

## Fair cross-model selected cases

Evaluate only the four official Hard triplets (rather than all 310) for every
checkpoint:

```bash
python tools/export_snufilm_qualitative.py --snu-root /data/SNU-FILM \
  --checkpoint phasenet_default=/checkpoints/default.pth \
  --checkpoint phasenet_big=/checkpoints/big.pth \
  --checkpoint complex_loss_matched=/checkpoints/complex-matched.pth \
  --checkpoint complex_full=/checkpoints/complex-full.pth \
  --mode hard --sample-indices 36,38,39,139 --image-size 256
```

Then generate compact and full figures, identical zooms, per-case assets, and
the merged `selected_samples_metrics.csv` verification table:

```bash
python tools/make_snufilm_cross_model_comparison.py \
  --mode extreme --sample-index 131
python tools/make_snufilm_cross_model_comparison.py \
  --mode hard --sample-indices 36,38,39,139
```

Each case's `crop_coordinates.json` is created once. Edit its `[x0,y0,x1,y1]`
boxes and rerun the figure command to apply the manual override identically to
the ground truth and every prediction. The generator rejects missing model
outputs, mismatched input triplets, and inconsistent resolutions instead of
silently producing an unfair comparison.

## Draft captions

**Internal.** Qualitative comparison on SNU-FILM. As motion difficulty increases,
all models exhibit stronger blur around motion boundaries and occluded regions.
ComplexPhaseNet generally preserves sharper local structure than the real-valued
PhaseNet baselines, especially in the challenging regions shown in the fixed,
shared zoom crops.

**External.** Qualitative comparison with representative video-frame-interpolation
methods on SNU-FILM, using an original layout inspired by the presentation of
IFRNet Figure 6. The reproducibly selected examples emphasize blur and
correspondence errors in large-motion cases; every method uses the same triplet,
resolution, and crop coordinates.
