#!/usr/bin/env python3
"""Evaluate pre-generated pretrained RIFE / FILM predictions on SNU-FILM.

This script never runs RIFE or FILM.  It reads the PNG predictions produced by
``tools/run_pretrained_snufilm.py`` from::

    <predictions-dir>/<model>/<mode>/<index:05d>_pred.png

and scores them against the official SNU-FILM ground-truth middle frames using
the *same* dataset reader (``utils.snufilm.SNUFILMTriplets``) and the *same*
metric implementations (``utils.metrics.compute_psnr`` / ``compute_ssim`` /
``compute_lpips``) as the PhaseNet / ComplexPhaseNet SNU-FILM evaluation.

Conventions shared with ``test_complex_safe_baseline.py``:

* native SNU-FILM resolution, no resizing (a shape mismatch is an error);
* tensors are ``float`` in ``[0, 1]`` shaped ``[1, 3, H, W]`` (batch size 1);
* a single ``lpips.LPIPS(net="alex")`` instance is reused for every sample;
* per-mode numbers are the arithmetic mean of per-sample metrics.

PCE is intentionally *not* computed: the existing PCE uses the internal complex
steerable-pyramid coefficients of PhaseNet models and is not available from
external RGB predictions.

Example::

    python tools/evaluate_external_snufilm.py \
        --dataset-path /PATH/TO/SNU-FILM \
        --predictions-dir outputs/external_snufilm \
        --models rife film \
        --device cuda
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import compute_lpips, compute_psnr, compute_ssim  # noqa: E402
from utils.snufilm import SNU_MODES, SNUFILMTriplets  # noqa: E402

EXPECTED_SAMPLES = 310
SUPPORTED_MODELS = ("rife", "film")
MODEL_LABELS = {"rife": "RIFE", "film": "FILM"}
PREDICTION_PATTERN = re.compile(r"^(\d+)_pred\.png$")
SUMMARY_FIELDS = ("model", "mode", "samples", "psnr", "ssim", "lpips")
PER_SAMPLE_FIELDS = ("model", "mode", "sample_index", "psnr", "ssim", "lpips")
DEFAULT_PREDICTIONS_DIR = Path("outputs/external_snufilm")
DEFAULT_OUTPUT_DIR = Path("results/snufilm/native")


class PredictionSetError(RuntimeError):
    """Raised when a prediction directory is incomplete or inconsistent."""


# --------------------------------------------------------------------------- #
# Completeness checks
# --------------------------------------------------------------------------- #
def collect_predictions(mode_dir, expected_samples=EXPECTED_SAMPLES):
    """Return ``{index: path}`` for a mode directory after strict validation.

    Requirements enforced:

    * the directory exists;
    * exactly ``expected_samples`` prediction files are present;
    * their indices are exactly ``0 .. expected_samples - 1`` with no gaps;
    * no index appears twice (for example ``0_pred.png`` and ``00000_pred.png``).
    """
    mode_dir = Path(mode_dir)
    if not mode_dir.is_dir():
        raise PredictionSetError(f"prediction directory not found: {mode_dir}")

    found = {}
    duplicates = {}
    for path in sorted(mode_dir.iterdir()):
        match = PREDICTION_PATTERN.match(path.name)
        if not match or not path.is_file():
            continue
        index = int(match.group(1))
        if index in found:
            duplicates.setdefault(index, [found[index]]).append(path)
        else:
            found[index] = path

    if duplicates:
        preview = "\n".join(
            f"  index {index}: " + ", ".join(p.name for p in paths)
            for index, paths in sorted(duplicates.items())[:10]
        )
        raise PredictionSetError(
            f"{mode_dir}: {len(duplicates)} duplicate prediction index/indices:\n{preview}"
        )

    expected = set(range(expected_samples))
    missing = sorted(expected - found.keys())
    unexpected = sorted(found.keys() - expected)
    if missing or unexpected:
        parts = [f"{mode_dir}: expected exactly indices 0..{expected_samples - 1} "
                 f"({expected_samples} predictions), found {len(found)}"]
        if missing:
            parts.append(f"  missing {len(missing)} index/indices: "
                         + _preview_indices(missing))
        if unexpected:
            parts.append(f"  unexpected {len(unexpected)} index/indices: "
                         + _preview_indices(unexpected))
        raise PredictionSetError("\n".join(parts))

    return {index: found[index] for index in sorted(found)}


def _preview_indices(indices, limit=10):
    text = ", ".join(str(i) for i in indices[:limit])
    if len(indices) > limit:
        text += f", ... (+{len(indices) - limit} more)"
    return text


def check_completeness(predictions_dir, models, modes=SNU_MODES,
                       expected_samples=EXPECTED_SAMPLES):
    """Validate *every* model/mode directory before any metric is computed.

    Returns ``{model: {mode: {index: path}}}``.  All problems are gathered and
    reported together so a single run reveals every incomplete directory.
    """
    predictions_dir = Path(predictions_dir)
    if not predictions_dir.is_dir():
        raise PredictionSetError(f"predictions directory not found: {predictions_dir}")
    collected, problems = {}, []
    for model in models:
        for mode in modes:
            try:
                collected.setdefault(model, {})[mode] = collect_predictions(
                    predictions_dir / model / mode, expected_samples)
            except PredictionSetError as error:
                problems.append(str(error))
    if problems:
        raise PredictionSetError(
            "Prediction completeness check failed:\n" + "\n".join(problems))
    return collected


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_prediction(path, expected_shape=None):
    """Load a prediction PNG as a float ``[1, 3, H, W]`` tensor in ``[0, 1]``.

    Uses the same conversion as ``SNUFILMTriplets.__getitem__``.  The image
    must be RGB and, when ``expected_shape`` is given, match the ground truth
    exactly; nothing is ever resized.
    """
    path = Path(path)
    if not path.is_file():
        raise PredictionSetError(f"prediction file not found: {path}")
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"{path}: expected an RGB image, found mode {image.mode!r}")
        tensor = TF.pil_to_tensor(image).float() / 255.0
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"{path}: expected shape [3, H, W], found {tuple(tensor.shape)}")
    if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(
            f"{path}: resolution mismatch; prediction is {tuple(tensor.shape)} but "
            f"ground truth is {tuple(expected_shape)}. Predictions are evaluated "
            "at native SNU-FILM resolution and are never resized."
        )
    return tensor.unsqueeze(0)


def load_ground_truth(dataset, position):
    """Ground-truth middle frame as ``[1, 3, H, W]`` in ``[0, 1]``."""
    return dataset[position]["inter"].unsqueeze(0)


def build_lpips(device):
    """Single LPIPS-AlexNet instance shared by every sample (same as ComplexPhaseNet)."""
    import lpips  # imported lazily so completeness checks work without it

    return lpips.LPIPS(net="alex", verbose=False).to(device).eval()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_mode(dataset, predictions, loss_fn_lpips, device, model_label, mode,
                  progress=True):
    """Score every sample of one mode.  Returns ``(summary_row, per_sample_rows)``."""
    if len(dataset) != len(predictions):
        raise PredictionSetError(
            f"{model_label}/{mode}: dataset has {len(dataset)} triplets but "
            f"{len(predictions)} predictions were collected")
    if list(dataset.sample_indices) != sorted(predictions):
        raise PredictionSetError(
            f"{model_label}/{mode}: prediction indices do not match official sample indices")

    iterator = zip(dataset.sample_indices, range(len(dataset)))
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc=f"{model_label} {mode}", unit="sample")
        except ImportError:  # pragma: no cover - tqdm is in requirements
            pass

    rows = []
    with torch.no_grad():
        for sample_index, position in iterator:
            truth = load_ground_truth(dataset, position)
            pred = load_prediction(predictions[sample_index], truth.shape[1:])
            truth, pred = truth.to(device), pred.to(device)
            rows.append({
                "model": model_label,
                "mode": mode,
                "sample_index": int(sample_index),
                "psnr": compute_psnr(pred, truth),
                "ssim": compute_ssim(pred, truth),
                "lpips": compute_lpips(pred, truth, loss_fn_lpips),
            })

    summary = {
        "model": model_label,
        "mode": mode,
        "samples": len(rows),
        "psnr": _mean(row["psnr"] for row in rows),
        "ssim": _mean(row["ssim"] for row in rows),
        "lpips": _mean(row["lpips"] for row in rows),
    }
    return summary, rows


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def evaluate_all(dataset_root, predictions_dir, models, device, modes=SNU_MODES,
                 expected_samples=EXPECTED_SAMPLES, progress=True, loss_fn_lpips=None):
    """Run the completeness check, then evaluate every model/mode."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"SNU-FILM root not found: {dataset_root}")

    collected = check_completeness(predictions_dir, models, modes, expected_samples)

    datasets = {}
    for mode in modes:
        dataset = SNUFILMTriplets(dataset_root, mode, "native")
        if len(dataset) != expected_samples:
            raise PredictionSetError(
                f"official SNU-FILM {mode} list contains {len(dataset)} triplets, "
                f"expected {expected_samples}")
        datasets[mode] = dataset

    if loss_fn_lpips is None:
        loss_fn_lpips = build_lpips(device)

    summaries, per_sample = [], []
    for model in models:
        label = MODEL_LABELS.get(model, model.upper())
        for mode in modes:
            dataset = datasets[mode]
            if progress:
                print(f"\n{label} / {mode}: {len(dataset)} samples at {dataset.input_resolution}")
            summary, rows = evaluate_mode(dataset, collected[model][mode], loss_fn_lpips,
                                          device, label, mode, progress)
            summaries.append(summary)
            per_sample.extend(rows)
    return summaries, per_sample, datasets


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_outputs(summaries, per_sample, output_dir, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "external_metrics_snufilm.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({k: row[k] for k in SUMMARY_FIELDS} for row in summaries)

    per_sample_path = output_dir / "external_metrics_per_sample.csv"
    with per_sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows({k: row[k] for k in PER_SAMPLE_FIELDS} for row in per_sample)

    json_path = output_dir / "external_metrics_snufilm.json"
    payload = {
        "results": [{k: row[k] for k in SUMMARY_FIELDS} for row in summaries],
        "metadata": {
            "resolution": "native",
            "metrics": {
                "psnr": "utils.metrics.compute_psnr",
                "ssim": "utils.metrics.compute_ssim",
                "lpips": "utils.metrics.compute_lpips (lpips.LPIPS net='alex')",
            },
            "aggregation": "arithmetic mean of per-sample metrics",
            "pce": "not computed: PCE requires internal complex pyramid coefficients",
            "per_sample_csv": str(per_sample_path),
            **(metadata or {}),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path, per_sample_path


def format_table(summaries):
    header = f"{'Model':<8} | {'Mode':<8} | {'N':>4} | {'PSNR':>7} | {'SSIM':>7} | {'LPIPS':>7}"
    lines = [header, "-" * len(header)]
    for row in summaries:
        lines.append(f"{row['model']:<8} | {row['mode']:<8} | {row['samples']:>4} | "
                     f"{row['psnr']:>7.2f} | {row['ssim']:>7.4f} | {row['lpips']:>7.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, required=True,
                        help="SNU-FILM root containing eval_modes/test-<mode>.txt.")
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR,
                        help="Root holding <model>/<mode>/<index>_pred.png "
                             f"(default: {DEFAULT_PREDICTIONS_DIR}).")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS,
                        default=list(SUPPORTED_MODELS),
                        help="External models to evaluate (default: rife film).")
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. cpu, cuda, cuda:0 (default: cuda if available).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Where the CSV/JSON files are written (default: {DEFAULT_OUTPUT_DIR}).")
    args = parser.parse_args(argv)
    # Preserve user order but drop repeats.
    args.models = list(dict.fromkeys(args.models))
    return args


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"SNU-FILM root: {args.dataset_path}")
    print(f"Predictions: {args.predictions_dir}")
    print(f"Models: {', '.join(MODEL_LABELS.get(m, m) for m in args.models)}")
    print(f"Completeness requirement: {EXPECTED_SAMPLES} predictions per mode "
          f"({', '.join(SNU_MODES)}), indices 0..{EXPECTED_SAMPLES - 1}")

    summaries, per_sample, datasets = evaluate_all(
        args.dataset_path, args.predictions_dir, args.models, device,
        expected_samples=EXPECTED_SAMPLES)

    metadata = {
        "dataset_root": str(Path(args.dataset_path).expanduser().resolve()),
        "predictions_dir": str(Path(args.predictions_dir).expanduser().resolve()),
        "device": str(device),
        "triplet_lists": {mode: str(ds.list_file) for mode, ds in datasets.items()},
    }
    csv_path, json_path, per_sample_path = write_outputs(
        summaries, per_sample, args.output_dir, metadata)

    print()
    print(format_table(summaries))
    print(f"\nSaved summary to {csv_path} and {json_path}")
    print(f"Saved per-sample metrics to {per_sample_path}")


if __name__ == "__main__":
    main()
