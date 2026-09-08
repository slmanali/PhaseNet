#!/usr/bin/env python3
"""Build fair, same-triplet cross-model SNU-FILM comparison figures."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

MODELS = ("phasenet_default", "phasenet_big", "complex_loss_matched", "complex_full")
LABELS = {
    "phasenet_default": "PhaseNet-default",
    "phasenet_big": "PhaseNet-big",
    "complex_loss_matched": "ComplexPhaseNet-Loss-Matched",
    "complex_full": "ComplexPhaseNet",
}
METRICS = ("psnr", "ssim", "lpips", "pce")


def read_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def sample_directory(root, model, mode, index):
    return root / model / mode / "samples" / str(index)


def load_case(root, mode, index):
    case = {}
    missing = []
    for model in MODELS:
        directory = sample_directory(root, model, mode, index)
        required = [directory / name for name in
                    ("input_1.png", "ground_truth.png", "input_2.png",
                     "prediction.png", "metrics.json")]
        missing.extend(path for path in required if not path.is_file())
        if not missing:
            case[model] = {
                "input_1": read_image(required[0]),
                "ground_truth": read_image(required[1]),
                "input_2": read_image(required[2]),
                "prediction": read_image(required[3]),
                "metrics": json.loads(required[4].read_text(encoding="utf-8")),
                "directory": directory,
            }
    if missing:
        preview = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "A fair comparison requires every model output for the same sample. "
            "Missing files:\n" + preview
        )
    reference = case[MODELS[0]]
    for model in MODELS[1:]:
        for key in ("input_1", "ground_truth", "input_2"):
            if not np.array_equal(reference[key], case[model][key]):
                raise ValueError(f"{mode} sample {index}: {key} differs for {model}")
        if case[model]["prediction"].shape != reference["ground_truth"].shape:
            raise ValueError(f"{mode} sample {index}: prediction size differs for {model}")
    return case


def suggest_crops(case, count=2):
    """Score shared windows by big-model error, model difference, and GT edges."""
    gt = case[MODELS[0]]["ground_truth"].astype(np.float32) / 255
    big = case["phasenet_big"]["prediction"].astype(np.float32) / 255
    full = case["complex_full"]["prediction"].astype(np.float32) / 255
    error = np.mean(np.abs(big - gt), axis=2)
    difference = np.mean(np.abs(big - full), axis=2)
    gray = np.mean(gt, axis=2)
    gy, gx = np.gradient(gray)
    score = error + difference + .35 * np.hypot(gx, gy)
    height, width = score.shape
    side = max(24, min(height, width) // 4)
    integral = np.pad(score, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    window = (integral[side:, side:] - integral[:-side, side:]
              - integral[side:, :-side] + integral[:-side, :-side])
    boxes = []
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(window), window.shape)
        boxes.append([int(x), int(y), int(x + side), int(y + side)])
        y0, y1 = max(0, y - side), min(window.shape[0], y + side)
        x0, x1 = max(0, x - side), min(window.shape[1], x + side)
        window[y0:y1, x0:x1] = -np.inf
    return boxes


def crop_config(output_dir, case, mode, index, count):
    path = output_dir / "crop_coordinates.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        boxes = payload["crops"]
    else:
        boxes = suggest_crops(case, count)
        payload = {"mode": mode, "sample_index": index,
                   "coordinate_format": "[x0, y0, x1, y1]", "crops": boxes,
                   "note": "Edit these coordinates and rerun to override automatic suggestions."}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    height, width = case[MODELS[0]]["ground_truth"].shape[:2]
    if not boxes or any(len(box) != 4 or box[0] < 0 or box[1] < 0
                        or box[2] > width or box[3] > height
                        or box[0] >= box[2] or box[1] >= box[3] for box in boxes):
        raise ValueError(f"Invalid crop coordinates in {path} for {width}x{height} images")
    return boxes


def save_case_files(case, output_dir):
    reference = case[MODELS[0]]
    for key in ("input_1", "ground_truth", "input_2"):
        Image.fromarray(reference[key]).save(output_dir / f"{key}.png")
    metrics = {}
    for model in MODELS:
        shutil.copyfile(case[model]["directory"] / "prediction.png", output_dir / f"{model}.png")
        metrics[model] = {name: case[model]["metrics"][name] for name in METRICS}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                              encoding="utf-8")


def make_figure(case, boxes, mode, index, layout, output_dir):
    keys = (["ground_truth", "phasenet_big", "complex_full"] if layout == "compact" else
            ["ground_truth", *MODELS])
    frames = {"ground_truth": case[MODELS[0]]["ground_truth"]}
    frames.update({model: case[model]["prediction"] for model in MODELS})
    titles = {"ground_truth": "Ground Truth", **LABELS}
    rows = 1 + len(boxes)
    fig, axes = plt.subplots(rows, len(keys), figsize=(3.0 * len(keys), 2.45 * rows),
                             squeeze=False)
    colors = ("#e41a1c", "#377eb8")
    for column, key in enumerate(keys):
        axes[0, column].imshow(frames[key])
        axes[0, column].set_title(titles[key], fontsize=10, weight="bold")
        for color, box in zip(colors, boxes):
            x0, y0, x1, y1 = box
            axes[0, column].add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.5))
        axes[0, column].axis("off")
        for row, (color, box) in enumerate(zip(colors, boxes), 1):
            x0, y0, x1, y1 = box
            axes[row, column].imshow(frames[key][y0:y1, x0:x1])
            for spine in axes[row, column].spines.values():
                spine.set_color(color); spine.set_linewidth(2)
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    fig.suptitle(f"SNU-FILM {mode.title()} · official sample {index}", weight="bold")
    fig.subplots_adjust(left=.015, right=.995, bottom=.02, top=.91, wspace=.025, hspace=.08)
    stem = output_dir / f"comparison_{layout}"
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def update_table(output_root, cases):
    path = output_root / "selected_samples_metrics.csv"
    old = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
    replaced = {str(index) for _, index, _ in cases}
    rows = [row for row in old if row["sample_index"] not in replaced]
    for mode, index, case in cases:
        for model in MODELS:
            rows.append({"sample_index": index, "model": model,
                         **{metric: case[model]["metrics"][metric] for metric in METRICS}})
    rows.sort(key=lambda row: (int(row["sample_index"]), row["model"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "model", *METRICS))
        writer.writeheader(); writer.writerows(rows)


def parse_indices(args, parser):
    if (args.sample_index is None) == (args.sample_indices is None):
        parser.error("provide exactly one of --sample-index or --sample-indices")
    values = str(args.sample_index) if args.sample_index is not None else args.sample_indices
    try:
        indices = [int(value) for value in values.split(",")]
    except ValueError:
        parser.error("sample indices must be comma-separated integers")
    if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        parser.error("sample indices must be unique non-negative integers")
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("easy", "medium", "hard", "extreme"))
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--sample-indices")
    parser.add_argument("--layout", choices=("compact", "full", "both"), default="both")
    parser.add_argument("--crop-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--source-dir", type=Path, default=Path("qualitative_snufilm_best"))
    parser.add_argument("--output-dir", type=Path, default=Path("qualitative_comparisons"))
    args = parser.parse_args()
    indices = parse_indices(args, parser)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for index in indices:
        case = load_case(args.source_dir, args.mode, index)
        directory = args.output_dir / f"{args.mode}_{index}"
        directory.mkdir(parents=True, exist_ok=True)
        boxes = crop_config(directory, case, args.mode, index, args.crop_count)
        save_case_files(case, directory)
        layouts = ("compact", "full") if args.layout == "both" else (args.layout,)
        for layout in layouts:
            make_figure(case, boxes, args.mode, index, layout, directory)
        cases.append((args.mode, index, case))
    update_table(args.output_dir, cases)
    print(f"Wrote {len(cases)} fair comparison(s) under {args.output_dir}")


if __name__ == "__main__":
    main()
