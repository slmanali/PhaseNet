#!/usr/bin/env python3
"""Create average-case SNU-FILM figures with method-neutral shared crops."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MODES = ("easy", "medium", "hard", "extreme")
MODELS = ("phasenet_default", "phasenet_big", "complex_loss_matched", "complex_full")
METRICS = ("psnr", "ssim", "lpips", "pce")
MAIN = ("phasenet_big", "complex_full", "ground_truth")
FULL = (*MODELS, "ground_truth")
LABELS = {"phasenet_default": "PhaseNet-default", "phasenet_big": "PhaseNet-big",
          "complex_loss_matched": "ComplexPhaseNet-Loss-Matched",
          "complex_full": "ComplexPhaseNet", "ground_truth": "Ground Truth"}
CAPTION = ("Representative qualitative comparison on SNU-FILM. For each difficulty level, "
           "the displayed triplet is the common sample whose metric profile is closest to "
           "the 310-sample average across all four evaluated models. The same ground truth "
           "and crop coordinates are used for every method. Enlarged regions show local "
           "reconstruction behavior as motion difficulty increases from Easy to Extreme.")
EPSILON = 1e-8


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(root, selections):
    cases = {}
    for mode in MODES:
        index = int(selections[mode]["sample_index"]); case = {}
        directories = {m: root / m / mode / "samples" / str(index) for m in MODELS}
        for model, directory in directories.items():
            required = [directory / n for n in ("input_1.png", "ground_truth.png", "input_2.png", "prediction.png", "metrics.json")]
            missing = [p for p in required if not p.is_file()]
            if missing: raise FileNotFoundError("Missing selected output(s):\n" + "\n".join(map(str, missing)))
            case[model] = {"prediction": np.asarray(Image.open(required[3]).convert("RGB")),
                           "metrics": json.loads(required[4].read_text()), "directory": directory}
        for filename in ("input_1.png", "ground_truth.png", "input_2.png"):
            hashes = {model: sha256(directory / filename) for model, directory in directories.items()}
            if len(set(hashes.values())) != 1:
                raise ValueError(f"{mode} sample {index}: SHA256 mismatch for {filename}: {hashes}")
        gt = np.asarray(Image.open(directories[MODELS[0]] / "ground_truth.png").convert("RGB"))
        if any(item["prediction"].shape != gt.shape for item in case.values()):
            raise ValueError(f"{mode} sample {index}: prediction does not preserve native resolution {gt.shape[:2]}")
        case["ground_truth"] = gt; cases[mode] = case
    return cases


def gradient_magnitude(image):
    gray = np.mean(image.astype(np.float32) / 255.0, axis=2)
    gy, gx = np.gradient(gray)
    return np.hypot(gx, gy)


def method_neutral_crop(case):
    gt = case["ground_truth"]; gt_gradient = gradient_magnitude(gt)
    deficits, errors = [], []
    for model in MODELS:
        pred = case[model]["prediction"]
        deficits.append(np.maximum(0, gt_gradient - gradient_magnitude(pred)))
        errors.append(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)) / 255.0, axis=2))
    score = gt_gradient * np.mean(deficits, axis=0) + .25 * np.mean(errors, axis=0)
    height, width = score.shape
    side = max(96, min(256, int(round(min(height, width) * .20))))
    side = min(side, height, width)
    integral = np.pad(score, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    windows = integral[side:, side:] - integral[:-side, side:] - integral[side:, :-side] + integral[:-side, :-side]
    y, x = np.unravel_index(np.argmax(windows), windows.shape)
    return [int(x), int(y), int(x + side), int(y + side)]


def validate_crop(box, shape):
    x0, y0, x1, y1 = box; height, width = shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"invalid crop {box} for native frame {width}x{height}")


def render(cases, selections, methods, stem):
    fig = plt.figure(figsize=(12, 3.2 * len(methods)), layout="constrained")
    grid = fig.add_gridspec(len(methods), len(MODES), wspace=.025, hspace=.08)
    for row, method in enumerate(methods):
        for col, mode in enumerate(MODES):
            cell = grid[row, col].subgridspec(2, 1, height_ratios=(3, 1.3), hspace=.025)
            full, zoom = fig.add_subplot(cell[0]), fig.add_subplot(cell[1])
            frame = cases[mode]["ground_truth"] if method == "ground_truth" else cases[mode][method]["prediction"]
            x0, y0, x1, y1 = selections[mode]["crop"]
            full.imshow(frame); full.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="red", linewidth=1.5)); full.axis("off")
            zoom.imshow(frame[y0:y1, x0:x1], interpolation="nearest"); zoom.set_xticks([]); zoom.set_yticks([])
            for spine in zoom.spines.values(): spine.set_color("red"); spine.set_linewidth(1.5)
            if row == 0: full.set_title(f"{mode.title()}\nSample {selections[mode]['sample_index']}", weight="bold")
            if col == 0: full.text(-.055, .5, LABELS[method], transform=full.transAxes, rotation=90, ha="right", va="center", weight="bold")
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_metadata(cases, selections, output_dir):
    selected_rows, crop_rows = [], []
    for mode in MODES:
        index, score = selections[mode]["sample_index"], selections[mode]["representative_score"]
        x0, y0, x1, y1 = selections[mode]["crop"]; gt = cases[mode]["ground_truth"][y0:y1, x0:x1]
        gt_gradient = gradient_magnitude(gt)
        for model in MODELS:
            metrics = cases[mode][model]["metrics"]
            selected_rows.append({"mode": mode, "sample_index": index, "model": model,
                                  **{m: metrics[m] for m in METRICS}, "representative_score": score})
            pred = cases[mode][model]["prediction"][y0:y1, x0:x1]
            pred_gradient = gradient_magnitude(pred)
            crop_rows.append({"mode": mode, "sample_index": index, "model": model,
                              "crop_mae": float(np.mean(np.abs(pred.astype(float)-gt.astype(float))/255)),
                              "gradient_error": float(np.mean(np.abs(pred_gradient-gt_gradient))),
                              "edge_strength_ratio": float(pred_gradient.sum()/(gt_gradient.sum()+EPSILON))})
    for filename, rows in (("selected_sample_metrics.csv", selected_rows), ("crop_metrics.csv", crop_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    return selected_rows, crop_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("figures/snufilm_average/selections.json"))
    parser.add_argument("--source-dir", type=Path, default=Path("qualitative_snufilm_average_selected"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("qualitative_snufilm_average"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/snufilm_average"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.config.read_text()); selections = payload["selections"]
    if set(selections) != set(MODES): parser.error("config must contain all four SNU-FILM modes")
    cases = load_cases(args.source_dir, selections)
    for mode in MODES:
        automatic = method_neutral_crop(cases[mode])
        if "crop" in selections[mode]:
            selections[mode]["crop_manually_overridden"] = selections[mode]["crop"] != automatic
        else:
            selections[mode]["crop"] = automatic; selections[mode]["crop_manually_overridden"] = False
        selections[mode]["automatic_crop"] = automatic
        validate_crop(selections[mode]["crop"], cases[mode]["ground_truth"].shape)
    payload["crop_policy"] = "GT gradient strength times four-model average gradient deficit plus 0.25 times four-model average absolute error."
    payload["caption_draft"] = CAPTION
    args.config.write_text(json.dumps(payload, indent=2) + "\n")
    main_stem = args.output_dir / "snufilm_average_qualitative_main"; full_stem = args.output_dir / "snufilm_average_qualitative_full"
    render(cases, selections, MAIN, main_stem); render(cases, selections, FULL, full_stem)
    selected_rows, crop_rows = write_metadata(cases, selections, args.output_dir)
    print("Selected crops:"); [print(f"  {m}: {selections[m]['crop']}") for m in MODES]
    print("Selected-sample metrics:"); [print("  " + str(row)) for row in selected_rows]
    print("Crop metrics (PhaseNet-big and ComplexPhaseNet):"); [print("  " + str(row)) for row in crop_rows if row["model"] in ("phasenet_big", "complex_full")]
    print("Output figures:")
    for stem in (main_stem, full_stem): print("  " + ", ".join(str(stem.with_suffix(s)) for s in (".png", ".pdf", ".svg")))
    print("Draft caption:\n  " + CAPTION)

if __name__ == "__main__": main()
