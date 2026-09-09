#!/usr/bin/env python3
"""Create the final, fair same-triplet SNU-FILM paper figures."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


MODES = ("easy", "medium", "hard", "extreme")
MAIN_METHODS = ("phasenet_big", "complex_full", "ground_truth")
FULL_METHODS = (
    "phasenet_default", "phasenet_big", "complex_loss_matched",
    "complex_full", "ground_truth",
)
LABELS = {
    "phasenet_default": "PhaseNet-default",
    "phasenet_big": "PhaseNet-big",
    "complex_loss_matched": "ComplexPhaseNet-Loss-Matched",
    "complex_full": "ComplexPhaseNet",
    "ground_truth": "Ground Truth",
}
PREDICTION_MODELS = FULL_METHODS[:-1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).digest()


def load_cases(source, selections):
    """Load cases and reject mixed triplets before drawing any panel."""
    cases = {}
    for mode in MODES:
        index = selections[mode]["sample_index"]
        model_dirs = {m: source / m / mode / "samples" / str(index)
                      for m in PREDICTION_MODELS}
        required = ("input_1.png", "ground_truth.png", "input_2.png")
        missing = [p for directory in model_dirs.values()
                   for name in (*required, "prediction.png")
                   if not (p := directory / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing fair-comparison inputs:\n" +
                                    "\n".join(map(str, missing)))
        reference = model_dirs["phasenet_big"]
        for name in required:
            expected = digest(reference / name)
            for model, directory in model_dirs.items():
                if digest(directory / name) != expected:
                    raise ValueError(f"{mode} sample {index}: {name} differs for {model}")
        gt = Image.open(reference / "ground_truth.png").convert("RGB")
        frames = {"ground_truth": gt}
        for model, directory in model_dirs.items():
            frames[model] = Image.open(directory / "prediction.png").convert("RGB")
            if frames[model].size != gt.size:
                raise ValueError(f"{mode} sample {index}: size differs for {model}")
        x0, y0, x1, y1 = selections[mode]["crop"]
        width, height = gt.size
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Invalid {mode} crop for a {width}x{height} frame")
        cases[mode] = frames
    return cases


def render(cases, selections, methods, stem):
    """Render full frames and one identically located enlargement per method."""
    rows, columns = len(methods), len(MODES)
    fig = plt.figure(figsize=(3.0 * columns, 3.2 * rows), layout="constrained")
    grid = fig.add_gridspec(rows, columns, wspace=.025, hspace=.08)
    for row, method in enumerate(methods):
        for column, mode in enumerate(MODES):
            cell = grid[row, column].subgridspec(2, 1, height_ratios=(3, 1.32), hspace=.025)
            full = fig.add_subplot(cell[0])
            crop = fig.add_subplot(cell[1])
            frame = cases[mode][method]
            box = selections[mode]["crop"]
            x0, y0, x1, y1 = box
            full.imshow(frame)
            full.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False,
                edgecolor="#f03b20", linewidth=1.5))
            full.axis("off")
            crop.imshow(frame.crop(box), interpolation="nearest")
            crop.set_xticks([]); crop.set_yticks([])
            for spine in crop.spines.values():
                spine.set_color("#f03b20"); spine.set_linewidth(1.6)
            if row == 0:
                full.set_title(
                    f"{mode.title()}\nSample {selections[mode]['sample_index']}",
                    fontsize=11, weight="bold", pad=5)
            if column == 0:
                full.text(-.055, .5, LABELS[method], transform=full.transAxes,
                          rotation=90, ha="right", va="center", fontsize=10,
                          weight="bold")
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("figures/snufilm_final/selections.json"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("qualitative_snufilm_worst"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("figures/snufilm_final"))
    args = parser.parse_args()
    selections = json.loads(args.config.read_text(encoding="utf-8"))["selections"]
    if set(selections) != set(MODES):
        parser.error("config must contain easy, medium, hard, and extreme")
    cases = load_cases(args.source_dir, selections)
    main_stem = args.output_dir / "snufilm_qualitative_main"
    full_stem = args.output_dir / "snufilm_qualitative_full"
    render(cases, selections, MAIN_METHODS, main_stem)
    render(cases, selections, FULL_METHODS, full_stem)
    print("Selected crops ([x0, y0, x1, y1]):")
    for mode in MODES:
        print(f"  {mode}: {selections[mode]['crop']}")
    print("Selected samples:")
    print("  " + ", ".join(f"{m}={selections[m]['sample_index']}" for m in MODES))
    print("Output figures:")
    for stem in (main_stem, full_stem):
        print("  " + ", ".join(str(stem.with_suffix(s)) for s in (".png", ".pdf", ".svg")))
    print("Draft caption:")
    print("  Qualitative comparison on SNU-FILM from Easy to Extreme. Each column uses "
          "the same official triplet, ground truth, and crop for every method. The "
          "enlargements highlight progressively stronger blur, loss of local structure, "
          "edge smearing, and motion-boundary degradation; PhaseNet-big is visually "
          "blurrier than ComplexPhaseNet, which better preserves fine structure.")


if __name__ == "__main__":
    main()
