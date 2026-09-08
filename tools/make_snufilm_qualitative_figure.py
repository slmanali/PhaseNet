#!/usr/bin/env python3
"""Render internal or optional external qualitative figures from export metadata."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

INTERNAL = [
    ("gt", "Ground truth"),
    ("phasenet_default", "PhaseNet-default"),
    ("phasenet_big", "PhaseNet-big"),
    ("complex_loss_matched", "ComplexPhaseNet\nLoss-Matched"),
    ("complex_full", "ComplexPhaseNet (full)"),
]
EXTERNAL_LABELS = {"gt": "Ground truth", "ifrnet": "IFRNet", "rife": "RIFE",
                   "cain": "CAIN", "dain": "DAIN", "sepconv": "SepConv",
                   "complex_full": "ComplexPhaseNet (full)"}


def read(path):
    return np.asarray(Image.open(path).convert("RGB"))


def panel(frame, boxes):
    """Original full-frame thumbnail plus identical enlarged crops."""
    h, w = frame.shape[:2]
    thumb_w = 260
    thumb_h = max(1, round(h * thumb_w / w))
    thumb = np.asarray(Image.fromarray(frame).resize((thumb_w, thumb_h)))
    crop_side = thumb_h // max(1, len(boxes))
    crops = []
    for box in boxes:
        crop = Image.fromarray(frame).crop(tuple(box)).resize((crop_side, crop_side))
        crops.append(np.asarray(crop))
    result = np.full((thumb_h, thumb_w + crop_side * len(crops) + 4 * len(crops), 3), 255, np.uint8)
    result[:, :thumb_w] = thumb
    x = thumb_w + 4
    for crop in crops:
        y = (thumb_h - crop_side) // 2
        result[y:y + crop_side, x:x + crop_side] = crop
        x += crop_side + 4
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--kind", choices=("internal", "external"), default="internal")
    parser.add_argument("--methods", nargs="*", help="External method keys/order.")
    parser.add_argument("--external-dir", type=Path, default=Path("outputs/external_snufilm"),
                        help="METHOD/subset/NNNNN_pred.png prediction convention.")
    parser.add_argument("--selection", choices=("all", "representative", "illustrative-failure"), default="all")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        help="Output stem; PNG/PDF/SVG are written.")
    parser.add_argument("--annotate", choices=("none", "lpips", "pce", "psnr", "ssim"), default="none")
    args = parser.parse_args()
    data = json.loads(args.metadata.read_text())
    samples = [s for s in data["samples"] if args.selection == "all" or s["selection"] == args.selection]
    samples = samples[:args.max_samples]
    if not samples:
        parser.error("metadata contains no matching samples")
    if args.kind == "internal":
        methods = INTERNAL
    else:
        keys = args.methods or ["gt", "ifrnet", "rife", "cain", "complex_full"]
        methods = [(key.lower(), EXTERNAL_LABELS.get(key.lower(), key)) for key in keys]
    output = args.output or Path(f"fig_snufilm_{args.kind}_qualitative")
    fig, axes = plt.subplots(len(methods), len(samples),
                             figsize=(3.25 * len(samples), 1.72 * len(methods)), squeeze=False)
    colors = ["#e41a1c", "#377eb8"]
    for col, sample in enumerate(samples):
        axes[0, col].set_title(f"{sample['subset'].title()} · {sample['selection']}\n"
                               f"sample {sample['sample_id']:03d}", fontsize=9, weight="bold")
        for row, (key, label) in enumerate(methods):
            if key in sample["source_paths"]:
                path = Path(sample["source_paths"][key])
            else:
                path = args.external_dir / key / sample["subset"] / f"{sample['sample_id']:05d}_pred.png"
            ax = axes[row, col]
            ax.axis("off")
            if not path.is_file():
                ax.text(.5, .5, f"{label}\nnot available", ha="center", va="center", color=".45")
            else:
                frame = read(path)
                ax.imshow(panel(frame, sample["crop_boxes"]))
                h, w = frame.shape[:2]
                for color, box in zip(colors, sample["crop_boxes"]):
                    x0, y0, x1, y1 = box
                    ax.add_patch(patches.Rectangle((x0 / w * 260, y0 / h * (260*h/w)),
                                 (x1-x0) / w * 260, (y1-y0) / h * (260*h/w),
                                 linewidth=1.2, edgecolor=color, facecolor="none"))
                value = sample.get("metrics", {}).get(key, {}).get(args.annotate)
                if args.annotate != "none" and value is not None:
                    ax.text(.99, .02, f"{args.annotate.upper()} {value:.3f}", transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=6, color="white",
                            bbox={"facecolor": "black", "alpha": .65, "pad": 1.5})
            if col == 0:
                ax.text(-.035, .5, label, transform=ax.transAxes, ha="right", va="center",
                        fontsize=8, weight="bold")
    resolution = "native resolution" if data.get("resolution") == "native" else "256×256 evaluation"
    fig.text(.995, .006, resolution, ha="right", fontsize=7, color=".35")
    fig.subplots_adjust(left=.15, right=.995, top=.91, bottom=.03, wspace=.025, hspace=.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}.png, {output}.pdf, and {output}.svg")


if __name__ == "__main__":
    main()
