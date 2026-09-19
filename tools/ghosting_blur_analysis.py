#!/usr/bin/env python3
"""Publication-oriented, same-sample SNU-FILM blur/ghosting visual analysis.

Rows: source I0, GT, source I1, ComplexPhaseNet, RIFE, FILM.
Columns: full frame (same ROI), nearest-neighbor zoom, RGB absolute error.
Optional --line adds GT-vs-model luminance profiles across a manually chosen edge.
Scores describe crop differences only; they are NOT validated ghosting detectors.
"""
import argparse
import csv
import hashlib
import json
import shlex
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MODES = ("easy", "medium", "hard", "extreme")
LABELS = (("input_0", "Input I0"), ("gt", "Ground truth"),
          ("input_1", "Input I1"), ("complex", "ComplexPhaseNet"),
          ("rife", "RIFE"), ("film", "FILM"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def official_triplet(root, mode, index):
    file = root / "eval_modes" / f"test-{mode}.txt"
    if not file.is_file():
        file = root / f"test-{mode}.txt"
    if not file.is_file():
        raise FileNotFoundError(f"Official list not found: {root}/eval_modes/test-{mode}.txt")
    lines = [shlex.split(s) for s in file.read_text(encoding="utf-8").splitlines()
             if s.strip() and not s.lstrip().startswith("#")]
    if not 0 <= index < len(lines) or len(lines[index]) != 3:
        raise ValueError(f"Index {index} invalid in {file} ({len(lines)} triplets)")
    result = []
    for entry in lines[index]:
        raw = Path(entry).expanduser()
        candidates = [raw] if raw.is_absolute() else [root / raw, file.parent / raw]
        matches = [p.resolve() for p in candidates if p.is_file()]
        if not matches:
            raise FileNotFoundError(f"Frame {entry!r} not found relative to {root} or {file.parent}")
        result.append(matches[0])
    return result, file


def read(path):
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError(f"Expected RGB image, got {im.mode}: {path}")
        return np.asarray(im, dtype=np.uint8).copy()


def gray(image):
    rgb = image.astype(np.float32) / 255.0
    return rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114


def gradients(image):
    gy, gx = np.gradient(gray(image))
    return np.hypot(gx, gy)


def bilinear(image, xs, ys):
    y_max, x_max = image.shape[:2]
    xs, ys = np.clip(xs, 0, x_max - 1), np.clip(ys, 0, y_max - 1)
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, x_max - 1); y1 = np.minimum(y0 + 1, y_max - 1)
    dx, dy = xs - x0, ys - y0
    return ((1 - dx)*(1 - dy)*image[y0, x0] + dx*(1 - dy)*image[y0, x1]
            + (1 - dx)*dy*image[y1, x0] + dx*dy*image[y1, x1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snu-root", type=Path, default=Path("SNU-FILM"))
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--index", type=int, help="Zero-based index; defaults to published representative sample")
    p.add_argument("--config", type=Path, default=Path("figures/snufilm_average/selections.json"))
    p.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="Manual native-resolution ROI, shared by all models")
    p.add_argument("--line", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="Optional edge-crossing line entirely within --crop (global image coordinates)")
    p.add_argument("--phenomenon", choices=("blur", "ghosting", "mixed", "inspection"),
                   default="inspection", help="Human hypothesis, not an automated detection result")
    p.add_argument("--roi-note", default="", help="Document object / contour used to select ROI")
    p.add_argument("--external-dir", type=Path, default=Path("outputs/external_snufilm"))
    p.add_argument("--internal-dir", type=Path, default=Path("qualitative_snufilm_average_selected"))
    p.add_argument("--complex-pred", type=Path, help="Override ComplexPhaseNet prediction PNG")
    p.add_argument("--output-dir", type=Path, default=Path("figures/snufilm_ghosting_blur"))
    a = p.parse_args()
    a.snu_root = a.snu_root.expanduser().resolve()
    selections = {}
    if a.config.is_file():
        selections = json.loads(a.config.read_text(encoding="utf-8")).get("selections", {})
    sample = selections.get(a.mode, {})
    index = a.index if a.index is not None else sample.get("sample_index")
    if index is None:
        p.error("Specify --index or supply a selection config for this mode")
    index = int(index)
    crop = list(a.crop) if a.crop is not None else sample.get("crop")
    if crop is None or len(crop) != 4:
        p.error("Specify --crop X0 Y0 X1 Y1 or provide crop in selection config")
    triplet, list_file = official_triplet(a.snu_root, a.mode, index)
    complex_path = (a.complex_pred if a.complex_pred is not None else
                    a.internal_dir / "complex_full" / a.mode / "samples" / str(index) / "prediction.png")
    paths = {"input_0": triplet[0], "gt": triplet[1], "input_1": triplet[2],
             "complex": complex_path,
             "rife": a.external_dir / "rife" / a.mode / f"{index:05d}_pred.png",
             "film": a.external_dir / "film" / a.mode / f"{index:05d}_pred.png"}
    images = {key: read(path) for key, path in paths.items()}
    shape = images["gt"].shape
    mismatch = {k: v.shape for k, v in images.items() if v.shape != shape}
    if mismatch:
        raise ValueError(f"Native image dimensions do not match GT {shape}: {mismatch}")
    H, W = shape[:2]
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        raise ValueError(f"Invalid shared crop {crop} for {W}x{H}")
    if a.line is not None:
        lx0, ly0, lx1, ly1 = a.line
        if not all(x0 <= x < x1 and y0 <= y < y1 for x, y in ((lx0, ly0), (lx1, ly1))):
            raise ValueError("Both --line endpoints must be INSIDE the shared --crop")
        if np.hypot(lx1 - lx0, ly1 - ly0) < 3:
            raise ValueError("--line must be at least 3 pixels long")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.mode}_{index:05d}"
    gt_crop = images["gt"][y0:y1, x0:x1]
    gt_grad = gradients(gt_crop)
    fields = ("mode", "sample_index", "model", "crop_x0", "crop_y0", "crop_x1", "crop_y1",
              "crop_mae", "gradient_mae", "edge_strength_ratio")
    records = []
    for key in ("complex", "rife", "film"):
        pcrop = images[key][y0:y1, x0:x1]
        g = gradients(pcrop)
        records.append({"mode": a.mode, "sample_index": index, "model": key,
                        "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
                        "crop_mae": float(np.mean(np.abs(pcrop.astype(float) - gt_crop.astype(float))) / 255.0),
                        "gradient_mae": float(np.mean(np.abs(g - gt_grad))),
                        "edge_strength_ratio": float(g.sum() / (gt_grad.sum() + 1e-8))})
    with (a.output_dir / f"{stem}_crop_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(records)

    # The absolute-error color scale is identical across all rows.
    fig, axes = plt.subplots(len(LABELS), 3, figsize=(17, 19), constrained_layout=True)
    for row, (key, label) in enumerate(LABELS):
        image = images[key]
        full, zoom, error = axes[row]
        full.imshow(image, interpolation="nearest")
        full.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                         edgecolor="yellow", linewidth=1.7))
        full.set_ylabel(label, fontsize=11, weight="bold")
        full.set_xticks([]); full.set_yticks([])
        crop_image = image[y0:y1, x0:x1]
        zoom.imshow(crop_image, interpolation="nearest")
        if a.line is not None:
            zoom.plot([a.line[0]-x0, a.line[2]-x0], [a.line[1]-y0, a.line[3]-y0],
                      color="cyan", linewidth=1.3)
        zoom.set_xticks([]); zoom.set_yticks([])
        abs_error = np.mean(np.abs(crop_image.astype(np.float32)-gt_crop.astype(np.float32)), axis=2)/255
        error.imshow(abs_error, cmap="magma", vmin=0, vmax=.3, interpolation="nearest")
        error.set_xticks([]); error.set_yticks([])
        if row == 0:
            for ax, title in zip((full, zoom, error),
                                 ("Full frame + common ROI", "Shared ROI (nearest-neighbor display)",
                                  "Mean absolute RGB error (fixed 0–0.3 scale)")):
                ax.set_title(title, fontsize=10, weight="bold")
    fig.suptitle(f"SNU-FILM {a.mode.title()}, official index {index}: {a.phenomenon} inspection",
                 fontsize=16, weight="bold")
    main_figure = a.output_dir / f"{stem}_comparison.png"
    fig.savefig(main_figure, dpi=210, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    if a.line is not None:
        n = max(64, int(np.ceil(np.hypot(a.line[2]-a.line[0], a.line[3]-a.line[1])*3)))
        xs = np.linspace(a.line[0], a.line[2], n)
        ys = np.linspace(a.line[1], a.line[3], n)
        fig, ax = plt.subplots(figsize=(10, 4.3), constrained_layout=True)
        for key, label in (LABELS[1], *LABELS[3:]):
            profile = bilinear(gray(images[key]), xs, ys)
            ax.plot(np.linspace(0, 1, n), profile, label=label, linewidth=1.6)
        ax.set(xlabel="Normalized distance along identical line", ylabel="Grayscale luminance [0, 1]",
               title=f"Edge profile: {a.mode} sample {index} (manual transect)")
        ax.legend(); ax.grid(alpha=.18)
        fig.savefig(a.output_dir / f"{stem}_edge_profile.png", dpi=240,
                    facecolor="white", bbox_inches="tight")
        plt.close(fig)
    metadata = {"mode": a.mode, "sample_index": index, "official_list": str(list_file),
                "native_dimensions_wh": [W, H], "crop_xyxy": crop, "line_xyxy": a.line,
                "phenomenon_to_inspect": a.phenomenon, "roi_note": a.roi_note,
                "selection_policy": "existing internal-model representative selection" if a.index is None else "explicit user index",
                "note": "Crop MAE/gradients are descriptive, not a validated ghosting or blur detector; annotate artifacts only after visual verification.",
                "inputs_sha256": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()}}
    (a.output_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(f"Saved {main_figure}")
    if a.line is not None:
        print(f"Saved {a.output_dir / (stem+'_edge_profile.png')}")
    print(f"Saved {a.output_dir / (stem+'_crop_metrics.csv')}")
    print(f"Saved {a.output_dir / (stem+'_metadata.json')}")
    for r in records:
        print(f"{r['model']:<8} crop MAE={r['crop_mae']:.5f} "
              f"gradient MAE={r['gradient_mae']:.5f} edge-strength ratio={r['edge_strength_ratio']:.4f}")


if __name__ == "__main__":
    main()
