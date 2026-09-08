#!/usr/bin/env python3
"""Select and export reproducible SNU-FILM qualitative examples.

Predictions may be generated with ``--checkpoint`` or supplied in directories.
The latter is deliberately model-agnostic and is also the interface for external
methods.  Run from the repository root (or from any directory).
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.snufilm import SNUFILMTriplets, SNU_MODES  # noqa: E402

MODELS = {
    "phasenet_default": ("real", 64),
    "phasenet_big": ("real", 93),
    "complex_loss_matched": ("complex", 64),
    "complex_full": ("complex", 64),
}


def image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def metrics(pred, gt):
    mse = float(np.mean((pred - gt) ** 2))
    # Dependency-free global SSIM is used only for reporting/ranking fallback.
    ux, uy = pred.mean(), gt.mean()
    vx, vy = pred.var(), gt.var()
    cov = np.mean((pred - ux) * (gt - uy))
    ssim = ((2 * ux * uy + .01**2) * (2 * cov + .03**2) /
            ((ux**2 + uy**2 + .01**2) * (vx + vy + .03**2)))
    return {"psnr": None if mse == 0 else -10 * math.log10(mse),
            "ssim": float(ssim), "lpips": None, "pce": None, "mse": mse}


def prediction_path(root, subset, index):
    candidates = [root / subset / f"{index:05d}_pred.png",
                  root / subset / f"sample_{index:03d}.png",
                  root / f"{index:05d}_pred.png"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("prediction missing; checked: " + ", ".join(map(str, candidates)))


def boxes(big, full, gt, size, count=2):
    h, w = gt.shape[:2]
    side = min(size, h, w)
    score = np.abs(big - gt).mean(2) + np.abs(big - full).mean(2)
    chosen = []
    work = score.copy()
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(work), work.shape)
        x0 = max(0, min(w - side, x - side // 2)); y0 = max(0, min(h - side, y - side // 2))
        chosen.append([int(x0), int(y0), int(x0 + side), int(y0 + side)])
        work[max(0, y0-side//2):min(h, y0+3*side//2),
             max(0, x0-side//2):min(w, x0+3*side//2)] = -1
    return chosen


def generate(args, pred_roots, checkpoints):
    for name, checkpoint in checkpoints.items():
        kind, feature = MODELS[name]
        script = "test.py" if kind == "real" else "test_complex_safe_baseline.py"
        command = [sys.executable, str(ROOT / script), "--dataset-type", "snufilm",
                   "--dataset-path", str(args.snu_root), "--snu-mode", "all",
                   "--image-size", args.image_size, "--batch-size", "1",
                   "--model-path", str(checkpoint), "--feature-dim", str(feature),
                   "--model-name", name, "--save-dir", str(pred_roots[name]), "--save-all"]
        if args.device:
            command += ["--device", args.device]
        subprocess.run(command, cwd=ROOT, check=True)


def parse_pairs(values, label):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must be NAME=PATH: {value}")
        name, path = value.split("=", 1)
        if name not in MODELS:
            raise ValueError(f"unknown internal model {name!r}")
        result[name] = Path(path).expanduser().resolve()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snu-root", type=Path, required=True)
    parser.add_argument("--pred-dir", action="append", default=[], metavar="MODEL=DIR")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="MODEL=FILE")
    parser.add_argument("--output-dir", type=Path, default=Path("qualitative_snufilm"))
    parser.add_argument("--metadata", type=Path, default=Path("qualitative_snufilm/metadata.json"))
    parser.add_argument("--image-size", choices=("native", "256"), default="native")
    parser.add_argument("--selection-metric", choices=("lpips", "pce", "mse"), default="mse",
                        help="LPIPS/PCE require values in --metrics-json; MSE is reproducible fallback.")
    parser.add_argument("--metrics-json", type=Path,
                        help="Optional {subset: {index: {model: {metric: value}}}} file.")
    parser.add_argument("--crop-overrides", type=Path,
                        help="Optional {subset: {index: [[x0,y0,x1,y1], ...]}} file.")
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--device")
    args = parser.parse_args()
    pred_roots = parse_pairs(args.pred_dir, "--pred-dir")
    checkpoints = parse_pairs(args.checkpoint, "--checkpoint")
    for name in MODELS:
        pred_roots.setdefault(name, (ROOT / "outputs" / f"{name}_snufilm").resolve())
    if checkpoints:
        generate(args, pred_roots, checkpoints)
    missing_models = [name for name in MODELS if not pred_roots[name].exists()]
    if missing_models:
        parser.error("missing prediction directories for: " + ", ".join(missing_models))
    supplied = json.loads(args.metrics_json.read_text()) if args.metrics_json else {}
    overrides = json.loads(args.crop_overrides.read_text()) if args.crop_overrides else {}
    payload = {"schema_version": 1, "resolution": args.image_size,
               "selection_metric": args.selection_metric, "samples": []}
    for subset in SNU_MODES:
        dataset = SNUFILMTriplets(args.snu_root, subset, args.image_size)
        candidates = []
        for idx, paths in enumerate(dataset.triplets):
            loaded = dataset[idx]
            frames = {key: value.permute(1, 2, 0).numpy()
                      for key, value in loaded.items()}
            gt = frames["inter"]
            model_metrics, arrays, sources = {}, {}, {}
            for name in MODELS:
                path = prediction_path(pred_roots[name], subset, idx)
                arr = image(path)
                if arr.shape != gt.shape:
                    raise ValueError(f"resolution mismatch for {path}: {arr.shape} != {gt.shape}")
                values = metrics(arr, gt)
                values.update(supplied.get(subset, {}).get(str(idx), {}).get(name, {}))
                model_metrics[name], arrays[name], sources[name] = values, arr, str(path)
            key = args.selection_metric
            if model_metrics["phasenet_big"].get(key) is None or model_metrics["complex_full"].get(key) is None:
                raise ValueError(f"{key} missing for {subset}/{idx}; provide --metrics-json or use --selection-metric mse")
            improvement = model_metrics["phasenet_big"][key] - model_metrics["complex_full"][key]
            difficulty = np.mean([v["mse"] for v in model_metrics.values()])
            candidates.append((idx, improvement, difficulty, model_metrics, arrays, sources, paths))
        improvements = np.array([c[1] for c in candidates])
        median = float(np.median(improvements))
        representative = min(candidates, key=lambda c: (abs(c[1] - median), c[0]))
        # Strong improvement first, then hard cases; stable index resolves ties.
        illustrative = max(candidates, key=lambda c: (c[1], c[2], -c[0]))
        if illustrative[0] == representative[0] and len(candidates) > 1:
            illustrative = sorted(candidates, key=lambda c: (c[1], c[2], -c[0]), reverse=True)[1]
        for selection, candidate in (("representative", representative),
                                     ("illustrative-failure", illustrative)):
            idx, improvement, difficulty, mm, arrays, sources, paths = candidate
            out = args.output_dir / subset / f"sample_{idx:03d}"
            out.mkdir(parents=True, exist_ok=True)
            selected_frames = dataset[idx]
            names = (("start", "input_1.png"), ("inter", "gt.png"), ("end", "input_2.png"))
            for key, name in names:
                array = (selected_frames[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                Image.fromarray(array).save(out / name)
            for name, source in sources.items():
                Image.open(source).convert("RGB").save(out / f"{name}.png")
            crop = overrides.get(subset, {}).get(str(idx)) or boxes(
                arrays["phasenet_big"], arrays["complex_full"],
                selected_frames["inter"].permute(1, 2, 0).numpy(), args.crop_size)
            payload["samples"].append({"subset": subset, "sample_id": idx,
                "selection": selection, "source_paths": {"input_1": str(paths[0]),
                "gt": str(paths[1]), "input_2": str(paths[2]), **sources},
                "metrics": mm, "improvement": float(improvement),
                "difficulty_mse": float(difficulty), "crop_boxes": crop,
                "export_dir": str(out.resolve())})
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Exported {len(payload['samples'])} selections; metadata: {args.metadata}")


if __name__ == "__main__":
    main()
