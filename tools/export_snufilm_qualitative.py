#!/usr/bin/env python3
"""Evaluate SNU-FILM and export only the top metric samples for each model."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "phasenet_default": ("test.py", 64),
    "phasenet_big": ("test.py", 93),
    "complex_loss_matched": ("test_complex_safe_baseline.py", 64),
    "complex_full": ("test_complex_safe_baseline.py", 64),
}


def parse_pairs(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--checkpoint must be NAME=PATH: {value}")
        name, path = value.split("=", 1)
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODELS)}")
        if name in result:
            raise ValueError(f"duplicate checkpoint for {name}")
        result[name] = Path(path).expanduser().resolve()
    return result


def build_command(args, name, checkpoint):
    script, feature_dim = MODELS[name]
    command = [sys.executable, str(ROOT / script),
               "--dataset-type", "snufilm", "--dataset-path", str(args.snu_root),
               "--snu-mode", getattr(args, "mode", "all"), "--image-size", args.image_size,
               "--batch-size", "1", "--model-path", str(checkpoint),
               "--feature-dim", str(feature_dim), "--model-name", name,
               "--save-dir", str(args.output_dir / name),
               "--metrics-dir", str(args.output_dir / "quantitative_metrics")]
    if getattr(args, "sample_indices", None):
        command.extend(("--sample-indices", args.sample_indices))
        # Keeping k equal to the requested set saves every common triplet while
        # retaining the established best-tracker behavior for this explicit mode.
        command.extend(("--best-k", str(len(args.sample_indices.split(",")))))
    elif getattr(args, "worst_k", None) is not None:
        command.extend(("--worst-k", str(args.worst_k)))
    else:
        command.extend(("--best-k", str(args.best_k)))
    if args.device:
        command.extend(("--device", args.device))
    return command


def write_candidate_union(output_dir):
    """Build the per-mode union of independently selected failure candidates."""
    summary = output_dir / "worst_samples.csv"
    if not summary.is_file():
        return None, None
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for row in rows:
        key = (row["mode"], int(row["sample_index"]))
        entry = selected.setdefault(key, {"models": set(), "metrics": set()})
        entry["models"].add(row["model"])
        entry["metrics"].add(row["selection_metric"])
    union = {mode: sorted(index for (row_mode, index) in selected if row_mode == mode)
             for mode in ("easy", "medium", "hard", "extreme")}
    json_path = output_dir / "candidate_union.json"
    json_path.write_text(json.dumps(union, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "candidate_union.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("mode", "sample_index",
                                                    "selected_by_model",
                                                    "selected_by_metric"))
        writer.writeheader()
        for key in sorted(selected, key=lambda value: (("easy", "medium", "hard", "extreme").index(value[0]), value[1])):
            entry = selected[key]
            writer.writerow({"mode": key[0], "sample_index": key[1],
                             "selected_by_model": ";".join(sorted(entry["models"])),
                             "selected_by_metric": ";".join(sorted(entry["metrics"]))})
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snu-root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        metavar="MODEL=FILE")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("qualitative_snufilm_best"))
    parser.add_argument("--best-k", type=int, default=1,
                        help="Use the unchanged best-sample workflow (direct evaluator default).")
    parser.add_argument("--worst-k", type=int, choices=(1, 3, 5), default=1,
                        help="Failure candidates retained per LPIPS/PCE metric (default: 1).")
    parser.add_argument("--mode", choices=("easy", "medium", "hard", "extreme", "all"),
                        default="all")
    parser.add_argument("--sample-indices",
                        help="Comma-separated official indices; requires a single --mode.")
    parser.add_argument("--image-size", choices=("native", "256"), default="256",
                        help="Use the paper protocol (256) unless native is explicitly requested.")
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.best_k < 1:
        parser.error("--best-k must be at least 1")
    if args.sample_indices and args.mode == "all":
        parser.error("--sample-indices requires a single --mode")
    args.snu_root = args.snu_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    checkpoints = parse_pairs(args.checkpoint)
    missing = [name for name in MODELS if name not in checkpoints]
    if missing:
        parser.error("missing checkpoints for: " + ", ".join(missing))
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found: {checkpoint}")
        subprocess.run(build_command(args, name, checkpoint), cwd=ROOT, check=True)
    if args.sample_indices:
        print(f"Common-triplet samples exported under: {args.output_dir}")
    else:
        union_json, union_csv = write_candidate_union(args.output_dir)
        print(f"Worst-sample CSV: {args.output_dir / 'worst_samples.csv'}")
        print(f"Worst-sample JSON: {args.output_dir / 'worst_samples.json'}")
        print(f"Candidate union: {union_json} and {union_csv}")


if __name__ == "__main__":
    main()
