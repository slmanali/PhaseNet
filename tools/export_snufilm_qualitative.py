#!/usr/bin/env python3
"""Evaluate and export joint representative average-case SNU-FILM samples."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.average_metrics import (METRICS, MODELS as AVERAGE_MODELS, MODES,
                                   mode_averages, representative_candidates,
                                   write_csv_json)

MODELS = {
    "phasenet_default": ("test.py", 64),
    "phasenet_big": ("test.py", 93),
    "complex_loss_matched": ("test_complex_safe_baseline.py", 64),
    "complex_full": ("test_complex_safe_baseline.py", 64),
}
PUBLICATION_NAMES = {
    "phasenet_default": "PhaseNet-default", "phasenet_big": "PhaseNet-big",
    "complex_loss_matched": "ComplexPhaseNet-Loss-Matched",
    "complex_full": "ComplexPhaseNet-full",
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


def build_command(args, name, checkpoint, metrics_only=False):
    script, feature_dim = MODELS[name]
    command = [sys.executable, str(ROOT / script), "--dataset-type", "snufilm",
               "--dataset-path", str(args.snu_root), "--snu-mode", getattr(args, "mode", "all"),
               "--image-size", args.image_size, "--batch-size", "1",
               "--model-path", str(checkpoint), "--feature-dim", str(feature_dim),
               "--model-name", name, "--metrics-dir", str(args.output_dir / "quantitative_metrics")]
    if metrics_only:
        command.extend(("--per-sample-metrics", str(args.output_dir / f".{name}_per_sample.csv")))
    else:
        command.extend(("--save-dir", str(args.output_dir / name)))
        if getattr(args, "sample_indices", None):
            command.extend(("--sample-indices", args.sample_indices,
                            "--best-k", str(len(args.sample_indices.split(",")))))
        elif getattr(args, "worst_k", None) is not None:
            command.extend(("--worst-k", str(args.worst_k)))
        else:
            command.extend(("--best-k", str(getattr(args, "best_k", 1))))
    if args.device:
        command.extend(("--device", args.device))
    return command


def write_candidate_union(output_dir):
    """Preserve the established worst-case candidate-union helper."""
    summary = output_dir / "worst_samples.csv"
    if not summary.is_file():
        return None, None
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for row in rows:
        entry = selected.setdefault((row["mode"], int(row["sample_index"])),
                                    {"models": set(), "metrics": set()})
        entry["models"].add(row["model"]); entry["metrics"].add(row["selection_metric"])
    union = {mode: sorted(index for (row_mode, index) in selected if row_mode == mode)
             for mode in MODES}
    json_path, csv_path = output_dir / "candidate_union.json", output_dir / "candidate_union.csv"
    json_path.write_text(json.dumps(union, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("mode", "sample_index", "selected_by_model", "selected_by_metric")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for (mode, index), entry in sorted(selected.items()):
            writer.writerow({"mode": mode, "sample_index": index,
                             "selected_by_model": ";".join(sorted(entry["models"])),
                             "selected_by_metric": ";".join(sorted(entry["metrics"]))})
    return json_path, csv_path


def read_scalar_outputs(output_dir):
    rows = []
    for model in AVERAGE_MODELS:
        path = output_dir / f".{model}_per_sample.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
        path.unlink()
    return rows


def validate_complete(rows, selected_modes):
    for model in AVERAGE_MODELS:
        for mode in selected_modes:
            indices = {int(r["sample_index"]) for r in rows
                       if r["model"] == model and r["mode"] == mode}
            if indices != set(range(310)):
                raise RuntimeError(f"{model}/{mode}: expected all 310 official samples (0..309), got {len(indices)}")


def validate_publication(averages, publication_csv, tolerance):
    if not publication_csv.is_file():
        print(f"WARNING: publication metrics not found at {publication_csv}; reproducibility check skipped")
        return
    with publication_csv.open(newline="", encoding="utf-8") as handle:
        published = {(r["model"], r["mode"]): r for r in csv.DictReader(handle)}
    failures = []
    for row in averages:
        reference = published.get((PUBLICATION_NAMES[row["model"]], row["mode"]))
        if reference:
            for metric in METRICS:
                delta = abs(row[f"mean_{metric}"] - float(reference[metric]))
                if delta > tolerance:
                    failures.append(f"{row['model']}/{row['mode']} {metric}: delta={delta:.6g}")
    if failures:
        raise RuntimeError("WARNING: native average metrics differ materially from publication metrics:\n" + "\n".join(failures))


def write_selections(candidates, path):
    selected = {row["mode"]: {"sample_index": row["sample_index"],
                              "representative_score": row["representative_score"]}
                for row in candidates if row["rank"] == 1}
    payload = {"selection_policy": "Joint representative sample closest to the 310-sample mean metric profile across all four models and all four metrics.",
               "resolution": "native", "selections": selected}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return selected


def print_report(averages, candidates, selected):
    for mode in MODES:
        print(f"\n{mode.title()} mean metrics:")
        for row in averages:
            if row["mode"] == mode:
                print("  " + row["model"] + ": " + ", ".join(f"{m}={row['mean_'+m]:.6f}±{row['std_'+m]:.6f}" for m in METRICS))
        print(f"Top representative samples for {mode.title()}:")
        for row in candidates:
            if row["mode"] == mode:
                print(f"  rank {row['rank']}: sample {row['sample_index']}, score={row['representative_score']:.8f}")
        print(f"Selected rank 1: sample {selected[mode]['sample_index']}, score={selected[mode]['representative_score']:.8f}")


def run_average(args, checkpoints):
    if args.image_size != "native":
        raise ValueError("average-case publication workflow requires --image-size native")
    if args.mode != "all":
        raise ValueError("average-case workflow requires --mode all to select all four difficulties")
    for name, checkpoint in checkpoints.items():
        subprocess.run(build_command(args, name, checkpoint, metrics_only=True), cwd=ROOT, check=True)
    rows = read_scalar_outputs(args.output_dir)
    validate_complete(rows, MODES)
    write_csv_json(rows, args.output_dir / "per_sample_metrics.csv",
                   args.output_dir / "per_sample_metrics.json", "samples")
    averages = mode_averages(rows)
    write_csv_json(averages, args.output_dir / "mode_averages.csv",
                   args.output_dir / "mode_averages.json", "mode_averages")
    validate_publication(averages, args.publication_metrics, args.publication_tolerance)
    candidates = representative_candidates(rows, averages, args.average_k)
    write_csv_json(candidates, args.output_dir / "average_candidates.csv",
                   args.output_dir / "average_candidates.json", "candidates")
    config = args.figure_dir / "selections.json"
    selected = write_selections(candidates, config)
    print_report(averages, candidates, selected)
    if not args.metrics_only:
        selected_root = args.selected_output_dir
        for mode in MODES:
            args.mode = mode
            args.sample_indices = str(selected[mode]["sample_index"])
            args.output_dir = selected_root
            for name, checkpoint in checkpoints.items():
                subprocess.run(build_command(args, name, checkpoint), cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "tools/make_snufilm_average_paper_figure.py"),
                        "--config", str(config), "--source-dir", str(selected_root),
                        "--metrics-dir", str(args.average_output_dir),
                        "--output-dir", str(args.figure_dir)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snu-root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="MODEL=FILE")
    parser.add_argument("--output-dir", type=Path, default=Path("qualitative_snufilm_average"))
    parser.add_argument("--selected-output-dir", type=Path, default=Path("qualitative_snufilm_average_selected"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/snufilm_average"))
    parser.add_argument("--average-k", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--image-size", choices=("native", "256"), default="native")
    parser.add_argument("--device")
    parser.add_argument("--metrics-only", action="store_true", help="Stop after scalar ranking; do not export predictions or figures.")
    parser.add_argument("--publication-metrics", type=Path, default=ROOT / "results/snufilm/native/metrics_snufilm.csv")
    parser.add_argument("--publication-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    args.snu_root = args.snu_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve(); args.output_dir.mkdir(parents=True, exist_ok=True)
    args.average_output_dir = args.output_dir
    args.selected_output_dir = args.selected_output_dir.expanduser().resolve()
    args.figure_dir = args.figure_dir.expanduser().resolve()
    checkpoints = parse_pairs(args.checkpoint)
    missing = [name for name in MODELS if name not in checkpoints]
    if missing: parser.error("missing checkpoints for: " + ", ".join(missing))
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file(): parser.error(f"checkpoint not found: {checkpoint}")
    run_average(args, checkpoints)


if __name__ == "__main__":
    main()
