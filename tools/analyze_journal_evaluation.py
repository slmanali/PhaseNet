"""Seed variation and paired source-video bootstrap, reported separately."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from tools.run_saved_evaluation import METRICS, aggregate
from utils.evaluation_protocol import write_csv, write_json

DEFAULT_PAIRS = (("complex-loss-matched-64", "phasenet-93"),
                 ("complex-full-64", "phasenet-93"),
                 ("complex-full-64", "complex-loss-matched-64"),
                 ("phasenet-93", "phasenet-64"))


def paired_cluster_bootstrap(differences, groups, repetitions=10000, seed=20261017):
    """Hold trained runs fixed and resample whole videos, preserving frame weights.

    differences: [training seed, paired triplet]. The same video resampling is
    used for every seed. This interval concerns the evaluation videos, not a new
    random training run. Its estimand is the equal-seed mean of triplet means.
    """
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 2 or differences.shape[1] != len(groups):
        raise ValueError("expected seed-by-triplet differences and one group per triplet")
    if not np.isfinite(differences).all() or repetitions < 100:
        raise ValueError("finite differences and at least 100 bootstrap replicates are required")
    names = sorted(set(groups))
    if len(names) < 2:
        raise ValueError("at least two independent source videos are needed")
    group_sums = np.stack([differences[:, np.array(groups) == name].sum(axis=1) for name in names], axis=1)
    counts = np.array([groups.count(name) for name in names])
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions)
    for start in range(0, repetitions, 500):
        end = min(start + 500, repetitions)
        draw = rng.integers(0, len(names), size=(end - start, len(names)))
        denominators = counts[draw].sum(axis=1)
        values[start:end] = (group_sums[:, draw].sum(axis=2) / denominators).mean(axis=0)
    return {"mean_difference": float(differences.mean()), "ci95_low": float(np.quantile(values, .025)),
            "ci95_high": float(np.quantile(values, .975)), "source_videos": len(names),
            "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
            "uncertainty": "evaluation videos conditional on the saved training runs"}


def load_runs(root):
    records = []
    indexed = {}
    for path in sorted(root.glob("*/seed-*/*/*/summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("status") != "completed":
            continue
        key = (summary["variant"], summary["seed"], summary["dataset"], summary["subset"])
        if key in indexed:
            raise ValueError(f"duplicate run: {key}")
        with (path.parent / "samples.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != summary["samples"] or len({row["sample_key"] for row in rows}) != len(rows):
            raise ValueError(f"incomplete or duplicate sample IDs: {path}")
        diagnostic_path = path.parent / "phase_diagnostics.csv"
        if diagnostic_path.exists():
            with diagnostic_path.open(newline="") as stream:
                diagnostics = {row["sample_key"]: row for row in csv.DictReader(stream)}
            if set(diagnostics) != {row["sample_key"] for row in rows}:
                raise ValueError("diagnostic sample IDs differ")
            for row in rows:
                row.update(diagnostics[row["sample_key"]])
            diagnostic_summary = json.loads((path.parent / "diagnostics.json").read_text())
            summary["phase_diagnostics"] = diagnostic_summary
        records.append(summary)
        indexed[key] = (summary, rows)
    if not records:
        raise ValueError("no completed evaluation summaries found")
    if len({r["study_id"] for r in records}) != 1 or len({r["stage"] for r in records}) != 1:
        raise ValueError("do not pool legacy and corrected studies")
    source_versions = {json.dumps(r["source_sha256"], sort_keys=True) for r in records}
    if len(source_versions) != 1:
        raise ValueError("evaluation implementations differ between runs")
    return records, indexed


def compare_pairs(indexed, pairs, seeds, repetitions):
    results, per_seed = [], []
    conditions = sorted({(key[2], key[3]) for key in indexed})
    for left, right in pairs:
        for dataset, subset in conditions:
            common = [seed for seed in seeds if (left, seed, dataset, subset) in indexed
                      and (right, seed, dataset, subset) in indexed]
            if not common:
                continue
            vectors = defaultdict(list)
            sample_keys, groups, verified = None, None, True
            for seed in common:
                a, a_rows = indexed[(left, seed, dataset, subset)]
                b, b_rows = indexed[(right, seed, dataset, subset)]
                if a["sample_manifest_sha256"] != b["sample_manifest_sha256"]:
                    raise ValueError("paired models evaluated on different sample manifests")
                a_map, b_map = {r["sample_key"]: r for r in a_rows}, {r["sample_key"]: r for r in b_rows}
                keys = sorted(a_map)
                if set(a_map) != set(b_map) or (sample_keys is not None and sample_keys != keys):
                    raise ValueError("pairing requires identical triplets across models and seeds")
                sample_keys = keys
                new_groups = [a_map[key]["sequence_id"] for key in keys]
                if groups is not None and groups != new_groups:
                    raise ValueError("source-video grouping differs across seeds")
                groups = new_groups
                verified &= all(str(a_map[key]["sequence_grouping_verified"]).lower() == "true" for key in keys)
                metrics = list(METRICS)
                if all("reconstructed_pce" in a_map[key] and "reconstructed_pce" in b_map[key] for key in keys):
                    metrics.append("reconstructed_pce")
                for metric in metrics:
                    difference = [float(a_map[key][metric]) - float(b_map[key][metric]) for key in keys]
                    vectors[metric].append(difference)
                    per_seed.append({"left": left, "right": right, "dataset": dataset, "subset": subset,
                        "seed": seed, "metric": metric, "left_minus_right": statistics.mean(difference)})
            for metric, values in vectors.items():
                seed_means = [statistics.mean(value) for value in values]
                row = {"left": left, "right": right, "dataset": dataset, "subset": subset,
                       "metric": metric, "difference_direction": "left minus right",
                       "training_seeds": common, "missing_seeds": sorted(set(seeds) - set(common)),
                       "mean_difference": statistics.mean(seed_means),
                       "training_seed_sample_std": statistics.stdev(seed_means) if len(seed_means) > 1 else None}
                if verified and len(set(groups)) >= 2:
                    row.update(paired_cluster_bootstrap(values, groups, repetitions))
                    row["interval_status"] = "computed"
                else:
                    row.update(interval_status="requires verified grouping into at least two source videos",
                               ci95_low=None, ci95_high=None)
                results.append(row)
    return results, per_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/computer_optics/corrected_12models"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--pair", nargs=2, action="append", metavar=("LEFT", "RIGHT"))
    args = parser.parse_args()
    records, indexed = load_runs(args.results)
    destination = args.results / "analysis"
    write_csv(destination / "seed_summary.csv", aggregate(records, args.seeds))
    diagnostics = []
    for record in records:
        if "phase_diagnostics" in record:
            diagnostics.append({key: record[key] for key in ("variant", "seed", "dataset", "subset")}
                               | record["phase_diagnostics"])
    write_csv(destination / "phase_diagnostics_per_seed.csv", diagnostics)
    pairs, per_seed = compare_pairs(indexed, args.pair or DEFAULT_PAIRS, args.seeds, args.bootstrap_repetitions)
    write_csv(destination / "paired_training_seeds.csv", per_seed)
    write_csv(destination / "paired_video_intervals.csv", pairs)
    write_json(destination / "paired_video_intervals.json", pairs)
    print(f"Saved summaries and paired uncertainty to {destination}")


if __name__ == "__main__":
    main()
