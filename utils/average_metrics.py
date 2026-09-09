"""Scalar-only tracking and joint average-case selection for SNU-FILM."""

import csv
import json
from pathlib import Path

import numpy as np

MODELS = ("phasenet_default", "phasenet_big", "complex_loss_matched", "complex_full")
MODES = ("easy", "medium", "hard", "extreme")
METRICS = ("psnr", "ssim", "lpips", "pce")
EPSILON = 1e-8
PER_SAMPLE_FIELDS = ("model", "mode", "sample_index", *METRICS)


class ScalarMetricsTracker:
    """Collect scalar values only; this class never retains image tensors."""

    def __init__(self, model, mode):
        self.model, self.mode, self.rows = model, mode, []

    def update(self, sample_index, psnr, ssim, lpips, pce):
        self.rows.append({"model": self.model, "mode": self.mode,
                          "sample_index": int(sample_index),
                          "psnr": float(psnr), "ssim": float(ssim),
                          "lpips": float(lpips), "pce": float(pce)})

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        old = []
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                old = [row for row in csv.DictReader(handle)
                       if (row["model"], row["mode"]) != (self.model, self.mode)]
        rows = old + self.rows
        rows.sort(key=lambda row: (row["model"], row["mode"], int(row["sample_index"])))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PER_SAMPLE_FIELDS)
            writer.writeheader(); writer.writerows(rows)


def mode_averages(rows):
    """Return population means/stds (ddof=0), matching publication aggregation."""
    grouped, output = {}, []
    for row in rows:
        grouped.setdefault((row["model"], row["mode"]), []).append(row)
    for (model, mode), items in sorted(grouped.items()):
        result = {"model": model, "mode": mode, "samples": len(items)}
        for metric in METRICS:
            values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
            result[f"mean_{metric}"] = float(values.mean())
            result[f"std_{metric}"] = float(values.std(ddof=0))
        output.append(result)
    return output


def representative_candidates(rows, averages, k=5, epsilon=EPSILON):
    """Rank one common sample per mode using all model/metric deviations."""
    if k not in (1, 3, 5):
        raise ValueError("average-k must be one of 1, 3, or 5")
    avg = {(row["model"], row["mode"]): row for row in averages}
    indexed = {(row["mode"], int(row["sample_index"]), row["model"]): row for row in rows}
    output = []
    for mode in MODES:
        sample_sets = [{int(row["sample_index"]) for row in rows
                        if row["mode"] == mode and row["model"] == model}
                       for model in MODELS]
        if not sample_sets or any(samples != sample_sets[0] for samples in sample_sets[1:]):
            raise ValueError(f"{mode}: models do not contain one common sample-index set")
        candidates = []
        for index in sorted(sample_sets[0]):
            record = {"mode": mode, "sample_index": index}
            deviations = []
            for model in MODELS:
                sample, mean = indexed[(mode, index, model)], avg[(model, mode)]
                for metric in METRICS:
                    value = float(sample[metric])
                    deviation = abs(value - mean[f"mean_{metric}"]) / (mean[f"std_{metric}"] + epsilon)
                    record[f"{model}_{metric}"] = value
                    record[f"{model}_{metric}_deviation"] = deviation
                    deviations.append(deviation)
            record["representative_score"] = float(np.mean(deviations))
            candidates.append(record)
        candidates.sort(key=lambda row: (row["representative_score"], row["sample_index"]))
        for rank, record in enumerate(candidates[:k], 1):
            output.append({"rank": rank, **record})
    return output


def write_csv_json(rows, csv_path, json_path, root_key):
    csv_path, json_path = Path(csv_path), Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields: writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps({root_key: rows}, indent=2) + "\n", encoding="utf-8")
