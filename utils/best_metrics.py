"""Memory-bounded best-sample tracking and export for qualitative evaluation."""

import csv
import json
from pathlib import Path

import torch
from torchvision.utils import save_image


METRIC_DIRECTIONS = {"psnr": True, "ssim": True, "lpips": False, "pce": False}
WORST_METRIC_DIRECTIONS = {"lpips": True, "pce": True}
SUMMARY_FIELDS = ("model", "mode", "selection_metric", "sample_index", "psnr",
                  "ssim", "lpips", "pce", "input_1_path", "gt_path",
                  "input_2_path", "prediction_path")


class _ExtremaMetricsTracker:
    """Memory-bounded metric extrema tracker (batch size must be one)."""

    def __init__(self, save_dir, model, mode, k, directions, label):
        if k < 1:
            raise ValueError(f"{label}_k must be at least 1")
        self.save_dir = Path(save_dir)
        self.model, self.mode, self.k = model, mode, k
        self.directions, self.label = directions, label
        self.extrema = {name: [] for name in directions}

    def update(self, sample_index, psnr, ssim, lpips, pce, pred, truth,
               input_1=None, input_2=None, source_paths=None):
        values = {"psnr": float(psnr), "ssim": float(ssim),
                  "lpips": float(lpips), "pce": float(pce)}
        candidate = {"sample_index": int(sample_index), "metrics": values,
                     "pred": pred.detach().cpu().squeeze(0),
                     "truth": truth.detach().cpu().squeeze(0),
                     "input_1": None if input_1 is None else input_1.detach().cpu().squeeze(0),
                     "input_2": None if input_2 is None else input_2.detach().cpu().squeeze(0),
                     "source_paths": dict(source_paths or {})}
        for metric, maximize in self.directions.items():
            entries = self.extrema[metric]
            # Keep one candidate object per metric; discarded images become collectible.
            entries.append(candidate.copy())
            entries.sort(key=lambda item: ((-item["metrics"][metric] if maximize else
                                            item["metrics"][metric]),
                                           item["sample_index"]))
            del entries[self.k:]

    def selections(self):
        return {metric: list(entries) for metric, entries in self.extrema.items()}

    def save(self):
        """Save unique image data, lightweight selection metadata, and global summaries."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        by_index = {}
        for metric, entries in self.extrema.items():
            for rank, item in enumerate(entries, 1):
                record = by_index.setdefault(item["sample_index"], item)
                selection = metric if self.label == "best" else f"{self.label}_{metric}"
                record.setdefault("selected_by", []).append(selection)
                record.setdefault("ranks", {})[selection] = rank

        rows, records = [], []
        for index, item in sorted(by_index.items()):
            sample_dir = self.save_dir / "samples" / str(index)
            sample_dir.mkdir(parents=True, exist_ok=True)
            image_tensors = {"input_1": item["input_1"], "ground_truth": item["truth"],
                             "input_2": item["input_2"], "prediction": item["pred"]}
            for name, tensor in image_tensors.items():
                if tensor is not None:
                    save_image(tensor.clamp(0, 1), sample_dir / f"{name}.png")
            if item["truth"] is not None:
                save_image(torch.cat((item["truth"].clamp(0, 1),
                                      item["pred"].clamp(0, 1)), dim=2),
                           sample_dir / "comparison.png")
            paths = {"input_1": str((sample_dir / "input_1.png").resolve()),
                     "ground_truth": str((sample_dir / "ground_truth.png").resolve()),
                     "input_2": str((sample_dir / "input_2.png").resolve()),
                     "prediction": str((sample_dir / "prediction.png").resolve())}
            metadata = {"model": self.model, "mode": self.mode,
                        "sample_index": index, "selected_by": item["selected_by"],
                        **item["metrics"], **paths,
                        "source_paths": item["source_paths"]}
            (sample_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
            records.append(metadata)
            for selection in item["selected_by"]:
                metric = selection.removeprefix(f"{self.label}_")
                selection_dir = self.save_dir / f"{self.label}_{metric}"
                if self.k > 1:
                    selection_dir /= f"rank_{item['ranks'][selection]}"
                selection_dir.mkdir(parents=True, exist_ok=True)
                pointer = {**metadata, "selection_metric": selection,
                           "rank": item["ranks"][selection],
                           "sample_directory": str(sample_dir.resolve())}
                (selection_dir / "metrics.json").write_text(json.dumps(pointer, indent=2) + "\n")
                rows.append({"model": self.model, "mode": self.mode,
                             "selection_metric": selection, "sample_index": index,
                             **item["metrics"], "input_1_path": paths["input_1"],
                             "gt_path": paths["ground_truth"],
                             "input_2_path": paths["input_2"],
                             "prediction_path": paths["prediction"]})
        return rows, records


class BestMetricsTracker(_ExtremaMetricsTracker):
    """Retain the existing publication best-k extrema with an unchanged API."""

    def __init__(self, save_dir, model="model", mode="mode", best_k=1):
        super().__init__(save_dir, model, mode, best_k, METRIC_DIRECTIONS, "best")
        self.best_k = best_k
        self.best = self.extrema

    def save_best(self):
        return self.save()


class WorstMetricsTracker(_ExtremaMetricsTracker):
    """Retain only maximum LPIPS and maximum PCE failure candidates."""

    def __init__(self, save_dir, model="model", mode="mode", worst_k=1):
        super().__init__(save_dir, model, mode, worst_k,
                         WORST_METRIC_DIRECTIONS, "worst")
        self.worst_k = worst_k
        self.worst = self.extrema

    def save_worst(self):
        return self.save()


def merge_best_summaries(output_dir, rows, records):
    """Replace one model/mode's rows in deterministic CSV and JSON summaries."""
    output_dir = Path(output_dir)
    csv_path, json_path = output_dir / "best_samples.csv", output_dir / "best_samples.json"
    old_rows = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            old_rows = list(csv.DictReader(handle))
    keys = {(row["model"], row["mode"]) for row in rows}
    all_rows = [row for row in old_rows if (row["model"], row["mode"]) not in keys] + rows
    all_rows.sort(key=lambda row: (row["model"], row["mode"], row["selection_metric"],
                                   int(row["sample_index"])))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader(); writer.writerows(all_rows)
    old_records = []
    if json_path.is_file():
        old_records = json.loads(json_path.read_text()).get("samples", [])
    all_records = [record for record in old_records
                   if (record["model"], record["mode"]) not in keys] + records
    all_records.sort(key=lambda record: (record["model"], record["mode"],
                                         record["sample_index"]))
    json_path.write_text(json.dumps({"samples": all_records}, indent=2) + "\n")
    return csv_path, json_path


def merge_worst_summaries(output_dir, rows, records):
    """Replace one model/mode's rows in worst-case summaries."""
    return _merge_summaries(output_dir, rows, records, "worst")


def _merge_summaries(output_dir, rows, records, label):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{label}_samples.csv"
    json_path = output_dir / f"{label}_samples.json"
    old_rows = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            old_rows = list(csv.DictReader(handle))
    keys = {(row["model"], row["mode"]) for row in rows}
    all_rows = [row for row in old_rows
                if (row["model"], row["mode"]) not in keys] + rows
    all_rows.sort(key=lambda row: (row["model"], row["mode"],
                                   row["selection_metric"], int(row["sample_index"])))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    old_records = []
    if json_path.is_file():
        old_records = json.loads(json_path.read_text()).get("samples", [])
    all_records = [record for record in old_records
                   if (record["model"], record["mode"]) not in keys] + records
    all_records.sort(key=lambda record: (record["model"], record["mode"],
                                         record["sample_index"]))
    json_path.write_text(json.dumps({"samples": all_records}, indent=2) + "\n")
    return csv_path, json_path
