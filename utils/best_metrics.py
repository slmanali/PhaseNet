"""Memory-bounded best-sample tracking and export for qualitative evaluation."""

import csv
import json
from pathlib import Path

import torch
from torchvision.utils import save_image


METRIC_DIRECTIONS = {"psnr": True, "ssim": True, "lpips": False, "pce": False}
SUMMARY_FIELDS = ("model", "mode", "selection_metric", "sample_index", "psnr",
                  "ssim", "lpips", "pce", "input_1_path", "gt_path",
                  "input_2_path", "prediction_path")


class BestMetricsTracker:
    """Retain only the top-k samples for each metric (batch size must be one)."""

    def __init__(self, save_dir, model="model", mode="mode", best_k=1):
        if best_k < 1:
            raise ValueError("best_k must be at least 1")
        self.save_dir = Path(save_dir)
        self.model, self.mode, self.best_k = model, mode, best_k
        self.best = {name: [] for name in METRIC_DIRECTIONS}

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
        for metric, maximize in METRIC_DIRECTIONS.items():
            entries = self.best[metric]
            # Keep one candidate object per metric; discarded images become collectible.
            entries.append(candidate.copy())
            entries.sort(key=lambda item: ((-item["metrics"][metric] if maximize else
                                            item["metrics"][metric]),
                                           item["sample_index"]))
            del entries[self.best_k:]

    def selections(self):
        return {metric: list(entries) for metric, entries in self.best.items()}

    def save_best(self):
        """Save unique image data, lightweight selection metadata, and global summaries."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        by_index = {}
        for metric, entries in self.best.items():
            for rank, item in enumerate(entries, 1):
                record = by_index.setdefault(item["sample_index"], item)
                record.setdefault("selected_by", []).append(metric)
                record.setdefault("ranks", {})[metric] = rank

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
            for metric in item["selected_by"]:
                selection_dir = self.save_dir / f"best_{metric}"
                if self.best_k > 1:
                    selection_dir /= f"rank_{item['ranks'][metric]}"
                selection_dir.mkdir(parents=True, exist_ok=True)
                pointer = {**metadata, "selection_metric": metric,
                           "rank": item["ranks"][metric],
                           "sample_directory": str(sample_dir.resolve())}
                (selection_dir / "metrics.json").write_text(json.dumps(pointer, indent=2) + "\n")
                rows.append({"model": self.model, "mode": self.mode,
                             "selection_metric": metric, "sample_index": index,
                             **item["metrics"], "input_1_path": paths["input_1"],
                             "gt_path": paths["ground_truth"],
                             "input_2_path": paths["input_2"],
                             "prediction_path": paths["prediction"]})
        return rows, records


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
