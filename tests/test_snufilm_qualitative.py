import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import torch

from utils.best_metrics import (BestMetricsTracker, WorstMetricsTracker,
                                merge_best_summaries, merge_worst_summaries)


def _load_tool():
    path = Path(__file__).parents[1] / "tools" / "export_snufilm_qualitative.py"
    spec = importlib.util.spec_from_file_location("export_snufilm_qualitative", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _update(tracker, index, values):
    image = torch.full((1, 3, 4, 5), index / 10)
    tracker.update(index, *values, image, image, image, image,
                   {"input_1": f"source/{index}/1.png",
                    "ground_truth": f"source/{index}/2.png",
                    "input_2": f"source/{index}/3.png"})


def test_metric_directions_best_k_one_and_true_index(tmp_path):
    tracker = BestMetricsTracker(tmp_path / "model" / "easy", "model", "easy", 1)
    _update(tracker, 17, (10, .9, .4, .3))
    _update(tracker, 42, (20, .8, .2, .5))
    selected = tracker.selections()
    assert selected["psnr"][0]["sample_index"] == 42
    assert selected["ssim"][0]["sample_index"] == 17
    assert selected["lpips"][0]["sample_index"] == 42
    assert selected["pce"][0]["sample_index"] == 17


def test_best_k_three_duplicate_export_and_summaries(tmp_path):
    mode_dir = tmp_path / "model" / "hard"
    tracker = BestMetricsTracker(mode_dir, "model", "hard", 3)
    for index in range(5):
        # Index four wins all four metrics, exercising de-duplication.
        _update(tracker, index, (index, index, 10 - index, 20 - index))
    assert [x["sample_index"] for x in tracker.selections()["psnr"]] == [4, 3, 2]
    rows, records = tracker.save_best()
    csv_path, json_path = merge_best_summaries(tmp_path, rows, records)
    assert len(rows) == 12
    assert len(list((mode_dir / "samples").iterdir())) == 3
    record = next(item for item in records if item["sample_index"] == 4)
    assert set(record["selected_by"]) == {"psnr", "ssim", "lpips", "pce"}
    assert record["source_paths"]["ground_truth"] == "source/4/2.png"
    with csv_path.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 12
    assert set(csv_rows[0]) == {"model", "mode", "selection_metric", "sample_index",
                               "psnr", "ssim", "lpips", "pce", "input_1_path",
                               "gt_path", "input_2_path", "prediction_path"}
    payload = json.loads(json_path.read_text())
    assert len(payload["samples"]) == 3
    for name in ("input_1.png", "ground_truth.png", "input_2.png", "prediction.png"):
        assert (mode_dir / "samples" / "4" / name).is_file()


def test_generated_command_never_uses_save_all(tmp_path):
    tool = _load_tool()
    args = Namespace(snu_root=tmp_path / "SNU", output_dir=tmp_path / "out",
                     image_size="256", best_k=1, device="cuda:0")
    command = tool.build_command(args, "phasenet_default", tmp_path / "model.pth")
    assert "--save-all" not in command
    assert command[command.index("--best-k") + 1] == "1"


def test_generated_command_selects_only_requested_official_indices(tmp_path):
    tool = _load_tool()
    args = Namespace(snu_root=tmp_path / "SNU", output_dir=tmp_path / "out",
                     image_size="256", best_k=1, device=None, mode="hard",
                     sample_indices="36,38,39,139")
    command = tool.build_command(args, "phasenet_big", tmp_path / "model.pth")
    assert command[command.index("--snu-mode") + 1] == "hard"
    assert command[command.index("--sample-indices") + 1] == "36,38,39,139"


def test_worst_k_one_selects_maximum_lpips_and_pce(tmp_path):
    tracker = WorstMetricsTracker(tmp_path / "model" / "easy", "model", "easy", 1)
    _update(tracker, 17, (30, .9, .2, .7))
    _update(tracker, 42, (20, .8, .8, .3))
    assert tracker.selections()["lpips"][0]["sample_index"] == 42
    assert tracker.selections()["pce"][0]["sample_index"] == 17


def test_worst_k_three_duplicate_export_and_summary_files(tmp_path):
    mode_dir = tmp_path / "model" / "hard"
    tracker = WorstMetricsTracker(mode_dir, "model", "hard", 3)
    for index in range(5):
        _update(tracker, index, (20, .8, index, index))
    assert [item["sample_index"] for item in tracker.selections()["lpips"]] == [4, 3, 2]
    rows, records = tracker.save_worst()
    csv_path, json_path = merge_worst_summaries(tmp_path, rows, records)
    assert len(rows) == 6
    assert len(records) == 3
    assert set(records[0]["selected_by"]) == {"worst_lpips", "worst_pce"}
    assert {path.name for path in (mode_dir / "samples").iterdir()} == {"2", "3", "4"}
    assert not list(mode_dir.glob("*_pred.png"))  # no save-all artifacts
    with csv_path.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 6
    assert len(json.loads(json_path.read_text())["samples"]) == 3


def test_worst_command_and_candidate_union(tmp_path):
    tool = _load_tool()
    args = Namespace(snu_root=tmp_path / "SNU", output_dir=tmp_path / "out",
                     image_size="256", best_k=1, worst_k=3, device=None,
                     mode="easy", sample_indices=None)
    command = tool.build_command(args, "complex_full", tmp_path / "model.pth")
    assert command[command.index("--worst-k") + 1] == "3"
    assert "--best-k" not in command and "--save-all" not in command
    args.output_dir.mkdir()
    fields = ("model", "mode", "selection_metric", "sample_index", "psnr",
              "ssim", "lpips", "pce", "input_1_path", "gt_path",
              "input_2_path", "prediction_path")
    with (args.output_dir / "worst_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for model, metric, index in (("a", "worst_lpips", 12),
                                     ("b", "worst_pce", 12),
                                     ("a", "worst_pce", 45)):
            writer.writerow({key: ({"model": model, "mode": "easy",
                                    "selection_metric": metric,
                                    "sample_index": index}.get(key, 0)) for key in fields})
    json_path, csv_path = tool.write_candidate_union(args.output_dir)
    assert json.loads(json_path.read_text()) == {
        "easy": [12, 45], "medium": [], "hard": [], "extreme": []}
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["selected_by_model"] == "a;b"
    assert rows[0]["selected_by_metric"] == "worst_lpips;worst_pce"
