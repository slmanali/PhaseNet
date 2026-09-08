import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import torch

from utils.best_metrics import BestMetricsTracker, merge_best_summaries


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
