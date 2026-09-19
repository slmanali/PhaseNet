"""Tests for tools/evaluate_external_snufilm.py (external RIFE / FILM scoring)."""

import csv
import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from tools import evaluate_external_snufilm as ext
from tools.evaluate_external_snufilm import (EXPECTED_SAMPLES, PredictionSetError,
                                              check_completeness, collect_predictions,
                                              evaluate_all, evaluate_mode, format_table,
                                              load_prediction, main, write_outputs)
from utils.metrics import compute_lpips, compute_psnr, compute_ssim
from utils.snufilm import SNU_MODES, SNUFILMTriplets

WIDTH, HEIGHT = 12, 8
N = 5  # small sample count for fast tests; a dedicated test covers 310


def _rgb(path, value, size=(WIDTH, HEIGHT)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)


def make_dataset(root, count, modes=SNU_MODES, size=(WIDTH, HEIGHT)):
    """Synthetic SNU-FILM: eval_modes/test-<mode>.txt with ``count`` triplets each."""
    lines = {mode: [] for mode in modes}
    for mode in modes:
        for index in range(count):
            frames = [root / "test" / mode / f"{index:03d}_{k}.png" for k in range(3)]
            for k, frame in enumerate(frames):
                _rgb(frame, min(255, 20 + index % 200 + k * 10), size)
            lines[mode].append(" ".join(str(f.relative_to(root)) for f in frames))
    (root / "eval_modes").mkdir(parents=True, exist_ok=True)
    for mode in modes:
        (root / "eval_modes" / f"test-{mode}.txt").write_text("\n".join(lines[mode]) + "\n")
    return root


def make_predictions(pred_root, models, count, modes=SNU_MODES, size=(WIDTH, HEIGHT),
                     offset=3):
    """``<pred_root>/<model>/<mode>/<index:05d>_pred.png`` with a fixed offset from GT."""
    for model in models:
        for mode in modes:
            for index in range(count):
                gt_value = min(255, 20 + index % 200 + 10)  # middle frame value
                _rgb(pred_root / model / mode / f"{index:05d}_pred.png",
                     min(255, gt_value + offset), size)
    return pred_root


class StubLPIPS(torch.nn.Module):
    """Cheap stand-in with the LPIPS call signature (mean absolute difference)."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pred, target):
        self.calls += 1
        return (pred - target).abs().mean(dim=(1, 2, 3), keepdim=True)


@pytest.fixture
def small_setup(tmp_path):
    root = make_dataset(tmp_path / "SNU-FILM", N)
    preds = make_predictions(tmp_path / "outputs", ("rife", "film"), N)
    return root, preds


# --------------------------------------------------------------------------- #
# Completeness checks
# --------------------------------------------------------------------------- #
def test_exactly_310_samples_per_mode_pass_completeness(tmp_path):
    preds = make_predictions(tmp_path / "outputs", ("rife", "film"), EXPECTED_SAMPLES,
                             size=(2, 2))
    collected = check_completeness(preds, ("rife", "film"))
    for model in ("rife", "film"):
        for mode in SNU_MODES:
            indices = list(collected[model][mode])
            assert len(indices) == EXPECTED_SAMPLES == 310
            assert indices == list(range(310))
            assert collected[model][mode][309].name == "00309_pred.png"


def test_309_samples_is_rejected(tmp_path):
    preds = make_predictions(tmp_path / "outputs", ("rife",), EXPECTED_SAMPLES - 1,
                             size=(2, 2))
    with pytest.raises(PredictionSetError, match=r"expected exactly indices 0\.\.309"):
        collect_predictions(preds / "rife" / "easy")


def test_311_samples_is_rejected(tmp_path):
    preds = make_predictions(tmp_path / "outputs", ("rife",), EXPECTED_SAMPLES + 1,
                             size=(2, 2))
    with pytest.raises(PredictionSetError, match=r"unexpected 1 index/indices: 310"):
        collect_predictions(preds / "rife" / "easy")


def test_missing_prediction_file_is_reported(tmp_path):
    preds = make_predictions(tmp_path / "outputs", ("rife",), N)
    (preds / "rife" / "hard" / "00002_pred.png").unlink()
    with pytest.raises(PredictionSetError, match=r"missing 1 index/indices: 2"):
        collect_predictions(preds / "rife" / "hard", expected_samples=N)


def test_missing_prediction_directory_is_reported(tmp_path):
    preds = make_predictions(tmp_path / "outputs", ("rife",), N)
    with pytest.raises(PredictionSetError, match="prediction directory not found"):
        check_completeness(preds, ("rife", "film"), expected_samples=N)


def test_missing_index_gap_is_reported_even_when_count_matches(tmp_path):
    preds = make_predictions(tmp_path / "outputs", ("film",), N)
    mode_dir = preds / "film" / "medium"
    # Rename index 1 -> 7: still N files, but a gap at 1 and an unexpected 7.
    (mode_dir / "00001_pred.png").rename(mode_dir / "00007_pred.png")
    with pytest.raises(PredictionSetError) as error:
        collect_predictions(mode_dir, expected_samples=N)
    message = str(error.value)
    assert "missing 1 index/indices: 1" in message
    assert "unexpected 1 index/indices: 7" in message


def test_duplicate_index_is_reported(tmp_path):
    preds = make_predictions(tmp_path / "outputs", ("rife",), N)
    mode_dir = preds / "rife" / "extreme"
    # Same official index written with a different zero-padding.
    _rgb(mode_dir / "3_pred.png", 0)
    with pytest.raises(PredictionSetError, match=r"duplicate prediction index.*\n.*index 3"):
        collect_predictions(mode_dir, expected_samples=N)


def test_completeness_reports_all_broken_directories_at_once(tmp_path):
    preds = make_predictions(tmp_path / "outputs", ("rife", "film"), N)
    (preds / "rife" / "easy" / "00000_pred.png").unlink()
    (preds / "film" / "extreme" / "00004_pred.png").unlink()
    with pytest.raises(PredictionSetError) as error:
        check_completeness(preds, ("rife", "film"), expected_samples=N)
    message = str(error.value)
    assert "rife/easy" in message.replace("\\", "/")
    assert "film/extreme" in message.replace("\\", "/")


def test_non_prediction_files_are_ignored(tmp_path):
    preds = make_predictions(tmp_path / "outputs", ("rife",), N)
    mode_dir = preds / "rife" / "easy"
    (mode_dir / "manifest.json").write_text("{}")
    (mode_dir / "00000_truth.png").write_bytes(b"")
    assert list(collect_predictions(mode_dir, expected_samples=N)) == list(range(N))


# --------------------------------------------------------------------------- #
# Loading / resolution checks
# --------------------------------------------------------------------------- #
def test_load_prediction_returns_float_rgb_in_unit_range(tmp_path):
    path = tmp_path / "00000_pred.png"
    _rgb(path, 51)
    tensor = load_prediction(path, (3, HEIGHT, WIDTH))
    assert tensor.shape == (1, 3, HEIGHT, WIDTH)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.full_like(tensor, 51 / 255))


def test_wrong_resolution_raises_instead_of_resizing(tmp_path):
    path = tmp_path / "00000_pred.png"
    _rgb(path, 51, size=(WIDTH + 2, HEIGHT))
    with pytest.raises(ValueError, match="resolution mismatch.*never resized"):
        load_prediction(path, (3, HEIGHT, WIDTH))


def test_non_rgb_prediction_is_rejected(tmp_path):
    path = tmp_path / "00000_pred.png"
    Image.new("L", (WIDTH, HEIGHT), 7).save(path)
    with pytest.raises(ValueError, match="expected an RGB image"):
        load_prediction(path, (3, HEIGHT, WIDTH))


def test_wrong_resolution_prediction_fails_mode_evaluation(small_setup):
    root, preds = small_setup
    bad = preds / "rife" / "hard" / "00003_pred.png"
    _rgb(bad, 60, size=(WIDTH * 2, HEIGHT * 2))
    dataset = SNUFILMTriplets(root, "hard")
    predictions = collect_predictions(preds / "rife" / "hard", expected_samples=N)
    with pytest.raises(ValueError, match="00003_pred.png: resolution mismatch"):
        evaluate_mode(dataset, predictions, StubLPIPS(), torch.device("cpu"),
                      "RIFE", "hard", progress=False)


# --------------------------------------------------------------------------- #
# Metric path
# --------------------------------------------------------------------------- #
def test_evaluate_mode_uses_shared_metrics_and_reuses_lpips(small_setup):
    root, preds = small_setup
    dataset = SNUFILMTriplets(root, "easy")
    predictions = collect_predictions(preds / "rife" / "easy", expected_samples=N)
    stub = StubLPIPS()
    summary, rows = evaluate_mode(dataset, predictions, stub, torch.device("cpu"),
                                  "RIFE", "easy", progress=False)

    assert stub.calls == N  # one shared instance, called once per sample
    assert summary["samples"] == N
    assert [row["sample_index"] for row in rows] == list(range(N))

    # Recompute sample 0 directly with the shared utils.metrics functions.
    truth = dataset[0]["inter"].unsqueeze(0)
    pred = load_prediction(predictions[0], truth.shape[1:])
    assert rows[0]["psnr"] == pytest.approx(compute_psnr(pred, truth))
    assert rows[0]["ssim"] == pytest.approx(compute_ssim(pred, truth))
    assert rows[0]["lpips"] == pytest.approx(compute_lpips(pred, truth, stub))
    # Constant 3/255 offset -> known PSNR.
    assert rows[0]["psnr"] == pytest.approx(20 * math.log10(255 / 3), rel=1e-4)
    # Mode value is the arithmetic mean of per-sample values.
    assert summary["psnr"] == pytest.approx(np.mean([r["psnr"] for r in rows]))
    assert summary["lpips"] == pytest.approx(np.mean([r["lpips"] for r in rows]))


def test_summary_and_per_sample_rows_have_no_pce(small_setup):
    root, preds = small_setup
    summaries, per_sample, _ = evaluate_all(root, preds, ["rife", "film"], torch.device("cpu"),
                                            expected_samples=N, progress=False,
                                            loss_fn_lpips=StubLPIPS())
    assert [(s["model"], s["mode"]) for s in summaries] == [
        (label, mode) for label in ("RIFE", "FILM") for mode in SNU_MODES]
    assert all(set(s) == set(ext.SUMMARY_FIELDS) for s in summaries)
    assert all(set(r) == set(ext.PER_SAMPLE_FIELDS) for r in per_sample)
    assert len(per_sample) == 2 * len(SNU_MODES) * N
    assert "pce" not in summaries[0] and "pce" not in per_sample[0]


def test_write_outputs_produces_expected_files(tmp_path, small_setup):
    root, preds = small_setup
    summaries, per_sample, _ = evaluate_all(root, preds, ["rife", "film"], torch.device("cpu"),
                                            expected_samples=N, progress=False,
                                            loss_fn_lpips=StubLPIPS())
    out = tmp_path / "results" / "snufilm" / "native"
    csv_path, json_path, per_sample_path = write_outputs(summaries, per_sample, out)

    assert csv_path.name == "external_metrics_snufilm.csv"
    assert json_path.name == "external_metrics_snufilm.json"
    assert per_sample_path.name == "external_metrics_per_sample.csv"

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["model", "mode", "samples", "psnr", "ssim", "lpips"]
        rows = list(reader)
    assert rows[0]["model"] == "RIFE" and rows[0]["mode"] == "easy"
    assert rows[4]["model"] == "FILM" and rows[4]["mode"] == "easy"
    assert int(rows[0]["samples"]) == N

    with per_sample_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["model", "mode", "sample_index", "psnr", "ssim", "lpips"]
        assert len(list(reader)) == 2 * 4 * N

    payload = json.loads(json_path.read_text())
    assert len(payload["results"]) == 8
    assert "pce" in payload["metadata"]  # documents why PCE is absent

    table = format_table(summaries)
    assert table.splitlines()[0].startswith("Model")
    assert "RIFE     | easy" in table


def test_evaluate_all_requires_official_list_length(tmp_path):
    """310 predictions but a truncated official list must be rejected."""
    root = make_dataset(tmp_path / "SNU-FILM", N, size=(2, 2))
    preds = make_predictions(tmp_path / "outputs", ("rife",), EXPECTED_SAMPLES, size=(2, 2))
    with pytest.raises(PredictionSetError, match=f"contains {N} triplets, expected 310"):
        evaluate_all(root, preds, ["rife"], torch.device("cpu"), progress=False,
                     loss_fn_lpips=StubLPIPS())


def test_evaluate_all_fails_completeness_before_touching_dataset(tmp_path, small_setup):
    root, preds = small_setup
    (preds / "film" / "medium" / "00001_pred.png").unlink()
    with pytest.raises(PredictionSetError, match="completeness check failed"):
        evaluate_all(root, preds, ["rife", "film"], torch.device("cpu"),
                     expected_samples=N, progress=False, loss_fn_lpips=StubLPIPS())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_end_to_end_with_stubbed_lpips(tmp_path, small_setup, monkeypatch, capsys):
    root, preds = small_setup
    monkeypatch.setattr(ext, "EXPECTED_SAMPLES", N)
    monkeypatch.setattr(ext, "build_lpips", lambda device: StubLPIPS())
    out = tmp_path / "results"
    main(["--dataset-path", str(root), "--predictions-dir", str(preds),
          "--models", "film", "rife", "--device", "cpu", "--output-dir", str(out)])
    captured = capsys.readouterr().out
    assert "Model" in captured and "FILM" in captured and "RIFE" in captured
    with (out / "external_metrics_snufilm.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["model"] for r in rows] == ["FILM"] * 4 + ["RIFE"] * 4


def test_cli_rejects_unknown_model(small_setup):
    root, preds = small_setup
    with pytest.raises(SystemExit):
        main(["--dataset-path", str(root), "--predictions-dir", str(preds),
              "--models", "xvfi", "--device", "cpu"])


@pytest.mark.slow
def test_real_lpips_alexnet_instance_is_built_once(small_setup, monkeypatch):
    """Exercises the real lpips.LPIPS(net='alex') construction path."""
    pytest.importorskip("lpips")
    root = make_dataset(small_setup[0].parent / "SNU-FILM-64", N, size=(64, 64))
    preds = make_predictions(small_setup[1].parent / "outputs-64", ("rife", "film"), N,
                             size=(64, 64))
    created = []
    real_build = ext.build_lpips

    def counting_build(device):
        created.append(device)
        return real_build(device)

    monkeypatch.setattr(ext, "build_lpips", counting_build)
    summaries, per_sample, _ = ext.evaluate_all(root, preds, ["rife", "film"],
                                                torch.device("cpu"), expected_samples=N,
                                                progress=False)
    assert len(created) == 1
    assert len(per_sample) == 2 * 4 * N
    assert all(0.0 <= row["lpips"] <= 1.0 for row in per_sample)
