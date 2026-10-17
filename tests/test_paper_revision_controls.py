"""Regression checks for recovery and the shared evaluation protocol."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.run_experiments import command
from utils.metrics import compute_complex_pce, compute_real_pce
from utils.reproducibility import (capture_rng_state, load_checkpoint,
                                   make_data_generator, restore_rng_state,
                                   save_training_checkpoint)


def test_real_and_complex_pce_use_same_raw_amplitude_mask():
    raw = torch.tensor([[[[1e-5, .1]]]])
    scale = torch.tensor([[[[1e-5, 10.0]]]])
    zero = torch.zeros_like(raw)
    angle = torch.tensor([[[[2.0, .25]]]])
    residual = torch.zeros(1, 1, 1, 2)
    truth = [residual, torch.cat((raw / scale, zero), dim=1)]
    pred = [residual, torch.cat((raw / scale, angle), dim=1)]
    real_error = compute_real_pce(truth, pred, amp_scales=[scale])
    truth_r = [residual, torch.cat((raw, torch.ones_like(raw)), dim=1)]
    truth_i = [residual, torch.cat((zero, zero), dim=1)]
    pred_r = [residual, torch.cat((raw, torch.cos(angle)), dim=1)]
    pred_i = [residual, torch.cat((zero, torch.sin(angle)), dim=1)]
    complex_error = compute_complex_pce(truth_r, truth_i, pred_r, pred_i)
    assert real_error == pytest.approx(.25)
    assert complex_error == pytest.approx(real_error)


def test_real_evaluator_averages_three_colour_channels(monkeypatch):
    import test as evaluator

    class Pyramid:
        def __init__(self, **kwargs): pass
        def build(self, *args, **kwargs): return None
        def reconstruct(self, *args, **kwargs): return torch.zeros(1, 4, 4)

    truth = [torch.zeros(1, 1, 4, 4),
             torch.cat((torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)), dim=1)]
    prediction = [truth[0], torch.cat((torch.ones(1, 1, 4, 4),
                                      torch.full((1, 1, 4, 4), .3)), dim=1)]
    monkeypatch.setattr(evaluator, "SCFpyr_PyTorch", Pyramid)
    monkeypatch.setattr(evaluator, "get_input", lambda _: (None, truth, [torch.ones(1, 1, 4, 4)]))
    monkeypatch.setattr(evaluator, "output_convert", lambda *a, **k: None)
    monkeypatch.setattr(evaluator.lpips, "LPIPS", lambda **_: nn.Identity())
    monkeypatch.setattr(evaluator, "shared_lpips", lambda *a: 0.0)
    frame = torch.zeros(1, 3, 4, 4)
    result = evaluator.evaluate(lambda _: prediction,
                                [{"start": frame, "inter": frame, "end": frame}],
                                torch.device("cpu"))
    assert result["pce"] == pytest.approx(.3)


def test_missing_resume_checkpoint_preserves_incomplete_run(tmp_path):
    directory = tmp_path / "complex-loss-matched-64" / "seed-33"
    directory.mkdir(parents=True)
    metadata = directory / "metadata.json"
    metadata.write_text(json.dumps({"status": "running", "completed_epochs": 21}))
    args = SimpleNamespace(dataset_path="DAVIS", deterministic="strict", run_root=tmp_path)
    before = metadata.read_bytes()
    with pytest.raises(FileNotFoundError, match="NEW --run-root"):
        command("train.py", {}, {"entrypoint": "train.py"},
                "complex-loss-matched-64", 33, args)
    assert metadata.read_bytes() == before


def test_resume_uses_full_checkpoint_and_checks_configuration(tmp_path):
    name = "complex-loss-matched-64"
    directory = tmp_path / name / "seed-33"
    directory.mkdir(parents=True)
    checkpoint = directory / "checkpoint_last.pth"
    checkpoint.touch()
    arguments = dict(seed=33, config_name=name, deterministic="strict", epochs=40)
    metadata = directory / "metadata.json"
    metadata.write_text(json.dumps({"status": "running", "arguments": arguments}))
    args = SimpleNamespace(dataset_path="DAVIS", deterministic="strict", run_root=tmp_path)
    result = command("train.py", {"epochs": 40}, {"entrypoint": "train.py"}, name, 33, args)
    assert result[result.index("--resume") + 1] == str(checkpoint)
    with pytest.raises(ValueError, match="epochs"):
        command("train.py", {"epochs": 50}, {"entrypoint": "train.py"}, name, 33, args)


def test_checkpoint_load_for_inference_does_not_change_rng(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "checkpoint.pth"
    save_training_checkpoint(path, model, optimizer, None, 1, 4, {}, make_data_generator(1))
    torch.manual_seed(92)
    before = torch.get_rng_state().clone()
    load_checkpoint(path, model, restore_rng=False)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_restore_rng_after_loading_checkpoint_onto_cuda():
    generator = make_data_generator(33)
    state = capture_rng_state(generator)
    expected_cpu = state["torch_cpu"].clone()
    expected_loader = state["data_loader_generator"].clone()
    for key in ("torch_cpu", "data_loader_generator"):
        state[key] = state[key].cuda()
    state["torch_cuda"] = [value.cuda() for value in state["torch_cuda"]]
    restore_rng_state(state, generator)
    assert torch.equal(torch.get_rng_state(), expected_cpu)
    assert torch.equal(generator.get_state(), expected_loader)
