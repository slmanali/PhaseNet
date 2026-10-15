"""Targeted launch and checkpoint recovery regression checks."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run_experiments import command


def test_configuration_filter_launches_only_requested_seed(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "tools.run_experiments", "--plan",
         "experiments/computer_optics_revision.json", "--configs",
         "complex-loss-matched-64", "--seeds", "33", "--run-root",
         str(tmp_path / "runs"), "--dry-run"],
        cwd=root, check=True, text=True, capture_output=True)
    assert result.stdout.count("train_complex_safe_baseline.py") == 1
    assert "--seed 33" in result.stdout
    assert "--config-name complex-loss-matched-64" in result.stdout
    assert "unrestricted_real" not in result.stdout


def test_missing_full_checkpoint_preserves_interrupted_run(tmp_path):
    name = "complex-loss-matched-64"
    directory = tmp_path / name / "seed-33"
    directory.mkdir(parents=True)
    metadata = directory / "metadata.json"
    metadata.write_text(json.dumps({"status": "running", "completed_epochs": 21}))
    before = metadata.read_bytes()
    args = SimpleNamespace(dataset_path="DAVIS", deterministic="strict", run_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="NEW --run-root"):
        command("train.py", {}, {"entrypoint": "train.py"}, name, 33, args)
    assert metadata.read_bytes() == before


def test_resume_keeps_total_epoch_target_and_rejects_changed_protocol(tmp_path):
    name = "complex-loss-matched-64"
    directory = tmp_path / name / "seed-33"
    directory.mkdir(parents=True)
    checkpoint = directory / "checkpoint_last.pth"
    checkpoint.touch()
    arguments = dict(seed=33, config_name=name, deterministic="strict", epochs=40)
    (directory / "metadata.json").write_text(json.dumps({"status": "running", "arguments": arguments}))
    args = SimpleNamespace(dataset_path="DAVIS", deterministic="strict", run_root=tmp_path)
    result = command("train.py", {"epochs": 40}, {"entrypoint": "train.py"}, name, 33, args)
    assert result[result.index("--resume") + 1] == str(checkpoint)
    assert result[result.index("--epochs") + 1] == "40"
    with pytest.raises(ValueError, match="epochs"):
        command("train.py", {"epochs": 20}, {"entrypoint": "train.py"}, name, 33, args)
