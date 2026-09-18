from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools import run_pretrained_snufilm
from tools.run_pretrained_snufilm import RIFEBackend, parse_indices, run, validate_paths


def _make_dataset(root):
    frames = root / "test"
    frames.mkdir(parents=True)
    for index, value in enumerate((0, 100, 200)):
        Image.new("RGB", (7, 5), (value,) * 3).save(frames / f"{index}.png")
    lists = root / "eval_modes"
    lists.mkdir()
    (lists / "test-hard.txt").write_text("test/0.png test/1.png test/2.png\n")


def test_run_uses_endpoint_frames_and_official_index(tmp_path):
    root = tmp_path / "SNU-FILM"
    _make_dataset(root)
    calls = []

    def backend(first, second):
        calls.append((np.asarray(first).mean(), np.asarray(second).mean()))
        return np.full((5, 7, 3), 42, dtype=np.uint8)

    args = Namespace(snu_root=root, snu_mode="hard", sample_indices=None,
                     output_dir=tmp_path / "out", model="rife", overwrite=False)
    completed = run(args, backend)
    output = tmp_path / "out/rife/hard/00000_pred.png"
    assert calls == [(0, 200)]
    assert completed[0]["sample_index"] == 0
    assert np.asarray(Image.open(output)).mean() == 42
    assert (tmp_path / "out/rife/manifest.json").is_file()


def test_parse_indices_validates_input():
    assert parse_indices("0,4,12") == [0, 4, 12]
    with pytest.raises(Exception, match="unique non-negative"):
        parse_indices("2,2")


def test_rife_backend_uses_upstream_inference_model(tmp_path, monkeypatch):
    imported = []

    class FlowNet:
        def to(self, device):
            self.device = device

    class Model:
        def __init__(self):
            self.flownet = FlowNet()

        def load_model(self, checkpoint, rank):
            self.loaded = (checkpoint, rank)

        def eval(self):
            pass

        def device(self):
            pass

    class Module:
        pass

    Module.Model = Model

    def fake_import(repo, module_name):
        imported.append((repo, module_name))
        return Module

    monkeypatch.setattr(run_pretrained_snufilm, "_import_from_repo", fake_import)
    backend = RIFEBackend(tmp_path / "RIFE", tmp_path / "train_log", "cpu")

    assert imported == [(tmp_path / "RIFE", "model.RIFE_HDv3")]
    assert backend.model.loaded == (str((tmp_path / "train_log").resolve()), -1)
    assert backend.model.flownet.device.type == "cpu"


def test_rife_backend_supports_model_code_in_checkpoint(tmp_path, monkeypatch):
    imported = []

    class Model:
        def load_model(self, checkpoint, rank):
            self.loaded = (checkpoint, rank)

        def eval(self):
            pass

        def device(self):
            pass

    class Module:
        pass

    Module.Model = Model

    def fake_import(path, module_name, dependency_paths=()):
        imported.append((path, module_name, dependency_paths))
        if module_name == "model.RIFE_HDv3":
            error = ModuleNotFoundError("No module named 'model.RIFE_HDv3'")
            error.name = "model.RIFE_HDv3"
            raise error
        return Module

    monkeypatch.setattr(run_pretrained_snufilm, "_import_from_repo", fake_import)
    checkpoint = tmp_path / "train_log"
    backend = RIFEBackend(tmp_path / "RIFE", checkpoint, "cpu")

    assert imported == [
        (tmp_path / "RIFE", "model.RIFE_HDv3", ()),
        (tmp_path, "train_log.RIFE_HDv3", (tmp_path / "RIFE",)),
    ]
    assert backend.model.loaded == (str(checkpoint.resolve()), -1)


def test_rife_checkpoint_module_can_import_checkout_dependencies(tmp_path):
    repo = tmp_path / "RIFE"
    model_package = repo / "model"
    model_package.mkdir(parents=True)
    (model_package / "__init__.py").write_text("", encoding="utf-8")
    (model_package / "warplayer.py").write_text("warp = object()\n", encoding="utf-8")
    checkpoint = tmp_path / "train_log"
    checkpoint.mkdir()
    (checkpoint / "RIFE_HDv3.py").write_text(
        "from train_log.IFNet_HDv3 import IFNet\n"
        "from model.warplayer import warp\n"
        "class Model:\n"
        "    def load_model(self, checkpoint, rank): pass\n"
        "    def eval(self): pass\n"
        "    def device(self): pass\n",
        encoding="utf-8",
    )
    (checkpoint / "IFNet_HDv3.py").write_text("IFNet = object()\n", encoding="utf-8")

    backend = RIFEBackend(repo, checkpoint, "cpu")

    assert backend.model is not None


def test_rife_backend_explains_checkpoint_version_mismatch(tmp_path, monkeypatch):
    class Model:
        def load_model(self, checkpoint, rank):
            raise RuntimeError("missing and unexpected keys")

    class Module:
        pass

    Module.Model = Model
    monkeypatch.setattr(run_pretrained_snufilm, "_import_from_repo",
                        lambda repo, module_name: Module)

    with pytest.raises(RuntimeError, match="checkpoint is incompatible") as error:
        RIFEBackend(tmp_path / "RIFE", tmp_path / "train_log", "cpu")

    assert "revision matching the downloaded weights" in str(error.value)


def test_validate_paths_reports_each_missing_input(tmp_path):
    class Parser:
        def error(self, message):
            raise ValueError(message)

    args = Namespace(repo=tmp_path / "RIFE", checkpoint=tmp_path / "train_log",
                     snu_root=tmp_path / "SNU-FILM", output_dir=tmp_path / "output")
    with pytest.raises(ValueError) as error:
        validate_paths(Parser(), args)
    message = str(error.value)
    assert f"--repo directory not found: {args.repo}" in message
    assert f"--checkpoint directory not found: {args.checkpoint}" in message
    assert f"--snu-root directory not found: {args.snu_root}" in message
    assert "Repository clones do not include pretrained weights" in message
    assert "creating an empty checkpoint directory is not sufficient" in message
    assert "extracted train_log directory" in message


def test_validate_paths_expands_and_resolves_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for directory in ("repo", "checkpoint", "SNU-FILM"):
        (tmp_path / directory).mkdir()
    args = Namespace(repo=Path("~/repo"), checkpoint=Path("~/checkpoint"),
                     snu_root=Path("~/SNU-FILM"), output_dir=Path("~/output"))

    validate_paths(None, args)

    assert args.repo == (tmp_path / "repo").resolve()
    assert args.checkpoint == (tmp_path / "checkpoint").resolve()
    assert args.snu_root == (tmp_path / "SNU-FILM").resolve()
    assert args.output_dir == (tmp_path / "output").resolve()
