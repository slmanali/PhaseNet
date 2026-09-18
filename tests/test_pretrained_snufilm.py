from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.run_pretrained_snufilm import parse_indices, run


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
