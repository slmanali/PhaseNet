import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from utils.snufilm import (SNUFILMTriplets, SNU_MODES, snufilm_modes,
                           write_snufilm_results)


def _image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (value, value, value)).save(path)


@pytest.fixture
def snu_root(tmp_path):
    root = tmp_path / "SNU-FILM"
    frames = [root / "test" / "GOPRO_test" / f"frame{i}.png" for i in range(3)]
    for index, frame in enumerate(frames):
        _image(frame, 10 + index * 50)
    lists = root / "eval_modes"
    lists.mkdir(parents=True)
    relative = " ".join(str(path.relative_to(root)) for path in frames)
    for mode in SNU_MODES:
        (lists / f"test-{mode}.txt").write_text(relative + "\n")
    return root, frames


@pytest.mark.parametrize("mode", SNU_MODES)
def test_selects_each_official_mode_and_relative_paths(snu_root, mode):
    root, frames = snu_root
    dataset = SNUFILMTriplets(root, mode)
    assert dataset.list_file == root / "eval_modes" / f"test-{mode}.txt"
    assert dataset.triplets == [tuple(path.resolve() for path in frames)]


def test_absolute_paths_and_root_level_legacy_list(snu_root):
    root, frames = snu_root
    (root / "eval_modes" / "test-easy.txt").unlink()
    (root / "test-easy.txt").write_text(" ".join(map(str, frames)) + "\n")
    dataset = SNUFILMTriplets(root, "easy")
    assert dataset.list_file == root / "test-easy.txt"
    assert dataset.triplets[0][1] == frames[1].resolve()


def test_start_middle_end_order_and_resize(snu_root):
    dataset = SNUFILMTriplets(snu_root[0], "hard", "256")
    sample = dataset[0]
    assert tuple(sample) == ("start", "inter", "end")
    assert sample["start"].shape == (3, 256, 256)
    assert sample["start"].mean() < sample["inter"].mean() < sample["end"].mean()


def test_missing_image_fails_during_validation(snu_root):
    snu_root[1][1].unlink()
    with pytest.raises(FileNotFoundError, match="referenced SNU-FILM frame"):
        SNUFILMTriplets(snu_root[0], "medium")


def test_all_modes_are_separate_and_paper_files_are_written(tmp_path):
    assert snufilm_modes("all") == SNU_MODES
    results = {
        mode: {"samples": 1, "l1": .1, "mse": .01, "psnr": 20 + index,
               "ssim": .8, "lpips": .2, "pce": .3}
        for index, mode in enumerate(SNU_MODES)
    }
    write_snufilm_results(results, tmp_path, "PhaseNet-big")
    with (tmp_path / "metrics_snufilm.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mode"] for row in rows] == list(SNU_MODES)
    payload = json.loads((tmp_path / "metrics_snufilm.json").read_text())
    assert len(payload["results"]) == 4


@pytest.mark.parametrize("script", ("test.py", "test_complex_safe_baseline.py"))
def test_snufilm_branch_returns_before_davis_split(script):
    source = (Path(__file__).parents[1] / script).read_text()
    main = source[source.rfind("def main():"):]
    assert main.index('if args.dataset_type == "snufilm":') < main.index(
        "davis_root = resolve_davis_root"
    )
    assert main.index("        return") < main.index("davis_root = resolve_davis_root")
