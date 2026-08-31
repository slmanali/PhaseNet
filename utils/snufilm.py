"""SNU-FILM official-triplet loading and evaluation reporting utilities."""

import csv
import json
import shlex
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

SNU_MODES = ("easy", "medium", "hard", "extreme")


class SNUFILMTriplets(Dataset):
    """Load exactly the triplets specified by an official SNU-FILM list."""

    def __init__(self, root, mode, image_size="native"):
        if mode not in SNU_MODES:
            raise ValueError(f"mode must be one of {SNU_MODES}, got {mode!r}")
        if image_size not in ("native", "256"):
            raise ValueError("image_size must be 'native' or '256'")
        self.root = Path(root).expanduser().resolve()
        self.mode = mode
        self.image_size = image_size
        self.list_file = self._find_list()
        self.triplets = self._read_and_validate()

    def _find_list(self):
        candidates = (
            self.root / "eval_modes" / f"test-{self.mode}.txt",
            self.root / f"test-{self.mode}.txt",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"SNU-FILM {self.mode} list not found; checked: "
            + ", ".join(map(str, candidates))
        )

    def _resolve_image(self, value):
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        # Official lists are normally relative to the dataset root.  The second
        # candidate also supports lists whose paths are relative to the list.
        candidates = (self.root / path, self.list_file.parent / path)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[0].resolve()

    def _read_and_validate(self):
        triplets = []
        missing = []
        seen = set()
        for line_number, raw_line in enumerate(
            self.list_file.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = shlex.split(line)
            if len(fields) != 3:
                raise ValueError(
                    f"{self.list_file}:{line_number}: expected three image paths, "
                    f"found {len(fields)}"
                )
            triplet = tuple(self._resolve_image(field) for field in fields)
            if len(set(triplet)) != 3:
                raise ValueError(
                    f"{self.list_file}:{line_number}: start, middle, and end "
                    "must be different files"
                )
            if triplet in seen:
                raise ValueError(
                    f"{self.list_file}:{line_number}: duplicate triplet: {triplet}"
                )
            seen.add(triplet)
            missing.extend(path for path in triplet if not path.is_file())
            triplets.append(triplet)
        if missing:
            preview = "\n".join(f"  {path}" for path in missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} referenced SNU-FILM frame(s) do not exist:\n{preview}"
            )
        if not triplets:
            raise ValueError(f"No triplets found in {self.list_file}")
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        tensors = []
        for path in self.triplets[index]:
            with Image.open(path) as image:
                tensor = TF.pil_to_tensor(image.convert("RGB")).float() / 255.0
            if self.image_size == "256":
                tensor = TF.resize(tensor, [256, 256], antialias=True)
            tensors.append(tensor)
        return {"start": tensors[0], "inter": tensors[1], "end": tensors[2]}

    @property
    def input_resolution(self):
        if self.image_size == "256":
            return "256x256 (explicit resize)"
        with Image.open(self.triplets[0][0]) as image:
            return f"native ({image.width}x{image.height} for first frame)"

    def print_validation_summary(self):
        print(f"SNU-FILM root: {self.root}")
        print(f"Mode: {self.mode}")
        print(f"Triplet-list file: {self.list_file}")
        print(f"Triplet count: {len(self)}")
        print(f"Input resolution: {self.input_resolution}")
        print("First 3 resolved triplets:")
        for triplet in self.triplets[:3]:
            print("  " + " | ".join(map(str, triplet)))
        if not 250 <= len(self) <= 400:
            print("WARNING: official SNU-FILM modes normally contain approximately 310 triplets.")


def snufilm_modes(selected):
    return SNU_MODES if selected == "all" else (selected,)


def write_snufilm_results(results, output_dir, model_name):
    """Merge this run into paper-ready CSV/JSON files and print its mode table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_snufilm.csv"
    fields = ("model", "mode", "samples", "psnr", "ssim", "lpips", "pce")
    rows = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    replacements = {(model_name, mode) for mode in results}
    rows = [row for row in rows if (row["model"], row["mode"]) not in replacements]
    rows.extend(
        {key: {"model": model_name, "mode": mode, **metrics}[key] for key in fields}
        for mode, metrics in results.items()
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    improvements = {}
    indexed = {(row["model"], row["mode"]): row for row in rows}
    for complex_name in ("ComplexPhaseNet-Loss-Matched", "ComplexPhaseNet-full"):
        for mode in SNU_MODES:
            real = indexed.get(("PhaseNet-big", mode))
            complex_row = indexed.get((complex_name, mode))
            if real and complex_row:
                lpips_real, pce_real = float(real["lpips"]), float(real["pce"])
                improvements.setdefault(complex_name, {})[mode] = {
                    "lpips_reduction_percent": 100 * (lpips_real - float(complex_row["lpips"])) / lpips_real if lpips_real else None,
                    "pce_reduction_percent": 100 * (pce_real - float(complex_row["pce"])) / pce_real if pce_real else None,
                }
    payload = {"results": rows, "complex_over_phasenet_big": improvements}
    (output_dir / "metrics_snufilm.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    print("\nMode       Samples   PSNR    SSIM    LPIPS    PCE(rad)")
    for mode in snufilm_modes("all"):
        if mode in results:
            m = results[mode]
            print(f"{mode.title():<10} {m['samples']:>7} {m['psnr']:>7.2f} {m['ssim']:>7.4f} {m['lpips']:>8.4f} {m['pce']:>11.4f}")
    print(f"Saved SNU-FILM results to {csv_path} and {output_dir / 'metrics_snufilm.json'}")
