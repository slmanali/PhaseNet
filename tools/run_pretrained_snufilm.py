#!/usr/bin/env python3
"""Run official pretrained RIFE (4.6+) or FILM models on SNU-FILM."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.snufilm import SNUFILMTriplets, SNU_MODES, snufilm_modes


def _import_from_repo(repo, module_name):
    """Import an upstream module while leaving the user's checkout untouched."""
    repo = str(Path(repo).expanduser().resolve())
    sys.path.insert(0, repo)
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(repo)


class RIFEBackend:
    """Thin adapter around the API shared by official RIFE 4.6+ releases."""

    def __init__(self, repo, checkpoint, device, scale=1.0):
        # This is the entry point used by upstream's own inference scripts for
        # released RIFE checkpoints.  model.RIFE is the training wrapper for a
        # different (older) IFNet layout; it imports successfully, but loading
        # current flownet.pkl files into it produces a very long and misleading
        # missing/unexpected-keys error.
        rife = _import_from_repo(repo, "model.RIFE_HDv3")
        self.model = rife.Model()
        checkpoint = str(Path(checkpoint).expanduser().resolve())
        try:
            self.model.load_model(checkpoint, -1)
        except RuntimeError as error:
            raise RuntimeError(
                "The RIFE checkpoint is incompatible with model.RIFE_HDv3 in "
                f"{Path(repo).expanduser().resolve()}. Download the checkpoint "
                "release recommended by that RIFE checkout (or check out the "
                "RIFE revision matching the downloaded weights)."
            ) from error
        self.model.eval()
        self.model.device()
        self.device = torch.device(device)
        # Upstream chooses CUDA solely from availability. Honour an explicit
        # --device cpu (and indexed CUDA devices) after its setup hook.
        if hasattr(self.model, "flownet"):
            self.model.flownet.to(self.device)
        self.scale = scale

    def __call__(self, first, second):
        tensors = []
        for image in (first, second):
            tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
            tensors.append(tensor.unsqueeze(0).float().to(self.device) / 255.0)
        height, width = tensors[0].shape[-2:]
        divisor = max(32, int(32 / self.scale))
        padded_h = ((height - 1) // divisor + 1) * divisor
        padded_w = ((width - 1) // divisor + 1) * divisor
        padding = (0, padded_w - width, 0, padded_h - height)
        tensors = [F.pad(tensor, padding) for tensor in tensors]
        with torch.inference_mode():
            prediction = self.model.inference(tensors[0], tensors[1], scale=self.scale)
        prediction = prediction[0, :, :height, :width].clamp(0, 1)
        return (prediction.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


class FILMBackend:
    """Adapter around google-research/frame-interpolation's FILM interpolator."""

    def __init__(self, repo, checkpoint):
        module = _import_from_repo(repo, "eval.interpolator")
        self.model = module.Interpolator(str(Path(checkpoint).expanduser().resolve()))

    def __call__(self, first, second):
        first = np.asarray(first, dtype=np.float32) / 255.0
        second = np.asarray(second, dtype=np.float32) / 255.0
        prediction = np.asarray(self.model(first, second, np.array([0.5], np.float32)))
        if prediction.ndim == 4:
            prediction = prediction[0]
        return (np.clip(prediction, 0, 1) * 255).round().astype(np.uint8)


def parse_indices(value):
    if value is None:
        return None
    try:
        indices = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from error
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        raise argparse.ArgumentTypeError("indices must be unique non-negative integers")
    return indices


def validate_paths(parser, args):
    """Resolve user paths and report every missing input with an actionable hint."""
    for name in ("repo", "checkpoint", "snu_root", "output_dir"):
        value = getattr(args, name)
        setattr(args, name, value.expanduser().resolve())

    missing = []
    if not args.repo.is_dir():
        missing.append(f"--repo directory not found: {args.repo}")
    if not args.checkpoint.is_dir():
        missing.append(f"--checkpoint directory not found: {args.checkpoint}")
    if not args.snu_root.is_dir():
        missing.append(f"--snu-root directory not found: {args.snu_root}")
    if missing:
        parser.error("\n".join(missing) +
                     "\nRepository clones do not include pretrained weights, and creating an "
                     "empty checkpoint directory is not sufficient. Download and extract the "
                     "model checkpoint separately, then pass the extracted train_log directory "
                     "for RIFE or SavedModel directory for FILM.")


def run(args, backend):
    completed = []
    for mode in snufilm_modes(args.snu_mode):
        dataset = SNUFILMTriplets(args.snu_root, mode, sample_indices=args.sample_indices)
        for position, paths in zip(dataset.sample_indices, dataset.triplets):
            destination = args.output_dir / args.model / mode / f"{position:05d}_pred.png"
            if destination.is_file() and not args.overwrite:
                continue
            with Image.open(paths[0]) as image:
                first = image.convert("RGB").copy()
            with Image.open(paths[2]) as image:
                second = image.convert("RGB").copy()
            prediction = backend(first, second)
            if prediction.shape[:2] != (first.height, first.width):
                raise ValueError(f"model returned {prediction.shape[:2]}, expected "
                                 f"{(first.height, first.width)} for {mode}:{position}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prediction).save(destination)
            completed.append({"mode": mode, "sample_index": position,
                              "prediction": str(destination)})
    manifest = args.output_dir / args.model / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"model": args.model, "predictions": completed}, indent=2)
                        + "\n", encoding="utf-8")
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("rife", "film"))
    parser.add_argument("--snu-root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path,
                        help="Checkout of the official model repository.")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="RIFE train_log directory or extracted FILM SavedModel directory.")
    parser.add_argument("--snu-mode", choices=(*SNU_MODES, "all"), default="all")
    parser.add_argument("--sample-indices", type=parse_indices)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/external_snufilm"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="RIFE inference scale (use 0.5 for very large motion).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("--scale must be positive")
    validate_paths(parser, args)
    backend = (RIFEBackend(args.repo, args.checkpoint, args.device, args.scale)
               if args.model == "rife" else FILMBackend(args.repo, args.checkpoint))
    completed = run(args, backend)
    print(f"Wrote {len(completed)} {args.model.upper()} prediction(s) to "
          f"{args.output_dir / args.model}")


if __name__ == "__main__":
    main()
