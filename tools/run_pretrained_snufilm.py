#!/usr/bin/env python3
"""Run official pretrained RIFE (4.6+) or FILM models on SNU-FILM."""

import argparse
import importlib
import json
import os
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


def _import_from_repo(repo, module_name, dependency_paths=()):
    """Import an upstream module with its repository dependencies available."""
    paths = [repo, *dependency_paths]
    paths = [str(Path(path).expanduser().resolve()) for path in paths]
    sys.path[:0] = paths
    try:
        return importlib.import_module(module_name)
    finally:
        for path in paths:
            sys.path.remove(path)


class RIFEBackend:
    """Thin adapter around the API shared by official RIFE 4.6+ releases."""

    def __init__(self, repo, checkpoint, device, scale=1.0):
        # This is the entry point used by upstream's own inference scripts for
        # released RIFE checkpoints.  model.RIFE is the training wrapper for a
        # different (older) IFNet layout; it imports successfully, but loading
        # current flownet.pkl files into it produces a very long and misleading
        # missing/unexpected-keys error.
        # Depending on the release, RIFE_HDv3 is shipped either in the Git
        # checkout's model package or alongside flownet.pkl in train_log.
        # Prefer the checkout, but support the latter official archive layout.
        try:
            rife = _import_from_repo(repo, "model.RIFE_HDv3")
        except ModuleNotFoundError as error:
            if error.name not in ("model", "model.RIFE_HDv3"):
                raise
            # Archive copies use package-qualified imports such as
            # ``from train_log.IFNet_HDv3 import *`` while still importing
            # helpers such as model.warplayer from the checkout.  Import the
            # checkpoint directory as a package from its parent so both forms
            # resolve while Python executes the module.
            checkpoint = Path(checkpoint).expanduser().resolve()
            rife = _import_from_repo(
                checkpoint.parent, f"{checkpoint.name}.RIFE_HDv3", (repo,)
            )
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

    def __init__(self, repo, checkpoint, device="cpu"):
        # PyTorch calls this backend "cuda", while TensorFlow commonly calls
        # the same device a GPU. Accept the latter spelling as a convenience,
        # especially because the exception emitted by TensorFlow says "GPU".
        if device == "gpu":
            device = "cuda"
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("FILM --device must be cpu, cuda, or cuda:<index>")

        # TensorFlow probes CUDA when it is first imported. Select (or hide) the
        # GPU before importing the upstream interpolator so an incompatible host
        # cuDNN installation cannot make the default CPU execution fail. An
        # indexed CUDA device becomes TensorFlow's sole visible GPU.
        if device.type == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif device.index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
        try:
            module = _import_from_repo(repo, "eval.interpolator")
        except ModuleNotFoundError as error:
            if error.name != "tensorflow":
                raise
            raise RuntimeError(
                "FILM requires TensorFlow, but it is not installed in this Python "
                "environment. Install the optional inference dependencies with "
                "`python -m pip install -r requirements-film.txt`, then rerun the "
                "command."
            ) from error
        self.model = module.Interpolator(
            str(Path(checkpoint).expanduser().resolve()),
            align=64,
            block_shape=[2, 2],
        )
        self.device = device

    def __call__(self, first, second):
        # FILM's exported SavedModel signature is batched: both endpoints must
        # have shape [batch, height, width, channels].  The upstream adapter
        # forwards its inputs unchanged, so add the singleton inference batch
        # here rather than passing bare HWC images to TensorFlow.
        first = np.asarray(first, dtype=np.float32)[None, ...] / 255.0
        second = np.asarray(second, dtype=np.float32)[None, ...] / 255.0
        try:
            prediction = np.asarray(self.model(first, second, np.array([0.5], np.float32)))
        except Exception as error:
            message = str(error).lower()
            if self.device.type == "cuda" and (
                "no dnn" in message or "cudnn" in message
            ):
                raise RuntimeError(
                    "FILM GPU inference could not initialize cuDNN. The installed "
                    "TensorFlow build and runtime cuDNN must be compatible. Rerun "
                    "with `--device cpu` (the safe default), or install the cuDNN "
                    "version required by your TensorFlow build. Any predictions "
                    "saved before this error are still on disk."
                ) from error
            raise
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
                print(f"Skipping existing {mode}:{position} ({destination})", flush=True)
                continue
            with Image.open(paths[0]) as image:
                first = image.convert("RGB").copy()
            with Image.open(paths[2]) as image:
                second = image.convert("RGB").copy()
            print(f"Processing {args.model.upper()} {mode}:{position} -> {destination}",
                  flush=True)
            prediction = backend(first, second)
            if prediction.shape[:2] != (first.height, first.width):
                raise ValueError(f"model returned {prediction.shape[:2]}, expected "
                                 f"{(first.height, first.width)} for {mode}:{position}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prediction).save(destination)
            completed.append({"mode": mode, "sample_index": position,
                              "prediction": str(destination)})
            print(f"Saved {destination}", flush=True)
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
    parser.add_argument(
        "--device",
        help=("Inference device. Defaults to CUDA when available for RIFE and CPU for "
              "FILM; use cuda (or its gpu alias) or cuda:<index> to opt into "
              "TensorFlow GPU inference."),
    )
    parser.add_argument("--scale", type=float, default=1.0,
                        help="RIFE inference scale (use 0.5 for very large motion).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("--scale must be positive")
    validate_paths(parser, args)
    device = args.device or (
        "cuda" if args.model == "rife" and torch.cuda.is_available() else "cpu"
    )
    backend = (RIFEBackend(args.repo, args.checkpoint, device, args.scale)
               if args.model == "rife" else FILMBackend(args.repo, args.checkpoint, device))
    completed = run(args, backend)
    print(f"Wrote {len(completed)} {args.model.upper()} prediction(s) to "
          f"{args.output_dir / args.model}")


if __name__ == "__main__":
    main()
