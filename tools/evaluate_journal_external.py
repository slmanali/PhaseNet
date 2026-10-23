"""Evaluate RIFE/FILM with the same floating-point SNU-FILM inputs as PhaseNet."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from tools.run_pretrained_snufilm import RIFEBackend, FILMBackend
from utils.evaluation_protocol import make_dataset, sha256, write_csv, write_json
from utils.metrics import compute_psnr, compute_ssim, compute_lpips
from utils.reproducibility import environment_metadata


@torch.inference_mode()
def predict_float(backend, kind, first, second):
    """Keep resized RGB endpoints and predictions as float; avoid PNG rounding."""
    height, width = first.shape[-2:]
    if first.shape != second.shape or first.shape[:2] != (1, 3):
        raise ValueError("expected two equally sized [1,3,H,W] endpoints")
    if kind == "rife":
        divisor = max(32, int(32 / backend.scale))
        padding = (0, (-width) % divisor, 0, (-height) % divisor)
        a, b = F.pad(first.to(backend.device), padding), F.pad(second.to(backend.device), padding)
        result = backend.model.inference(a, b, scale=backend.scale)[..., :height, :width]
        return result.float().clamp(0, 1)
    a = first.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    b = second.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    result = np.asarray(backend.model(a, b, np.array([.5], np.float32)))
    if result.shape != a.shape:
        raise ValueError(f"FILM output shape differs: {result.shape} versus {a.shape}")
    return torch.from_numpy(result.copy()).permute(0, 3, 1, 2).float().clamp(0, 1)


def model_provenance(repo, checkpoint):
    repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file()) if checkpoint.is_dir() else [checkpoint]
    if not files:
        raise FileNotFoundError(f"empty checkpoint directory: {checkpoint}")
    return {"repository": str(repo), "repository_commit": commit, "repository_dirty": dirty,
            "checkpoint": str(checkpoint), "checkpoint_files_sha256": {
                str(path.relative_to(checkpoint)) if checkpoint.is_dir() else path.name: sha256(path) for path in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rife", "film"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=Path("SNU-FILM"))
    parser.add_argument("--image-size", choices=("256", "native"), default="256")
    parser.add_argument("--modes", nargs="+", choices=("easy", "medium", "hard", "extreme"),
                        default=["easy", "medium", "hard", "extreme"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("results/computer_optics/external_references"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    device = args.device or ("cuda:0" if args.model == "rife" and torch.cuda.is_available() else "cpu")
    provenance = model_provenance(args.repo, args.checkpoint)
    backend = (RIFEBackend(args.repo, args.checkpoint, device, 1.0) if args.model == "rife"
               else FILMBackend(args.repo, args.checkpoint, device))
    import lpips
    metric_device = torch.device(device)
    loss_fn = lpips.LPIPS(net="alex").to(metric_device).eval()
    for mode in args.modes:
        destination = args.output / args.model / args.image_size / mode
        if (destination / "summary.json").exists() and not args.overwrite:
            raise FileExistsError(f"{destination}: choose a new output path or --overwrite")
        dataset, manifest, protocol = make_dataset({"kind": "snufilm", "path": str(args.dataset_path),
                                                   "image_size": args.image_size}, mode)
        rows = []
        for index, sample in enumerate(dataset):
            with torch.no_grad():
                prediction = predict_float(backend, args.model, sample["start"].unsqueeze(0), sample["end"].unsqueeze(0))
                prediction, truth = prediction.to(metric_device), sample["inter"].unsqueeze(0).to(metric_device)
                rows.append({"model": args.model, "subset": mode, **manifest[index],
                    "psnr": compute_psnr(prediction, truth), "ssim": compute_ssim(prediction, truth),
                    "lpips": compute_lpips(prediction, truth, loss_fn)})
            if (index + 1) % 25 == 0:
                print(f"{args.model}/{mode}: {index + 1}/{len(dataset)}", flush=True)
        write_csv(destination / "samples.csv", rows)
        write_json(destination / "summary.json", {
            "model": args.model, "subset": mode, "samples": len(rows),
            **{metric: sum(row[metric] for row in rows) / len(rows) for metric in ("psnr", "ssim", "lpips")},
            "protocol": protocol, "model_provenance": provenance, "environment": environment_metadata(),
            "input_and_output": "float32 RGB; same SNU-FILM tensor resize; output clamp [0,1]; no PNG quantization",
            "padding": "RIFE zero-pad to multiple of 32, crop to input size" if args.model == "rife"
            else "upstream FILM Interpolator align=64, block_shape=[2,2]",
            "training_scope": "pretrained external reference; not a training-data-matched control"})


if __name__ == "__main__":
    main()
