#!/usr/bin/env python3
"""Count parameters of the EXACT local RIFE or FILM checkpoint used for inference.

Run from PhaseNet repository root, after placing this script in tools/.
Use venv_phasenet for RIFE, venv_film for FILM. No inference is performed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Make counting independent of GPU allocation and the local TensorFlow allocator.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            data = handle.read(4 * 1024 * 1024)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def git_revision(path):
    run = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    return run.stdout.strip() if run.returncode == 0 else None


def count_rife(args):
    # Import the same checkpoint loader used to generate the saved predictions.
    import torch
    from tools.run_pretrained_snufilm import RIFEBackend

    backend = RIFEBackend(args.repo, args.checkpoint, "cpu")
    network = getattr(backend.model, "flownet", None)
    if not isinstance(network, torch.nn.Module):
        raise RuntimeError("Cannot find RIFE loaded flownet; do NOT count a newly instantiated random model.")
    named = list(network.named_parameters())
    if not named:
        raise RuntimeError("RIFE checkpoint loaded, but no model parameters were found")
    weights = Path(args.checkpoint) / "flownet.pkl"
    if not weights.is_file():
        raise FileNotFoundError(f"Cannot establish RIFE checkpoint identity: {weights}")
    return {
        "model": "RIFE",
        "count_definition": "unique registered torch.nn.Parameter elements in loaded flownet",
        "total_parameters": int(sum(p.numel() for _, p in named)),
        "trainable_parameters": int(sum(p.numel() for _, p in named if p.requires_grad)),
        "parameter_tensors": len(named),
        "checkpoint_file": str(weights.resolve()),
        "checkpoint_sha256": sha256(weights),
        "upstream_repo": str(Path(args.repo).resolve()),
        "upstream_commit": git_revision(args.repo),
        "backend": "tools.run_pretrained_snufilm.RIFEBackend",
        "torch_version": torch.__version__,
    }


def count_film(args):
    # The upstream FILM Interpolator stores a tf.saved_model.load object in _model.
    # Count model variables, not file size, SavedModel graph constants, or Adam slots.
    import tensorflow as tf
    from tools.run_pretrained_snufilm import FILMBackend

    backend = FILMBackend(args.repo, args.checkpoint, "cpu")
    model = backend.model._model
    if not hasattr(model, "variables"):
        raise RuntimeError(
            "FILM SavedModel does not expose .variables. Please send this error and "
            "the checkpoint variable listing; do not guess counts from file size."
        )
    variables = list(model.variables)
    if not variables:
        raise RuntimeError("FILM model.variables is empty: cannot establish parameter count")
    unique = {}
    for variable in variables:
        # A TF Variable might be reachable through more than one attribute.
        unique[id(variable)] = variable
    vars_unique = list(unique.values())
    # The loaded SavedModel itself supplies trainable flags; do not infer from names.
    trainable = [v for v in vars_unique if bool(v.trainable)]
    total = sum(int(tf.size(v).numpy()) for v in vars_unique)
    trainable_count = sum(int(tf.size(v).numpy()) for v in trainable)
    ckpt_prefix = Path(args.checkpoint) / "variables" / "variables"
    checkpoint_files = [Path(args.checkpoint) / "saved_model.pb",
                        Path(str(ckpt_prefix) + ".index"),
                        *sorted((Path(args.checkpoint) / "variables").glob("variables.data-*"))]
    missing = [str(p) for p in checkpoint_files[:2] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing FILM SavedModel checkpoint file(s): " + ", ".join(missing))
    return {
        "model": "FILM",
        "count_definition": "unique tf.Variable elements in exact loaded SavedModel",
        "total_parameters": int(total),
        "trainable_parameters": int(trainable_count),
        "nontrainable_state_elements": int(total - trainable_count),
        "variable_tensors": len(vars_unique),
        "checkpoint_dir": str(Path(args.checkpoint).resolve()),
        "checkpoint_file_sha256": {str(path.resolve()): sha256(path)
                                   for path in checkpoint_files if path.is_file()},
        "upstream_repo": str(Path(args.repo).resolve()),
        "upstream_commit": git_revision(args.repo),
        "backend": "tools.run_pretrained_snufilm.FILMBackend._model",
        "tensorflow_version": tf.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rife", "film"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.repo.is_dir() or not args.checkpoint.is_dir():
        parser.error("--repo and --checkpoint must point to existing directories")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    payload = count_rife(args) if args.model == "rife" else count_film(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
