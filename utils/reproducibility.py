"""Shared, version-compatible reproducibility and checkpoint utilities."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def configure_reproducibility(seed: int, deterministic: str = "strict") -> None:
    """Seed model RNGs and configure strict or best-effort deterministic execution."""
    if deterministic not in {"strict", "warn", "off"}:
        raise ValueError("deterministic must be strict, warn, or off")
    # The launcher sets these before Python starts. This fallback is useful for
    # direct invocation, and CUBLAS_WORKSPACE_CONFIG is still early enough when
    # this function is called before device/model creation.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic != "off":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = deterministic == "off"
    torch.backends.cudnn.deterministic = deterministic != "off"
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(
            deterministic != "off", warn_only=deterministic == "warn"
        )


def make_data_generator(seed: int) -> torch.Generator:
    """Make the independent, persistent DataLoader RNG for a run."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Top-level DataLoader callback; seed augmentation RNGs from worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def ordered_hash(values) -> str:
    payload = json.dumps([str(value) for value in values], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_sample_ids(dataset):
    """Return stable sample identifiers without loading images."""
    base = getattr(dataset, "dataset", dataset)
    indices = getattr(dataset, "indices", range(len(base)))
    samples = getattr(base, "triplets", getattr(base, "samples",
                       getattr(base, "sample", None)))
    return [str(samples[i] if samples is not None else i) for i in indices]


def capture_rng_state(data_generator=None):
    state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if data_generator is not None:
        state["data_loader_generator"] = data_generator.get_state()
    return state


def restore_rng_state(state, data_generator=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if data_generator is not None and state.get("data_loader_generator") is not None:
        data_generator.set_state(state["data_loader_generator"])


def save_training_checkpoint(path, model, optimizer, scheduler, epoch, step,
                             configuration, data_generator, scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1, "resume_boundary": "completed_epoch",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch, "global_step": step, "configuration": configuration,
        "rng_state": capture_rng_state(data_generator),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    data_generator=None, scaler=None, map_location="cpu"):
    """Load new resumable checkpoints or historical bare state dictionaries."""
    # Full checkpoints deliberately contain Python/NumPy RNG tuples. PyTorch
    # 2.6+ therefore needs the explicit trusted-checkpoint opt-out below.
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        checkpoint = torch.load(path, map_location=map_location)
    is_full = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
    model.load_state_dict(checkpoint["model_state_dict"] if is_full else checkpoint)
    if not is_full:
        return {"legacy": True, "epoch": 0, "global_step": 0}
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if checkpoint.get("rng_state"):
        restore_rng_state(checkpoint["rng_state"], data_generator)
    checkpoint["legacy"] = False
    return checkpoint


def environment_metadata():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    cuda = torch.cuda.is_available()
    driver = None
    if cuda:
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True).splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            pass
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_count": torch.cuda.device_count() if cuda else 0,
        "nvidia_driver": driver,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
    }


def create_run_directory(root, configuration, seed, resume=False):
    path = Path(root) / configuration / f"seed-{seed}"
    if path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing run: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_metadata(path, metadata):
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(destination)
