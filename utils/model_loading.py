"""Configuration-aware inference loading for trusted research checkpoints."""
from pathlib import Path
import hashlib
import json

import torch

MODEL_DEFAULTS = {
    "feature_dim": 64, "convolution": "complex", "phase_correction_scale": 0.1,
    "residual_correction_scale": 0.1,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint(path):
    # Full training files contain Python/NumPy RNG state. Only load files trusted
    # by the researcher; inference does not restore that state.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_inference_model(path, device="cpu", architecture=None, **overrides):
    """Use saved settings and reject conflicting explicit architectural flags.

    A bare state dictionary should remain beside its metadata.json. Historical
    files without metadata use the declared flags and documented legacy defaults.
    Model construction is isolated from the caller's CPU random generator.
    """
    from net.phasenet import PhaseNet
    from net.complex_phasenet_safe_baseline import ComplexPhaseNetSafe

    path = Path(path)
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    checkpoint = read_checkpoint(path)
    full = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
    state = checkpoint["model_state_dict"] if full else checkpoint
    saved = dict(metadata.get("arguments", {}))
    embedded = checkpoint.get("configuration", {}) if full else {}
    for key in MODEL_DEFAULTS:
        if key in saved and key in embedded and saved[key] != embedded[key]:
            raise ValueError(f"checkpoint/metadata disagree on {key}")
    saved.update(embedded)
    unknown = set(overrides) - set(MODEL_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown model overrides: {sorted(unknown)}")
    spec = {}
    for key, default in MODEL_DEFAULTS.items():
        explicit = overrides.get(key)
        if explicit is not None and key in saved and explicit != saved[key]:
            raise ValueError(f"{key}={explicit!r} conflicts with saved value {saved[key]!r}")
        spec[key] = saved.get(key, default) if explicit is None else explicit

    inferred = "complex" if any("running_var_rr" in key for key in state) else "real"
    if architecture is not None and architecture != inferred:
        raise ValueError(f"requested {architecture} model but checkpoint is {inferred}")
    architecture = inferred
    with torch.random.fork_rng(devices=[]):
        model = (PhaseNet(spec["feature_dim"]) if architecture == "real"
                 else ComplexPhaseNetSafe(**spec))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    info = {
        "checkpoint": str(path), "checkpoint_sha256": file_sha256(path),
        "architecture": architecture, "model_spec": spec,
        "configuration": saved.get("config_name", metadata.get("configuration")),
        "seed": saved.get("seed", metadata.get("seed")),
        "study_id": saved.get("study_id", metadata.get("study_id")),
        "epoch": checkpoint.get("epoch") if full else metadata.get("completed_epochs"),
        "training_arguments": saved,
        "training_environment": metadata.get("environment"),
    }
    return model, info
