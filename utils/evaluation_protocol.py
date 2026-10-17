"""Evaluation manifests and result I/O; no model or metric implementation."""
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=json_default) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.with_suffix(".tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, default=json_default) if isinstance(value, (list, dict))
                          else value for key, value in row.items()} for row in rows)
    path.with_suffix(".tmp").replace(path)


def source_hashes(paths, root=Path(".")):
    return {name: sha256(root / name) for name in paths}


def verify_source_lock(plan, root=Path(".")):
    expected = plan.get("source_lock", {})
    actual = source_hashes(expected, root)
    changed = [name for name in expected if actual[name] != expected[name]]
    if changed:
        raise ValueError("baseline evaluator source changed: " + ", ".join(changed)
                         + "; use feature/complex-valued-network for the unchanged baseline")
    return actual


def make_dataset(spec, subset):
    # These are the same readers and transforms used by the original CLIs.
    from PIL import Image
    from torchvision import transforms
    from net.phasenet import Triplets
    from test_complex_safe_baseline import UCF101Triplets, MiddleburyTriplets
    from utils.davis import load_davis_train_val, resolve_davis_root
    from utils.snufilm import SNUFILMTriplets

    root = Path(spec["path"]).expanduser().resolve()
    kind = spec["kind"]
    size = spec.get("image_size", "256")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    if kind == "davis":
        if size != "256" or subset != "val":
            raise ValueError("this journal plan requires DAVIS val at 256x256")
        davis_root = resolve_davis_root(dataset_path=root)
        train, val = load_davis_train_val(davis_root)
        images = root if root.name == "480p" else davis_root / "JPEGImages/480p"
        dataset = Triplets(str(images), transform, allowed_sequences=val)
        paths = [[Path(item[0]) for item in triplet] for triplet in dataset.sample]
        split_info = {"train_sequences": train, "evaluation_sequences": val,
                      "sequence_overlap": sorted(set(train) & set(val))}
    elif kind == "snufilm":
        dataset = SNUFILMTriplets(root, subset, image_size=size)
        paths = dataset.triplets
        split_info = {"list_file": str(dataset.list_file), "list_sha256": sha256(dataset.list_file)}
    else:
        if size != "256":
            raise ValueError("legacy UCF101 and Middlebury readers evaluate at 256x256")
        cls = UCF101Triplets if kind == "ucf101" else MiddleburyTriplets
        dataset = cls(root, transform)
        names = (("frame_00.png", "frame_01_gt.png", "frame_02.png") if kind == "ucf101"
                 else ("frame10.png", "frame11.png", "frame12.png"))
        paths = [[folder / filename for filename in names] for folder in dataset.samples]
        split_info = {"selection": "all valid folders found by the existing dataset reader"}
    if not paths:
        raise ValueError("empty evaluation dataset")
    manifest = []
    for index, frames in enumerate(paths):
        names = [str(path.resolve().relative_to(root)) if path.resolve().is_relative_to(root)
                 else str(path.resolve()) for path in frames]
        sizes = []
        for frame in frames:
            with Image.open(frame) as image:
                sizes.append([image.height, image.width])
        sample_key = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
        manifest.append({"sample_index": index, "sample_key": sample_key, "frames": names,
            "sequence_id": str(Path(names[0]).parent),
            "sequence_grouping_verified": kind == "davis",
            "source_sizes_hw": sizes, "metric_size_hw": [256, 256] if size == "256" else sizes[1]})
    if spec.get("sequence_map"):
        with Path(spec["sequence_map"]).open(newline="") as stream:
            mapping_rows = list(csv.DictReader(stream))
        mapping = {row["sample_key"]: row["sequence_id"] for row in mapping_rows}
        if len(mapping) != len(mapping_rows) or set(mapping) != {row["sample_key"] for row in manifest}:
            raise ValueError("sequence map must contain each evaluated sample_key exactly once")
        for row in manifest:
            if not mapping[row["sample_key"]]:
                raise ValueError("empty verified source-video identifier")
            row.update(sequence_id=mapping[row["sample_key"]], sequence_grouping_verified=True)
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return dataset, manifest, {**split_info, "samples": len(dataset),
        "sample_manifest_sha256": manifest_hash, "hash_scope": "ordered paths, grouping and dimensions; not image bytes",
        "image_size": size, "rgb": True, "output_clamp": [0, 1],
        "resize": "256x256 bilinear antialiased; PIL for DAVIS, tensor for external datasets" if size == "256" else "none",
        "evaluation_padding_crop": "none; pyramid uses its existing internal conventions"}
