"""Official DAVIS sequence-split discovery and reporting."""

from pathlib import Path


def resolve_davis_root(davis_root=None, dataset_path=None):
    """Resolve the DAVIS root, accepting either root or JPEGImages/480p."""
    if davis_root is not None:
        root = Path(davis_root).expanduser().resolve()
    elif dataset_path is not None:
        images = Path(dataset_path).expanduser().resolve()
        root = images.parent.parent if images.name == "480p" else images
    else:
        raise ValueError("Pass --davis-root or --dataset-path")
    return root


def davis_image_root(davis_root, dataset_path=None):
    return (Path(dataset_path).expanduser().resolve() if dataset_path else
            Path(davis_root) / "JPEGImages" / "480p")


def _candidate_split_files(root, split):
    # DAVIS 2017 lists live in ImageSets/2017; older installations may use 2016.
    return [root / "ImageSets" / year / f"{split}.txt"
            for year in ("2017", "2016")]


def load_davis_split(davis_root, split="train"):
    """Load an official DAVIS train/val sequence list; never fabricate one."""
    if split not in {"train", "val"}:
        raise ValueError(f"DAVIS split must be 'train' or 'val', got {split!r}")
    root = Path(davis_root).expanduser().resolve()
    for path in _candidate_split_files(root, split):
        if path.is_file():
            names = [line.strip().split()[0] for line in path.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
            if not names:
                raise RuntimeError(f"Official DAVIS split file is empty: {path}")
            if len(names) != len(set(names)):
                raise RuntimeError(f"Duplicate sequence in DAVIS split file: {path}")
            return names
    searched = "\n  ".join(str(p) for p in _candidate_split_files(root, split))
    raise FileNotFoundError(f"No official DAVIS {split} split file found. Searched:\n  {searched}")


def load_davis_train_val(davis_root):
    train = load_davis_split(davis_root, "train")
    val = load_davis_split(davis_root, "val")
    overlap = set(train) & set(val)
    if overlap:
        raise RuntimeError("DAVIS train/val sequence overlap: " + ", ".join(sorted(overlap)))
    assert set(train).isdisjoint(set(val))
    return train, val


def print_davis_split(split, sequences, triplet_count, evaluation=False):
    label = "DAVIS evaluation split" if evaluation else "DAVIS split"
    print(f"{label}: {split}")
    print("DAVIS source resolution: 480p")
    print("Model input resolution: 256 x 256")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Number of {'triplets' if evaluation else 'generated triplets'}: {triplet_count}")
    print("Sequence names:")
    print("\n".join(sequences))
