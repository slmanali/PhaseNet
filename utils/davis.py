"""Official DAVIS sequence-split discovery and reporting."""

from pathlib import Path


def resolve_davis_root(davis_root=None, dataset_path=None):
    """Resolve the DAVIS root, accepting either root or JPEGImages/480p."""
    if davis_root is not None:
        root = Path(davis_root).expanduser().resolve()
    elif dataset_path is not None:
        images = Path(dataset_path).expanduser().resolve()
        if images.name == "480p" and images.parent.name == "JPEGImages":
            root = images.parent.parent
        elif images.name == "JPEGImages":
            root = images.parent
        else:
            root = images
    else:
        raise ValueError("Pass --davis-root or --dataset-path")
    return root


def resolve_imageset_root(imageset_root):
    """Resolve a metadata location from a DAVIS root or image directory."""
    path = Path(imageset_root).expanduser().resolve()
    if path.name == "480p" and path.parent.name == "JPEGImages":
        return path.parent.parent
    if path.name == "JPEGImages":
        return path.parent
    return path


def davis_image_root(davis_root, dataset_path=None):
    return (Path(dataset_path).expanduser().resolve() if dataset_path else
            Path(davis_root) / "JPEGImages" / "480p")


def _candidate_split_files(root, split):
    # DAVIS 2017 lists live in ImageSets/2017; older installations may use 2016.
    return [root / "ImageSets" / year / f"{split}.txt"
            for year in ("2017", "2016")]


def load_davis_split(davis_root, split="train", imageset_root=None):
    """Load an official DAVIS train/val sequence list; never fabricate one."""
    if split not in {"train", "val"}:
        raise ValueError(f"DAVIS split must be 'train' or 'val', got {split!r}")
    root = (resolve_imageset_root(imageset_root) if imageset_root is not None
            else Path(davis_root).expanduser().resolve())
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
    raise FileNotFoundError(
        f"No official DAVIS {split} split file found. Searched:\n  {searched}\n"
        "The JPEGImages-only download does not contain the official split metadata. "
        "Install/extract the complete DAVIS trainval package so that "
        "<DAVIS_ROOT>/ImageSets/2017/train.txt and val.txt exist, or pass the "
        "directory containing ImageSets with --imageset-root."
    )


def load_davis_train_val(davis_root, imageset_root=None):
    train = load_davis_split(davis_root, "train", imageset_root)
    val = load_davis_split(davis_root, "val", imageset_root)
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
