"""DAVIS sequence-split discovery, parsing, and reporting."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DavisSplit:
    """Parsed contents of one DAVIS split file."""

    path: Path
    sequences: list
    frames: list


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
    # Frame lists supplied with 480p data are authoritative for a 480p run.
    return [root / "ImageSets" / version / f"{split}.txt"
            for version in ("480p", "2017", "2016")]


def _non_comment_lines(path):
    return [(number, line.split())
            for number, line in enumerate(path.read_text().splitlines(), 1)
            if line.strip() and not line.lstrip().startswith("#")]


def _parse_frame_path(value, path, line_number):
    parts = value.strip("/").split("/")
    valid = (len(parts) == 4 and parts[0] == "JPEGImages" and
             parts[1] == "480p" and bool(parts[2]) and
             Path(parts[3]).suffix.lower() in {".jpg", ".jpeg"})
    if not valid:
        raise RuntimeError(
            f"Invalid 480p JPEG path in {path} line {line_number}: {value!r}; "
            "expected /JPEGImages/480p/<sequence>/<frame>.jpg"
        )
    return parts[-2]


def parse_davis_split(path):
    """Parse a sequence list or a two-column 480p image/annotation frame list."""
    path = Path(path)
    lines = _non_comment_lines(path)
    if not lines:
        raise RuntimeError(f"Official DAVIS split file is empty: {path}")

    widths = {len(fields) for _, fields in lines}
    if widths == {2}:
        sequences = []
        seen = set()
        frames = []
        for line_number, fields in lines:
            image_path, annotation_path = fields
            sequence = _parse_frame_path(image_path, path, line_number)
            annotation_parts = annotation_path.strip("/").split("/")
            if (len(annotation_parts) != 4 or
                    annotation_parts[:2] != ["Annotations", "480p"] or
                    annotation_parts[2] != sequence or
                    Path(annotation_parts[3]).suffix.lower() != ".png"):
                raise RuntimeError(
                    f"Invalid/mismatched annotation path in {path} line "
                    f"{line_number}: {annotation_path!r}; expected "
                    "/Annotations/480p/<sequence>/<frame>.png"
                )
            frames.append(image_path)
            if sequence not in seen:
                seen.add(sequence)
                sequences.append(sequence)
        return DavisSplit(path.resolve(), sequences, frames)

    if widths == {1}:
        sequences = [fields[0] for _, fields in lines]
        if len(sequences) != len(set(sequences)):
            raise RuntimeError(f"Duplicate sequence in DAVIS split file: {path}")
        return DavisSplit(path.resolve(), sequences, [])

    raise RuntimeError(
        f"Unrecognized DAVIS split format in {path}: each non-comment line "
        "must contain either one sequence name or an image/annotation pair"
    )


def load_davis_split_info(davis_root, split="train", imageset_root=None):
    """Locate and parse an official DAVIS split without fabricating one."""
    if split not in {"train", "val"}:
        raise ValueError(f"DAVIS split must be 'train' or 'val', got {split!r}")
    root = (resolve_imageset_root(imageset_root) if imageset_root is not None
            else Path(davis_root).expanduser().resolve())
    for path in _candidate_split_files(root, split):
        if path.is_file():
            return parse_davis_split(path)
    searched = "\n  ".join(str(p) for p in _candidate_split_files(root, split))
    raise FileNotFoundError(
        f"No official DAVIS {split} split file found. Searched:\n  {searched}\n"
        "Install/extract the complete DAVIS trainval package so that "
        "<DAVIS_ROOT>/ImageSets/480p/train.txt and val.txt exist (preferred), "
        "or provide ImageSets/2017 metadata with --imageset-root."
    )


def load_davis_split(davis_root, split="train", imageset_root=None):
    """Load the unique, order-preserving sequence names for a DAVIS split."""
    return load_davis_split_info(davis_root, split, imageset_root).sequences


def load_davis_train_val_info(davis_root, imageset_root=None):
    train = load_davis_split_info(davis_root, "train", imageset_root)
    val = load_davis_split_info(davis_root, "val", imageset_root)
    overlap = set(train.sequences) & set(val.sequences)
    if overlap:
        raise RuntimeError("DAVIS train/val sequence overlap: " + ", ".join(sorted(overlap)))
    return train, val


def load_davis_train_val(davis_root, imageset_root=None):
    train, val = load_davis_train_val_info(davis_root, imageset_root)
    assert set(train.sequences).isdisjoint(set(val.sequences))
    return train.sequences, val.sequences


def print_davis_split(split, sequences, triplet_count, evaluation=False):
    label = "DAVIS evaluation split" if evaluation else "DAVIS split"
    print(f"{label}: {split}")
    print("DAVIS source resolution: 480p")
    print("Model input resolution: 256 x 256")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Number of {'triplets' if evaluation else 'generated triplets'}: {triplet_count}")
    print("Sequence names:")
    print("\n".join(sequences))
