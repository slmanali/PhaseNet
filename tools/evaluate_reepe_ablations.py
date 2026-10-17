"""Evaluate only the fresh REEPE runs and export traceable per-triplet metrics."""
import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import statistics

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from net.phasenet import Triplets
from test_complex_safe_baseline import evaluate, UCF101Triplets, MiddleburyTriplets
from tools.reepe_common import completed_checkpoint, select_runs, write_csv, write_json
from utils.davis import load_davis_train_val, resolve_davis_root
from utils.model_loading import load_inference_model
from utils.reproducibility import configure_reproducibility, environment_metadata, ordered_hash
from utils.snufilm import SNUFILMTriplets, snufilm_modes

PROTOCOL = "reepe_rgb_raw_amplitude_pce_v2"
METRICS = ("psnr", "ssim", "lpips", "pce")


def make_dataset(kind, root, subset, image_size):
    root = Path(root).expanduser().resolve()
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    if kind == "davis":
        if image_size != "256":
            raise ValueError("DAVIS uses the declared 256x256 evaluation protocol")
        davis_root = resolve_davis_root(dataset_path=root)
        _, sequences = load_davis_train_val(davis_root)
        images = (root if root.name == "480p" else davis_root / "JPEGImages" / "480p")
        dataset = Triplets(str(images), transform, allowed_sequences=sequences)
        paths = [[Path(item[0]) for item in triplet] for triplet in dataset.sample]
    elif kind == "snufilm":
        dataset = SNUFILMTriplets(root, subset, image_size=image_size)
        paths = dataset.triplets
    else:
        transform = transform if image_size == "256" else None
        cls = UCF101Triplets if kind == "ucf101" else MiddleburyTriplets
        dataset = cls(root, transform)
        names = (("frame_00.png", "frame_01_gt.png", "frame_02.png") if kind == "ucf101"
                 else ("frame10.png", "frame11.png", "frame12.png"))
        paths = [[folder / filename for filename in names] for folder in dataset.samples]
    identities = []
    for index, triplet in enumerate(paths):
        frames = [str(path.resolve().relative_to(root)) if path.resolve().is_relative_to(root)
                  else str(path.resolve()) for path in triplet]
        identities.append({"sample_index": index, "frames": frames,
                           "sequence_id": str(Path(frames[0]).parent),
                           "sequence_grouping": "DAVIS sequence" if kind == "davis"
                           else "frame-directory proxy; verify original-video grouping"})
    if not identities:
        raise ValueError("empty evaluation dataset")
    return dataset, identities


def summarize(rows, expected_seeds):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["variant"], row["seed"], row["dataset"], row["subset"])
        if key in seen:
            raise ValueError(f"duplicate evaluation: {key}")
        seen.add(key)
        groups[(row["variant"], row["dataset"], row["subset"])].append(row)
    result = []
    for (variant, dataset, subset), values in groups.items():
        if len({row["sample_manifest_sha256"] for row in values}) != 1:
            raise ValueError("cannot pool seeds evaluated on different triplets")
        missing = sorted(set(expected_seeds) - {row["seed"] for row in values})
        summary = dict(variant=variant, dataset=dataset, subset=subset,
                       runs=len(values), missing_seeds=missing, complete=not missing)
        for metric in METRICS:
            scores = [row[metric] for row in values]
            summary[metric + "_mean"] = statistics.mean(scores)
            summary[metric + "_sample_std"] = statistics.stdev(scores) if len(scores) > 1 else None
        result.append(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/reepe_10epoch_ablations.json"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results_reepe/quality_10ep"))
    parser.add_argument("--dataset", choices=("davis", "snufilm", "ucf101", "middlebury"), required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--snu-mode", choices=("all", "easy", "medium", "hard", "extreme"), default="all")
    parser.add_argument("--image-size", choices=("256", "native"), default="256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("study_id") != "reepe_component_10ep_v1" or plan["common"]["epochs"] != 10:
        parser.error("use the independent 10-epoch REEPE study plan")
    root = args.run_root or Path(plan["run_root"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; pass --device cpu only for a deliberate CPU evaluation")
    configure_reproducibility(0, "strict")
    subsets = snufilm_modes(args.snu_mode) if args.dataset == "snufilm" else ("val" if args.dataset == "davis" else "test",)
    selections = select_runs(plan, args.configs, args.seeds)
    for subset in subsets:
        dataset, identities = make_dataset(args.dataset, args.dataset_path, subset, args.image_size)
        manifest_hash = ordered_hash([json.dumps(item, sort_keys=True) for item in identities])
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        destination = args.output_root / f"{args.dataset}-{subset}-{args.image_size}"
        rows = []
        for name, seed in selections:
            checkpoint, metadata = completed_checkpoint(root, name, seed, 10, plan["study_id"])
            model, info = load_inference_model(checkpoint, device, architecture="complex")
            if (info["epoch"] != 10 or info["seed"] != seed or info["configuration"] != name
                    or info["study_id"] != plan["study_id"]):
                raise ValueError(f"checkpoint identity differs from completed metadata: {checkpoint}")
            requested = dict(plan["common"], **plan["configurations"][name])
            for key, value in requested.items():
                if key != "entrypoint" and info["training_arguments"].get(key) != value:
                    raise ValueError(f"{checkpoint}: saved {key} differs from the frozen plan")
            identity = {"study_id": plan["study_id"], "protocol": PROTOCOL,
                        "variant": name, "seed": seed, "dataset": args.dataset, "subset": subset,
                        "image_size": args.image_size, "sample_manifest_sha256": manifest_hash,
                        "checkpoint_sha256": info["checkpoint_sha256"]}
            run_output = destination / name / f"seed-{seed}"
            summary_path = run_output / "summary.json"
            if summary_path.exists() and not args.overwrite:
                previous = json.loads(summary_path.read_text())
                if any(previous.get(key) != value for key, value in identity.items()):
                    raise ValueError(f"{summary_path}: different protocol/checkpoint; use a new output root")
                rows.append(previous)
                print(f"SKIP completed {name} seed={seed} {subset}", flush=True)
                del model
                continue
            samples = []
            def record(index, metrics):
                if not all(math.isfinite(metrics[key]) for key in METRICS):
                    raise ValueError(f"nonfinite metric at triplet {index}; inspect before reporting")
                samples.append({**identity, **identities[index], **metrics})
            print(f"Evaluating {name} seed={seed} {args.dataset}/{subset}", flush=True)
            metrics = evaluate(model, loader, device, sample_callback=record)
            summary = {**identity, **metrics, "status": "completed", "checkpoint": info,
                       "training_split_sha256": metadata["split_sample_sha256"],
                       "environment": environment_metadata(), "batch_size": 1,
                       "pce": metrics["pce"], "pce_definition": "internal coefficients; raw amplitude > 1e-4; RGB mean; radians",
                       "ssim_definition": "RGB; 11x11 Gaussian sigma=1.5; zero padding",
                       "lpips_definition": "AlexNet; RGB converted from [0,1] to [-1,1]"}
            write_csv(run_output / "samples.csv", samples)
            sequences = defaultdict(list)
            for sample in samples:
                sequences[sample["sequence_id"]].append(sample)
            write_csv(run_output / "sequences.csv", [
                {**identity, "sequence_id": sequence, "triplets": len(values),
                 **{metric: statistics.mean(item[metric] for item in values) for metric in METRICS}}
                for sequence, values in sequences.items()])
            write_json(summary_path, summary)
            rows.append(summary)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        # Include previously completed selections, so sequential subset invocations
        # build the same table as one full invocation. Never mix protocols/manifests.
        indexed = {(row["variant"], row["seed"]): row for row in rows}
        for path in destination.glob("*/seed-*/summary.json"):
            saved = json.loads(path.read_text())
            if (saved.get("protocol") == PROTOCOL and saved.get("study_id") == plan["study_id"]
                    and saved.get("sample_manifest_sha256") == manifest_hash
                    and saved.get("variant") in plan["configurations"] and saved.get("seed") in plan["seeds"]):
                indexed.setdefault((saved["variant"], saved["seed"]), saved)
        rows = list(indexed.values())
        write_csv(destination / "per_seed.csv", [{key: row[key] for key in
                  ("variant", "seed", "dataset", "subset", "samples", *METRICS, "checkpoint_sha256")} for row in rows])
        aggregates = summarize(rows, plan["seeds"])
        write_json(destination / "aggregates.json", aggregates)
        write_csv(destination / "aggregates.csv", aggregates)


if __name__ == "__main__":
    main()
