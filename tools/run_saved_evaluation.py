"""Sequential evaluation of saved models. The legacy stage leaves evaluators unchanged."""
import argparse
from collections import defaultdict
import csv
import importlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

from utils.evaluation_protocol import make_dataset, sha256, source_hashes, verify_source_lock, write_csv, write_json

METRICS = ("psnr", "ssim", "lpips", "pce")


def jobs_from_plan(plan, configs=None, seeds=None, datasets=None):
    configs, seeds, datasets = configs or list(plan["models"]), seeds or plan["seeds"], datasets or list(plan["datasets"])
    for selected, allowed in ((configs, plan["models"]), (seeds, plan["seeds"]), (datasets, plan["datasets"])):
        if len(set(selected)) != len(selected) or set(selected) - set(allowed):
            raise ValueError("duplicate or unknown selection")
    return [{"variant": name, "seed": seed, "dataset": dataset, "subset": subset}
            for name in configs for seed in seeds for dataset in datasets
            for subset in plan["datasets"][dataset]["subsets"]]


def check_checkpoint(plan, job):
    folder = Path(plan["run_root"]) / job["variant"] / f"seed-{job['seed']}"
    metadata = json.loads((folder / "metadata.json").read_text())
    if (metadata.get("status") != "completed" or metadata.get("completed_epochs") != plan["expected_epochs"]
            or metadata.get("seed") != job["seed"] or metadata.get("configuration") != job["variant"]):
        raise ValueError(f"incomplete or mismatched training metadata: {folder}")
    if metadata["arguments"]["feature_dim"] != plan["models"][job["variant"]]["feature_dim"]:
        raise ValueError(f"model width differs from plan: {folder}")
    # Bare weights avoid the full-checkpoint pickle/RNG incompatibility in the
    # old real-model loader. This does not change inference or trained weights.
    checkpoint = folder / "model_final_state_dict.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"final exported weights required: {checkpoint}")
    return checkpoint, metadata


def run_job(plan, job, destination, device, overwrite=False):
    import torch
    from torch.utils.data import DataLoader
    from utils.reproducibility import configure_reproducibility, environment_metadata

    if plan["stage"] == "legacy":
        hashes = verify_source_lock(plan)
    else:
        hashes = source_hashes(plan["evaluation_sources"])
    checkpoint, metadata = check_checkpoint(plan, job)
    configure_reproducibility(0, plan["deterministic"])
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu only for deliberate CPU evaluation")
    spec = plan["datasets"][job["dataset"]]
    dataset, manifest, protocol = make_dataset(spec, job["subset"])
    identity = {**job, "study_id": plan["study_id"], "stage": plan["stage"],
                "checkpoint_sha256": sha256(checkpoint), "source_sha256": hashes,
                "sample_manifest_sha256": protocol["sample_manifest_sha256"],
                "device": str(device), "deterministic": plan["deterministic"],
                "tile_size": plan.get("tile_size", 0)}
    summary_path = destination / "summary.json"
    if summary_path.exists() and not overwrite:
        saved = json.loads(summary_path.read_text())
        if saved.get("status") != "completed" or any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("existing evaluation differs; use a new output root or --overwrite")
        if not (destination / "samples.csv").is_file():
            raise FileNotFoundError("completed summary has no per-sample results")
        if plan["stage"] == "corrected" and not (destination / "diagnostics.json").is_file():
            raise FileNotFoundError("completed corrected summary has no phase diagnostics")
        return saved
    architecture = plan["models"][job["variant"]]
    module = importlib.import_module("test" if architecture["architecture"] == "real" else "test_complex_safe_baseline")
    model, _ = module.load_model(checkpoint, device, architecture["feature_dim"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    write_json(destination / "manifest.json", {"protocol": protocol, "samples": manifest})
    arguments = dict(model_name=job["variant"], mode=job["subset"],
                     scalar_metrics_path=destination / "raw_samples.csv")
    if architecture["architecture"] == "complex":
        arguments["tile_size"] = plan.get("tile_size", 0) or None
    if plan["stage"] == "corrected":
        from utils.journal_diagnostics import JournalDiagnostics
        diagnostics = JournalDiagnostics(device)
        arguments["diagnostics"] = diagnostics
    metrics = module.evaluate(model, loader, device, **arguments)
    if metrics["samples"] != len(manifest):
        raise ValueError("evaluator did not return the complete requested sample set")
    with (destination / "raw_samples.csv").open(newline="") as stream:
        scalar_rows = list(csv.DictReader(stream))
    if len(scalar_rows) != len(manifest) or [int(row["sample_index"]) for row in scalar_rows] != list(range(len(manifest))):
        raise ValueError("per-sample indices do not match the dataset manifest")
    rows = []
    for index, row in enumerate(scalar_rows):
        values = {key: float(row[key]) for key in METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"nonfinite metric at sample {index}; inspect before aggregation")
        rows.append({**job, "study_id": plan["study_id"], "stage": plan["stage"],
                     **manifest[index], **values})
    write_csv(destination / "samples.csv", rows)
    if plan["stage"] == "corrected":
        diagnostics.save(destination, rows)
    summary = {**identity, **metrics, "status": "completed", "protocol": protocol,
               "checkpoint": str(checkpoint), "training_metadata": metadata,
               "environment": environment_metadata(), "batch_size": 1,
               "metric_note": "legacy arithmetic, including known real PCE issues" if plan["stage"] == "legacy"
               else "corrected RGB PCE and raw amplitude mask; see diagnostics.json"}
    write_json(summary_path, summary)
    return summary


def aggregate(rows, seeds):
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            groups[(row["variant"], row["dataset"], row["subset"])].append(row)
    result = []
    for (variant, dataset, subset), values in sorted(groups.items()):
        if len({r["seed"] for r in values}) != len(values):
            raise ValueError("duplicate seed results")
        if len({r["sample_manifest_sha256"] for r in values}) != 1:
            raise ValueError("seeds evaluated on different data")
        item = {"variant": variant, "dataset": dataset, "subset": subset,
                "runs": len(values), "missing_seeds": sorted(set(seeds) - {r["seed"] for r in values})}
        for metric in METRICS:
            numbers = [r[metric] for r in values]
            item[metric + "_mean"] = statistics.mean(numbers)
            item[metric + "_sample_std"] = statistics.stdev(numbers) if len(numbers) > 1 else None
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/computer_optics_eval_12_legacy.json"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--dataset-path", action="append", default=[], help="Override as dataset=path; repeat if needed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.job:
        request = json.loads(args.job.read_text())
        run_job(request["plan"], request["job"], Path(request["destination"]), request["device"], request["overwrite"])
        return
    plan = json.loads(args.plan.read_text())
    if plan["stage"] == "legacy":
        verify_source_lock(plan)
    if args.run_root:
        plan["run_root"] = str(args.run_root)
    for override in args.dataset_path:
        name, path = override.split("=", 1)
        if name not in plan["datasets"]:
            parser.error(f"unknown dataset {name}")
        plan["datasets"][name]["path"] = path
    destination = args.output_root or Path(plan["output_root"])
    jobs = jobs_from_plan(plan, args.configs, args.seeds, args.datasets)
    print(f"{len(jobs)} evaluations; no training is performed.", flush=True)
    if not args.dry_run:
        # Fail before launching any job if a selected model is incomplete.
        for job in {(j["variant"], j["seed"]): j for j in jobs}.values():
            check_checkpoint(plan, job)
        for name in {job["dataset"] for job in jobs}:
            if not Path(plan["datasets"][name]["path"]).is_dir():
                raise FileNotFoundError(plan["datasets"][name]["path"])
    rows = []
    for number, job in enumerate(jobs, 1):
        folder = destination / job["variant"] / f"seed-{job['seed']}" / job["dataset"] / job["subset"]
        request_path = folder / "job.json"
        command = [sys.executable, "-m", "tools.run_saved_evaluation", "--job", str(request_path)]
        print(f"[{number}/{len(jobs)}] {job['variant']} seed={job['seed']} {job['dataset']}/{job['subset']}", flush=True)
        if args.dry_run:
            continue
        write_json(request_path, {"plan": plan, "job": job, "destination": str(folder),
                                  "device": args.device, "overwrite": args.overwrite})
        print(shlex.join(command), flush=True)
        with (folder / "evaluation.log").open("a") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"evaluation failed; see {folder / 'evaluation.log'}. Completed outputs remain intact.")
        rows.append(json.loads((folder / "summary.json").read_text()))
        write_json(destination / "selected_runs.json", rows)
        write_csv(destination / "selected_runs.csv", [{key: row[key] for key in
            ("variant", "seed", "dataset", "subset", "samples", *METRICS, "checkpoint_sha256")} for row in rows])
        write_csv(destination / "aggregates.csv", aggregate(rows, plan["seeds"]))


if __name__ == "__main__":
    main()
