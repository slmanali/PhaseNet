"""Profile the 12 completed models, sequentially, in separate subprocesses."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from tools.reepe_common import completed_checkpoint, select_runs, write_csv, write_json
from utils.model_loading import file_sha256


def profile_command(checkpoint, output, measurement, device, study_id, epochs):
    command = [sys.executable, "-m", "tools.benchmark_interpolation",
               "--checkpoint", str(checkpoint), "--output", str(output), "--device", device,
               "--study-id", study_id, "--expected-epochs", str(epochs)]
    for key, value in measurement.items():
        command.extend(("--" + key.replace("_", "-"), str(value)))
    return command


def flatten(row):
    flat = {key: row.get(key) for key in ("variant", "seed", "status", "error", "oom",
        "resolution", "precision", "device", "total_parameters", "trainable_parameters",
        "conv_macs", "conv_flops", "publication_gpu_measurement", "operation_coverage")}
    flat["checkpoint_sha256"] = row.get("model", {}).get("checkpoint_sha256")
    for scope in ("network_inference", "full_interpolation"):
        for key, value in row.get(scope, {}).items():
            if key not in {"latency_samples_ms", "cached", "placement"}:
                flat[scope + "_" + key] = value
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/reepe_profile_12models.json"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    for flag in ("resolution", "warmup", "repetitions"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.run_root or Path(plan["run_root"])
    destination = args.output_root or Path(plan["output_root"])
    measurement = dict(plan["measurement"])
    for key in ("resolution", "warmup", "repetitions"):
        if getattr(args, key) is not None:
            measurement[key] = getattr(args, key)
    rows = []
    for name, seed in select_runs(plan, args.configs, args.seeds):
        output = destination / name / f"seed-{seed}" / "profile.json"
        # A dry run lists the intended twelve calls even in a sparse checkout.
        checkpoint = root / name / f"seed-{seed}" / "checkpoint_last.pth"
        if args.dry_run:
            print(shlex.join(profile_command(checkpoint, output, measurement, args.device,
                                            plan["study_id"], plan["expected_epochs"])))
            continue
        try:
            checkpoint, _ = completed_checkpoint(root, name, seed, plan["expected_epochs"])
            sha = file_sha256(checkpoint)
            expected = {**measurement, "device": args.device, "study_id": plan["study_id"]}
            if output.exists() and not args.overwrite:
                previous = json.loads(output.read_text())
                matches = (all(previous.get(key) == value for key, value in expected.items())
                           and previous.get("model", {}).get("checkpoint_sha256") == sha)
                if previous.get("status") != "completed" or not matches:
                    raise ValueError(f"{output}: incomplete or different settings; choose a new output root or --overwrite")
                row = previous
                print(f"SKIP completed {name} seed={seed}", flush=True)
            else:
                command = profile_command(checkpoint, output, measurement, args.device,
                                          plan["study_id"], plan["expected_epochs"])
                if args.overwrite:
                    command.append("--overwrite")
                print(shlex.join(command), flush=True)
                result = subprocess.run(command, check=False)
                row = json.loads(output.read_text()) if output.exists() else {
                    "status": "failed", "error": f"profiler exited {result.returncode} without an output"}
                if result.returncode and row.get("status") == "completed":
                    row.update(status="failed", error=f"profiler exited {result.returncode}")
            if row.get("status") == "completed":
                identity = row.get("model", {})
                if identity.get("configuration") != name or identity.get("seed") != seed:
                    raise ValueError("embedded checkpoint identity differs from requested model/seed")
        except (OSError, ValueError) as error:
            row = {"status": "failed", "error": str(error)}
        row.update(variant=name, seed=seed)
        rows.append(row)
        write_json(destination / "profiles.json", rows)
        write_csv(destination / "profiles.csv", [flatten(item) for item in rows])
    if not args.dry_run:
        failures = sum(row["status"] != "completed" for row in rows)
        print(f"Profiles completed: {len(rows) - failures}/{len(rows)}. See {destination / 'profiles.csv'}")
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
