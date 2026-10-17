#!/usr/bin/env python3
"""Sequential, resumable launcher. It never runs two GPU jobs concurrently."""
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path


def command(entry, common, config, name, seed, args):
    merged = dict(common); merged.update(config)
    program = merged.pop("entrypoint"); merged.pop("initialization", None)
    merged["dataset_path"] = args.dataset_path
    merged.update(seed=seed, deterministic=args.deterministic,
                  run_root=str(args.run_root), config_name=name)
    checkpoint = args.run_root / name / f"seed-{seed}" / "checkpoint_last.pth"
    metadata = checkpoint.parent / "metadata.json"
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        if previous.get("status") == "completed":
            if not checkpoint.exists() and not (checkpoint.parent / "model_final_state_dict.pth").exists():
                raise FileNotFoundError(f"completed run has no checkpoint or final weights: {checkpoint.parent}")
            return None
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"incomplete run has no resumable checkpoint: {checkpoint}. "
                "Recover this file from the training machine, or restart only this "
                "configuration and seed with a NEW --run-root. The existing run is preserved."
            )
        old_arguments = previous.get("arguments", {})
        for key, value in merged.items():
            if key in {"run_root", "dataset_path"}:
                continue
            if key not in old_arguments or old_arguments[key] != value:
                raise ValueError(f"resume configuration differs for {key}: "
                                 f"saved={old_arguments.get(key)!r}, requested={value!r}")
        for key in ("device", "num_workers", "davis_root", "overfit_samples",
                    "max_steps_per_epoch", "log_interval"):
            if old_arguments.get(key) is not None:
                merged[key] = old_arguments[key]
        merged["resume"] = str(checkpoint)
    elif checkpoint.exists():
        raise FileNotFoundError(f"refusing to resume without run metadata: {metadata}")
    result = [sys.executable, program]
    for key, value in merged.items():
        result.extend((f"--{key.replace('_', '-')}", str(value)))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, default=Path("experiments/article_seeded.json"))
    p.add_argument("--run-root", type=Path, default=Path("runs_seeded"))
    p.add_argument("--dataset-path")
    p.add_argument("--seeds", type=int, nargs="+", default=[11,22,33])
    p.add_argument("--configs", nargs="+", help="Run only these named configurations from the plan.")
    p.add_argument("--deterministic", choices=("strict","warn","off"), default="strict")
    p.add_argument("--dry-run", action="store_true")
    a=p.parse_args(); plan=json.loads(a.plan.read_text())
    if a.configs:
        unknown = set(a.configs) - set(plan["configurations"])
        if unknown:
            p.error(f"unknown configurations: {sorted(unknown)}")
    selected = a.configs or list(plan["configurations"])
    print(f"Selected configurations: {', '.join(selected)}; "
          f"seeds: {', '.join(str(seed) for seed in a.seeds)}", flush=True)
    a.dataset_path = a.dataset_path or plan["dataset_path"]
    for name, config in plan["configurations"].items():
        if name not in selected:
            continue
        for seed in a.seeds:
            cmd=command(config["entrypoint"], plan["common"], config, name, seed, a)
            if cmd is None:
                print(f"SKIP completed {name} seed={seed}"); continue
            print(shlex.join(cmd), flush=True)
            if not a.dry_run:
                env=os.environ.copy(); env["PYTHONHASHSEED"]=str(seed)
                env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                subprocess.run(cmd, check=True, env=env)
if __name__ == "__main__": main()
