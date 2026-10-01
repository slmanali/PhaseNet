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
    if checkpoint.exists() and metadata.exists():
        status = json.loads(metadata.read_text()).get("status")
        if status == "completed": return None
        merged["resume"] = str(checkpoint)
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
    p.add_argument("--deterministic", choices=("strict","warn","off"), default="strict")
    p.add_argument("--dry-run", action="store_true")
    a=p.parse_args(); plan=json.loads(a.plan.read_text())
    a.dataset_path = a.dataset_path or plan["dataset_path"]
    for name, config in plan["configurations"].items():
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
