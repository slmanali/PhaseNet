#!/usr/bin/env python3
"""Machine-readable PhaseNet latency, allocator-memory, and convolution subtotal."""
import argparse, gc, json, statistics, time
from pathlib import Path
import torch
from torch import nn
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from utils.inference import interpolate_endpoints
from utils.reproducibility import configure_reproducibility, environment_metadata
from utils.model_loading import load_inference_model
from utils.evaluation_protocol import write_csv, write_json

EXCLUDED=("steerable-pyramid FFT/decomposition/reconstruction", "batch normalization",
          "activation", "bilinear interpolation", "trigonometric and elementwise operations")

def count_conv_macs(model, call):
    total=0; handles=[]
    def hook(module, inputs, output):
        nonlocal total
        y=output; batch=y.shape[0]; out_h,out_w=y.shape[-2:]
        kh,kw=module.kernel_size
        total += batch*out_h*out_w*module.out_channels*(module.in_channels//module.groups)*kh*kw
    for module in model.modules():
        if isinstance(module,nn.Conv2d): handles.append(module.register_forward_hook(hook))
    try: call()
    finally:
        for h in handles: h.remove()
    return total

def percentile95(values):
    values=sorted(values); return values[max(0, int(.95*len(values)+.999999)-1)]


def measure(call, device, warmup, repetitions, device_events):
    """Warm up, then measure one RGB midpoint, without retaining prior outputs."""
    if warmup < 1 or repetitions < 1:
        raise ValueError("warmup and repetitions must be positive")
    for _ in range(warmup):
        output = call()
        del output
    gc.collect()
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = None
    samples = []
    for _ in range(repetitions):
        if cuda and device_events:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            output = call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            if cuda:
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = call()
            if cuda:
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000)
        del output
    peak = torch.cuda.max_memory_allocated(device) if cuda else None
    return {
        "timing_method": "CUDA events" if cuda and device_events else "synchronized wall clock",
        "latency_mean_ms": statistics.mean(samples), "latency_median_ms": statistics.median(samples),
        "latency_p95_ms": percentile95(samples), "latency_samples_ms": samples,
        "baseline_allocated_mib": baseline / 2**20 if cuda else None,
        "peak_allocated_mib": peak / 2**20 if cuda else None,
        "incremental_peak_allocated_mib": (peak - baseline) / 2**20 if cuda else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if cuda else None,
    }


@torch.inference_mode()
def benchmark(model, architecture, device, resolution, tile_size, warmup, repetitions):
    pyramid = SCFpyr_PyTorch(height=12, nbands=4, scale_factor=2**.5, device=device)
    start = torch.rand(1, 3, resolution, resolution, device=device)
    end = torch.rand_like(start)
    tile = tile_size or None
    complex_model = architecture == "complex"
    probe, prepared = interpolate_endpoints(model, pyramid, start, end,
        complex_model=complex_model, tile_size=tile, return_prepared=True)
    del probe

    def network():
        return [model(item[0], item[1], tile_size=tile) if complex_model else model(item[0])
                for item in prepared]

    macs = count_conv_macs(model, network)
    network_result = measure(network, device, warmup, repetitions, device_events=True)
    # Cached pyramids belong only to network-only measurement. Keeping them here
    # would incorrectly increase the complete-pipeline GPU peak.
    del network, prepared
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def full():
        return interpolate_endpoints(model, pyramid, start, end,
                                     complex_model=complex_model, tile_size=tile)

    full_result = measure(full, device, warmup, repetitions, device_events=False)
    return {"conv_macs": macs, "conv_flops": 2 * macs,
            "network_inference": {"cached": "endpoint pyramids and coefficient inputs; no reconstruction",
                                  **network_result},
            "full_interpolation": {"placement": "RGB endpoints and output on device; includes both pyramids and reconstruction",
                                   **full_result}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture", choices=("real", "complex"))
    parser.add_argument("--feature-dim", type=int)
    parser.add_argument("--convolution", choices=("complex", "separate_real", "unrestricted_real"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--deterministic", choices=("strict", "warn", "off"), default="strict")
    parser.add_argument("--study-id", default="computer_optics_optional_cost_v1")
    parser.add_argument("--expected-epochs", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; use another path or --overwrite")
    if args.resolution < 256 or args.warmup < 1 or args.repetitions < 1 or args.tile_size < 0:
        parser.error("resolution must be >=256; warmup/repetitions positive; tile size nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA unavailable; --device cpu is a debugging option, not GPU evidence")
        torch.cuda.set_device(device)
    configure_reproducibility(0, args.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    row = {"study_id": args.study_id, "status": "running", "checkpoint": str(args.checkpoint),
           "resolution": args.resolution, "batch_size": 1, "precision": args.precision,
           "device": str(device), "tile_size": args.tile_size, "warmup": args.warmup,
           "repetitions": args.repetitions, "deterministic": args.deterministic,
           "compiled": False, "input_seed": 0, "tf32_matmul": False, "tf32_cudnn": False,
           "environment": environment_metadata(), "publication_gpu_measurement": device.type == "cuda",
           "operation_coverage": "real Conv2d subtotal; actual calls, all RGB channels; 2 FLOPs per MAC",
           "excluded_operations": list(EXCLUDED) + ["convolution bias additions"],
           "memory_definition": "PyTorch allocator bytes, MiB=2^20; excludes driver/other-process allocations",
           "excluded_from_timing": ["model loading", "disk I/O", "host-device transfer", "quality metrics"],
           "oom": False}
    row["measured_device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    try:
        model, info = load_inference_model(args.checkpoint, device, args.architecture,
            feature_dim=args.feature_dim, convolution=args.convolution)
        if args.expected_epochs is not None and info["epoch"] != args.expected_epochs:
            raise ValueError("checkpoint epoch differs from the requested completed training epoch")
        if info["architecture"] == "real" and args.tile_size:
            raise ValueError("tiling is implemented only for the complex model")
        row.update(model=info, architecture=info["architecture"],
                   total_parameters=sum(p.numel() for p in model.parameters()),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        row.update(benchmark(model, info["architecture"], device, args.resolution,
                             args.tile_size, args.warmup, args.repetitions))
        row["status"] = "completed"
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}",
                   oom=isinstance(error, torch.cuda.OutOfMemoryError))
    write_json(args.output, row)
    write_csv(args.output.with_suffix(".csv"), [row])
    print(f"{row['status']}: {args.output}", flush=True)
    if row["status"] != "completed":
        raise SystemExit(row["error"])


if __name__ == "__main__":
    main()
