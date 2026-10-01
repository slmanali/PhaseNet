#!/usr/bin/env python3
"""Machine-readable PhaseNet latency, allocator-memory, and convolution subtotal."""
import argparse, csv, json, statistics, time
from pathlib import Path
import torch
from torch import nn
from net.phasenet import PhaseNet
from net.complex_phasenet_safe_baseline import ComplexPhaseNetSafe
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from utils.inference import interpolate_endpoints
from utils.reproducibility import configure_reproducibility, load_checkpoint

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
    for _ in range(warmup): call()
    if device.type=="cuda": torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    samples=[]
    for _ in range(repetitions):
        if device_events:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record(); out=call(); b.record(); b.synchronize(); samples.append(a.elapsed_time(b))
        else:
            if device.type=="cuda": torch.cuda.synchronize(device)
            start=time.perf_counter(); out=call()
            if device.type=="cuda": torch.cuda.synchronize(device)
            samples.append((time.perf_counter()-start)*1000)
        del out
    return {"latency_mean_ms":statistics.mean(samples),"latency_median_ms":statistics.median(samples),
            "latency_p95_ms":percentile95(samples),
            "peak_allocated_mib":torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else None,
            "peak_reserved_mib":torch.cuda.max_memory_reserved(device)/2**20 if device.type=="cuda" else None}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--architecture",choices=("real","complex"),required=True)
    p.add_argument("--convolution",choices=("complex","separate_real","unrestricted_real"),default="complex")
    p.add_argument("--feature-dim",type=int,default=64); p.add_argument("--resolution",type=int,default=256)
    p.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision",choices=("fp32",),default="fp32"); p.add_argument("--tile-size",type=int,default=0)
    p.add_argument("--warmup",type=int,default=30); p.add_argument("--repetitions",type=int,default=100)
    p.add_argument("--deterministic",choices=("strict","warn","off"),default="strict")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); configure_reproducibility(0,a.deterministic); device=torch.device(a.device)
    model=(PhaseNet(a.feature_dim) if a.architecture=="real" else ComplexPhaseNetSafe(a.feature_dim,convolution=a.convolution)).to(device)
    load_checkpoint(a.checkpoint,model,map_location=device); model.eval()
    pyramid=SCFpyr_PyTorch(height=12,nbands=4,scale_factor=2**.5,device=device)
    start=torch.rand(1,3,a.resolution,a.resolution,device=device); end=torch.rand_like(start)
    tile=a.tile_size or None; full=lambda: interpolate_endpoints(model,pyramid,start,end,complex_model=a.architecture=="complex",tile_size=tile)
    # Network scope caches endpoint decomposition/coefficient preparation and excludes reconstruction.
    _, prepared=interpolate_endpoints(model,pyramid,start,end,complex_model=a.architecture=="complex",tile_size=tile,return_prepared=True)
    def network():
        result=[]
        for item in prepared:
            result.append(model(item[0],item[1],tile_size=tile) if a.architecture=="complex" else model(item[0]))
        return result
    row={"checkpoint":str(a.checkpoint),"architecture":a.architecture,"convolution":a.convolution,
         "resolution":a.resolution,"batch_size":1,"precision":a.precision,"device":str(device),
         "tile_size":a.tile_size,"padding":"pyramid internal","warmup":a.warmup,"repetitions":a.repetitions,
         "deterministic":a.deterministic,"compiled":False,"tf32":getattr(torch.backends.cuda.matmul,"allow_tf32",None),
         "cudnn_benchmark":torch.backends.cudnn.benchmark,"total_parameters":sum(x.numel() for x in model.parameters()),
         "trainable_parameters":sum(x.numel() for x in model.parameters() if x.requires_grad),
         "operation_coverage":"real Conv2d MAC subtotal only; observed calls including reused modules and RGB channels",
         "excluded_operations":list(EXCLUDED),"oom":False}
    try:
        with torch.inference_mode():
            macs=count_conv_macs(model,network); row.update(conv_macs=macs,conv_flops=2*macs)
            row["network_inference"]={"cached":"endpoint pyramids and prepared coefficients; reconstruction excluded",**measure(network,device,a.warmup,a.repetitions,device.type=="cuda")}
            row["full_interpolation"]={"placement":"input and output on selected device; no host transfer",**measure(full,device,a.warmup,a.repetitions,False)}
    except torch.cuda.OutOfMemoryError as e:
        row.update(oom=True,error=str(e)); torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(row,indent=2)+"\n")
    with a.output.with_suffix('.csv').open('w',newline='') as f:
        flat={k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in row.items()}; w=csv.DictWriter(f,fieldnames=flat); w.writeheader(); w.writerow(flat)
if __name__=='__main__': main()
