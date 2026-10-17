#!/usr/bin/env python3
"""Aggregate per-run JSON metrics over independent training seeds."""
import argparse, json, math, statistics
from collections import defaultdict
from pathlib import Path
METRICS=("psnr","ssim","lpips","pce")

def aggregate(records, expected_seeds=None):
    groups=defaultdict(list); failed=[]
    for r in records:
        key=(r["variant"],r["dataset"],r.get("resolution"),r.get("protocol"))
        if r.get("status") != "completed": failed.append(r); continue
        groups[key].append(r)
    out=[]
    for key, rows in sorted(groups.items(), key=str):
        item=dict(zip(("variant","dataset","resolution","protocol"),key))
        item["completed_runs"]=len(rows); item["seeds"]=sorted(r["seed"] for r in rows)
        item["missing_seeds"]=sorted(set(expected_seeds or [])-set(item["seeds"]))
        for metric in METRICS:
            vals=[float(r[metric]) for r in rows]
            item[metric+"_mean"]=statistics.mean(vals)
            item[metric+"_sample_std"]=statistics.stdev(vals) if len(vals)>1 else None
        out.append(item)
    return {"aggregates":out,"failed_or_incomplete":failed}

def paired(records, left, right):
    done={(r["variant"],r["dataset"],r["seed"],r.get("resolution"),r.get("protocol")):r
          for r in records if r.get("status")=="completed"}
    result=[]
    for key,a in done.items():
        if key[0]!=left: continue
        b=done.get((right,*key[1:]))
        if b: result.append({"dataset":key[1],"seed":key[2],
            **{m+"_difference":a[m]-b[m] for m in METRICS}})
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("inputs",nargs="+",type=Path)
    p.add_argument("--expected-seeds",nargs="+",type=int,default=[11,22,33])
    p.add_argument("--paired",nargs=2,metavar=("LEFT","RIGHT")); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); records=[]
    for path in a.inputs:
        data=json.loads(path.read_text()); records.extend(data if isinstance(data,list) else data.get("runs",[data]))
    result=aggregate(records,a.expected_seeds)
    if a.paired: result["paired_differences"]=paired(records,*a.paired)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
if __name__=="__main__": main()
