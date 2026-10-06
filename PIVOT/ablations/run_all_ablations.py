from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

SCRIPT_NAME="spherical_gradient_pinn.py"

def discover(base,names):
    folders=[]
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p/SCRIPT_NAME).is_file() and (not names or p.name in names): folders.append(p)
    if names:
        missing=sorted(set(names)-{p.name for p in folders})
        if missing: raise RuntimeError("unknown ablations: "+", ".join(missing))
    if not folders: raise RuntimeError("no ablation scripts found")
    return folders

def run_one(folder,repeat,args):
    run_root=folder/("validation" if args.validate_only else "runs")/f"repeat_{repeat:02d}"
    if args.rerun and run_root.exists():
        index=1
        while True:
            candidate=run_root.with_name(run_root.name+f"_rerun_{index:02d}")
            if not candidate.exists(): run_root=candidate; break
            index+=1
    done=run_root/"completed.json"
    if done.exists() and not args.rerun:
        data=json.loads(done.read_text(encoding="utf-8")); data["Skipped"]=True; return data
    if run_root.exists() and any(run_root.iterdir()) and not args.rerun:
        return {"Module":folder.name,"Repeat":repeat,"ExitCode":-2,"Status":"INCOMPLETE_EXISTS",
          "RunRoot":str(run_root),"Seconds":0.0,"Skipped":False}
    run_root.mkdir(parents=True,exist_ok=True)
    mode="switch_test" if args.validate_only else args.solver_mode
    command=[args.python,str(folder/SCRIPT_NAME),"--mode",mode,"--device",args.device,
             "--output-root",str(run_root)]
    if args.metrics_only: command.append("--metrics-only")
    env=os.environ.copy(); env["PYTHONUTF8"]="1"; env["PYTHONIOENCODING"]="utf-8"
    tic=time.perf_counter()
    with open(run_root/"run.log","w",encoding="utf-8",errors="replace") as log:
        proc=subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,env=env,check=False)
    elapsed=time.perf_counter()-tic
    data={"Module":folder.name,"Repeat":repeat,"ExitCode":proc.returncode,
      "Status":"PASS" if proc.returncode==0 else "FAIL","RunRoot":str(run_root.resolve()),
      "Seconds":elapsed,"Skipped":False,"Command":command}
    (run_root/("completed.json" if proc.returncode==0 else "failed.json")).write_text(
        json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    return data

def write_summary(base,rows):
    payload={"Total":len(rows),"Passed":sum(x["Status"]=="PASS" for x in rows),
      "Failed":sum(x["Status"] not in ("PASS",) for x in rows),"Runs":rows}
    (base/"run_all_summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    with open(base/"run_all_summary.txt","w",encoding="utf-8") as f:
        f.write("Module\tRepeat\tStatus\tExitCode\tSeconds\tRunRoot\n")
        for x in rows: f.write(f"{x['Module']}\t{x['Repeat']}\t{x['Status']}\t{x['ExitCode']}\t{x['Seconds']:.6f}\t{x['RunRoot']}\n")
    return payload

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--parallel",type=int,default=3)
    ap.add_argument("--solver-mode",choices=("configured","all"),default="configured")
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--python",default=sys.executable)
    ap.add_argument("--modules",nargs="*")
    ap.add_argument("--metrics-only",action="store_true")
    ap.add_argument("--validate-only",action="store_true")
    ap.add_argument("--rerun",action="store_true")
    args=ap.parse_args()
    if args.repeats<1 or args.parallel<1: raise ValueError("repeats and parallel must be positive")
    base=Path(__file__).resolve().parent; folders=discover(base,args.modules); rows=[]
    for repeat in range(1,args.repeats+1):
        print(f"repeat {repeat}/{args.repeats}: {len(folders)} ablations",flush=True)
        with ThreadPoolExecutor(max_workers=min(args.parallel,len(folders))) as pool:
            jobs={pool.submit(run_one,folder,repeat,args):folder.name for folder in folders}
            for future in as_completed(jobs):
                try: row=future.result()
                except Exception as exc:
                    row={"Module":jobs[future],"Repeat":repeat,"ExitCode":-1,"Status":"RUNNER_ERROR",
                      "RunRoot":"","Seconds":0.0,"Skipped":False,"Error":repr(exc)}
                rows.append(row); print(f"{row['Module']} repeat={repeat} {row['Status']} {row['Seconds']:.3f}s",flush=True)
        write_summary(base,rows)
    payload=write_summary(base,rows)
    print(json.dumps({k:payload[k] for k in ("Total","Passed","Failed")},ensure_ascii=False),flush=True)
    if payload["Failed"]: raise SystemExit(1)

if __name__=="__main__": main()
