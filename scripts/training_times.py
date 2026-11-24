#!/usr/bin/env python3
"""Report Trainer invocation time and cumulative Slurm allocation time separately."""

import argparse
import json
from pathlib import Path
import re
import subprocess


def report(workspace, extra_jobs=()):
    jobs = {}
    for path in (workspace / "logs").glob("*.out"):
        with path.open(errors="replace") as handle:
            # Read only the header, even for very large output files.
            for _ in range(12):
                line = handle.readline()
                if not line:
                    break
                match = re.match(r"Job (\d+) on .*; run (.+?)\s*$", line)
                if match:
                    jobs.setdefault(Path(match[2]).name, set()).add(match[1])
                    break
    for item in extra_jobs:
        name, job_id = item.split("=", 1)
        if not job_id.isdigit():
            raise ValueError("Extra job must have the form RUN_NAME=NUMERIC_JOB_ID")
        jobs.setdefault(name, set()).add(job_id)

    paths = sorted((workspace / "runs").glob("*a100*/train_results.json"))
    ids = sorted({job for path in paths for job in jobs.get(path.parent.name, ())})
    accounting = {}
    if ids:
        # Invoke sacct directly, bypassing any shell aliases that change formatting.
        output = subprocess.check_output([
            "sacct", "-X", "-n", "-P", "-j", ",".join(ids),
            "--format=JobIDRaw,JobName%100,State,ElapsedRaw",
        ], text=True)
        for line in output.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 4 and fields[0] in ids:
                accounting[fields[0]] = fields[1:4]

    for path in paths:
        name = path.parent.name
        metrics = json.loads(path.read_text())
        run = json.loads((path.parent / "run.json").read_text())
        seconds = metrics["train_runtime"]
        print(f"{name}:\n  GPU: {run.get('device', 'unknown')}")
        print(f"  Final Trainer invocation: {seconds:.1f} s ({seconds / 60:.2f} min)")
        print(f"  Final epoch: {metrics.get('epoch')}")
        total, complete, counted = 0, bool(jobs.get(name)), 0
        for job_id in sorted(jobs.get(name, ()), key=int):
            record = accounting.get(job_id)
            if record is None:
                print(f"  Job {job_id}: accounting missing")
                complete = False
                continue
            job_name, state, elapsed = record
            if not job_name.endswith("-train"):
                print(f"  Job {job_id}: excluded ({job_name})")
                continue
            print(f"  Job {job_id}: {state}, {elapsed} allocated seconds")
            if state.split()[0].rstrip("+") in {"PENDING", "RUNNING", "COMPLETING", "SUSPENDED"}:
                complete = False
            try:
                total += int(elapsed)
                counted += 1
            except ValueError:
                complete = False
        if counted:
            label = "Total logged allocation time" if complete else "Partial logged allocation time"
            print(f"  {label}: {total} s ({total / 60:.2f} min; {total / 3600:.4f} h)")
        else:
            print("  Allocation total unavailable: no matched training-job accounting")
        print()
    print("Allocation time includes startup, training, saving and any evaluation.")
    print("Totals cover matched retained logs/extra IDs only; deleted logs can hide attempts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.home() / "nlbse27")
    parser.add_argument("--extra-job", action="append", default=[], metavar="RUN=JOBID",
                        help="Include an attempt whose output log is missing; may repeat")
    args = parser.parse_args()
    report(args.workspace, args.extra_job)
