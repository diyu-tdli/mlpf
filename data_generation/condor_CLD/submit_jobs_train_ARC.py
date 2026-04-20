#!/usr/bin/env python
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

# ____________________________________________________________________________________________________________


# ____________________________________________________________________________________________________________
def absoluteFilePaths(directory):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files


def collect_existing_outputs(outdir, arc=False, require_pandora=False):
    existing = {"05": set(), "arc": set()}

    mode_dirs = {"05": os.path.join(outdir, "05")}
    if arc:
        mode_dirs["arc"] = os.path.join(outdir, "arc")

    if not require_pandora:
        for mode, mode_dir in mode_dirs.items():
            for path in glob.glob(os.path.join(mode_dir, "*.parquet")):
                existing[mode].add(os.path.abspath(path))
        return existing

    required_fields = ["X_pandora", "pfo_calohit", "pfo_track"]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
        json.dump(
            {
                "mode_dirs": mode_dirs,
                "required_fields": required_fields,
            },
            tmp,
        )
        payload = tmp.name

    check_cmd = f"""
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh >/dev/null 2>&1
python3 - <<'PY'
import glob
import json
import pyarrow.parquet as pq

with open({payload!r}) as f:
    payload = json.load(f)

required_fields = set(payload["required_fields"])
valid = {{}}
for mode, mode_dir in payload["mode_dirs"].items():
    valid[mode] = []
    for path in glob.glob(mode_dir + "/*.parquet"):
        try:
            schema_fields = set(pq.read_schema(path).names)
        except Exception:
            continue
        if required_fields.issubset(schema_fields):
            valid[mode].append(path)

print(json.dumps(valid))
PY
"""

    try:
        result = subprocess.run(
            ["bash", "-lc", check_cmd],
            check=True,
            capture_output=True,
            text=True,
        )
        valid = json.loads(result.stdout)
        for mode, paths in valid.items():
            existing[mode] = {os.path.abspath(path) for path in paths}
    finally:
        try:
            os.unlink(payload)
        except OSError:
            pass

    return existing


# _____________________________________________________________________________________________________________
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        help="output directory ",
        default="/eos/experiment/fcc/ee/simulation/ClicDet/test/",
    )

    parser.add_argument(
        "--config",
        help="gun config file (has to be in gun/ directory)",
        default="config.gun",
    )
    
    parser.add_argument(
        "--sample",
        help="gun / p8_ee_tt_ecm365",
        default="gun",
    )
    parser.add_argument(
        "--cldgeo",
        help="which cld geometry version to use",
        default="CLD_o2_v06",
    )
    parser.add_argument(
        "--cldconfig",
        help="path to CLD config",
        default="",
    )

    parser.add_argument(
        "--condordir",
        help="output directory ",
        default="/eos/experiment/fcc/ee/simulation/ClicDet/test/",
    )
    parser.add_argument(
        "--gentracking",
        help="using tracking from gen ",
        default="False",
        action="store_true",
    )
    parser.add_argument(
        "--arc",
        help="generating also ARC data ",
        default="False",
        action="store_true",
    )
    parser.add_argument(
        "--pandora",
        help="require Pandora fields in parquet outputs and resubmit incomplete files",
        default="False",
        action="store_true",
    )
    parser.add_argument(
        "--debug-logs",
        help="write per-job condor .out/.err files to std/ instead of /dev/null",
        default="False",
        action="store_true",
    )

    parser.add_argument("--njobs", help="max number of jobs", default=2)

    parser.add_argument(
        "--nev", help="max number of events (-1 runs on all events)", default=-1
    )

    parser.add_argument(
        "--queue",
        help="queue for condor",
        choices=[
            "espresso",
            "microcentury",
            "longlunch",
            "workday",
            "tomorrow",
            "testmatch",
            "nextweek",
        ],
        default="longlunch",
    )

    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    condor_dir = os.path.abspath(args.condordir)
    config = args.config
    sample = args.sample
    cldgeo = args.cldgeo
    cldconfig = args.cldconfig
    gentracking = args.gentracking
    arc = args.arc
    pandora = args.pandora
    debug_logs = args.debug_logs
    njobs = int(args.njobs)
    nev = args.nev
    queue = args.queue
    homedir = "/eos/user/v/vriecher/mlpf_arc/mlpf"

    os.system("mkdir -p {}".format(outdir))
    os.makedirs(condor_dir, exist_ok=True)

    existing_outputs = collect_existing_outputs(outdir, arc=arc, require_pandora=pandora)
    print(
        "Found {} valid 05 parquet files{}".format(
            len(existing_outputs["05"]),
            " and {} valid ARC parquet files".format(len(existing_outputs["arc"])) if arc else "",
        )
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(script_dir, "run_sequence_CLD_train_ARC.sh")

    jobCount = 0

    submit_log_dir = os.path.join(script_dir, "std")
    os.makedirs(submit_log_dir, exist_ok=True)

    output_target = "/dev/null"
    error_target = "/dev/null"
    if debug_logs:
        output_target = os.path.join(submit_log_dir, "condor.$(ClusterId).$(ProcId).out")
        error_target = os.path.join(submit_log_dir, "condor.$(ClusterId).$(ProcId).err")

    cmdfile = """# here goes your shell script
executable    = {}

# here you specify where to put .log, .out and .err files
output                = {}
error                 = {}
log                   = {}/condor.$(ClusterId).log

+AccountingGroup = "group_u_CMST3.all"
+JobFlavour    = "{}"
""".format(
        script, output_target, error_target, submit_log_dir, queue
    )

    print(njobs)
    for seed in range(1, njobs + 1):
        seed = str(seed)
        output_file_05 = os.path.abspath(os.path.join(outdir, "05", f"pf_tree_{seed}.parquet"))
        output_file_arc = os.path.abspath(os.path.join(outdir, "arc", f"pf_tree_{seed}_arc.parquet"))

        missing_05 = output_file_05 not in existing_outputs["05"]
        missing_arc = arc and output_file_arc not in existing_outputs["arc"]

        if missing_05 or missing_arc:
            missing_parts = []
            if missing_05:
                missing_parts.append("05")
            if missing_arc:
                missing_parts.append("arc")
            print("{} : missing {}".format(seed, ",".join(missing_parts)))
            jobCount += 1

            args = [
            f"--homedir {homedir}",
            f"--guncard {config}",
            f"--nev {nev}",
            f"--seed {seed}",
            f"--outputdir {outdir}",
            f"--dir {condor_dir}",
            f"--sample {sample}",
            f"--cldgeo {cldgeo}",
            ]
            # Only pass cldconfig if provided
            if cldconfig:
                args.append(f"--pathcldconfig {cldconfig}")

            # Convert string booleans safely
            if gentracking:
                args.append("--gentracking")

            if arc:
                args.append("--arc")

            if pandora:
                args.append("--pandora")

            argts = " ".join(args)

            cmdfile += 'arguments="{}"\n'.format(argts)
            cmdfile += "queue\n"

            cmd = "rm -rf job*; ./{} {}".format(script, argts)
            if jobCount == 1:
                print("")
                print(cmd)

    with open(os.path.join(script_dir, "condor_{}.sub".format(sample)), "w") as f:
        f.write(cmdfile)

    ### submitting jobs
    if jobCount > 0:
        print("")
        print("[Submitting {} jobs] ... ".format(jobCount))
        os.system("condor_submit {}".format(os.path.join(script_dir, "condor_{}.sub".format(sample))))


# _______________________________________________________________________________________
if __name__ == "__main__":
    main()
