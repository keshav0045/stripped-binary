"""Stage 1b: run Ghidra headless over every stripped binary.

    python -m fnname.decompile --work work --jobs 8

Requires $GHIDRA_HOME (the Docker image sets it). Output goes to
work/decomp/<project>/<config>/<binary>.jsonl, one function per line.
"""

import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "ghidra_scripts"


def decompile(stripped, out, timeout):
    out.parent.mkdir(parents=True, exist_ok=True)
    headless = Path(os.environ["GHIDRA_HOME"]) / "support" / "analyzeHeadless"
    with tempfile.TemporaryDirectory() as proj_dir:
        cmd = [
            str(headless), proj_dir, "tmp",
            "-import", str(stripped),
            "-scriptPath", str(SCRIPTS),
            "-postScript", "ExportDecompiled.java", str(out) + ".part",
            "-deleteProject", "-analysisTimeoutPerFile", str(timeout),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
    part = Path(str(out) + ".part")
    if r.returncode == 0 and part.exists():
        part.rename(out)
        return f"ok {stripped}"
    return f"FAIL {stripped}: {r.stdout[-500:]}{r.stderr[-500:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work", type=Path)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--timeout", type=int, default=600, help="analysis seconds per binary")
    args = ap.parse_args()

    bins, decomp = args.work / "bins", args.work / "decomp"
    todo = []
    for stripped in sorted(bins.rglob("*.stripped")):
        rel = stripped.relative_to(bins)
        out = decomp / rel.parent / (rel.name.removesuffix(".stripped") + ".jsonl")
        if not out.exists():
            todo.append((stripped, out))
    print(f"{len(todo)} binaries to decompile")

    with ProcessPoolExecutor(args.jobs) as pool:
        futs = [pool.submit(decompile, s, o, args.timeout) for s, o in todo]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


if __name__ == "__main__":
    main()
