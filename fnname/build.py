"""Stage 1a: clone each project and build it under every toolchain config.

Build systems love to override CFLAGS, so instead of trusting them we put a
compiler wrapper first on PATH that appends the config's flags *last*
(`gcc "$@" -O2`): the final -O wins. Every ELF with a symbol table found in
the build tree is copied out next to a stripped twin.

    python -m fnname.build --projects configs/projects.yaml --work work [--only zlib lua]
"""

import argparse
import os
import shutil
import stat
import subprocess
from pathlib import Path

import yaml
from elftools.elf.elffile import ELFFile

BUILD_PRESETS = {
    "autotools": "([ -x configure ] || autoreconf -fi) && ./configure && make -j$(nproc)",
    "cmake": "cmake -B _build -DCMAKE_BUILD_TYPE=None && cmake --build _build -j$(nproc)",
    "make": "make -j$(nproc)",
}


def make_wrappers(work, cfg):
    bindir = work / "cc" / cfg["name"]
    bindir.mkdir(parents=True, exist_ok=True)
    real = shutil.which(cfg["cc"])
    for alias in ("cc", "gcc", "clang", cfg["cc"]):
        w = bindir / alias
        w.write_text(f'#!/bin/sh\nexec {real} "$@" {cfg["cflags"]}\n')
        w.chmod(w.stat().st_mode | stat.S_IEXEC)
    return bindir


def checkout(proj, src_root):
    dest = src_root / proj["name"]
    if not dest.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", str(proj["ref"]), proj["git"], str(dest)],
            check=True,
        )
    return dest


def has_symtab(path):
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"\x7fELF":
                return False
            f.seek(0)
            elf = ELFFile(f)
            return elf["e_type"] in ("ET_EXEC", "ET_DYN") and elf.get_section_by_name(".symtab") is not None
    except Exception:
        return False


def collect_binaries(tree, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    found = 0
    for p in tree.rglob("*"):
        if not p.is_file() or p.is_symlink() or not has_symtab(p):
            continue
        rel = str(p.relative_to(tree)).replace("/", "__")
        shutil.copy2(p, out_dir / rel)
        subprocess.run(["strip", "-o", str(out_dir / f"{rel}.stripped"), str(p)], check=True)
        found += 1
    return found


def build_one(proj, cfg, work):
    out_dir = work / "bins" / proj["name"] / cfg["name"]
    if (out_dir / ".done").exists():
        return "cached"
    src = checkout(proj, work / "src")
    tree = work / "build" / proj["name"] / cfg["name"]
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(src, tree, symlinks=True)

    wrappers = make_wrappers(work, cfg)
    env = dict(os.environ, CC="cc", CFLAGS=cfg["cflags"], PATH=f"{wrappers}:{os.environ['PATH']}")
    cmd = BUILD_PRESETS.get(proj["build"], proj["build"])
    log = work / "logs" / f"{proj['name']}.{cfg['name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        rc = subprocess.run(["bash", "-c", cmd], cwd=tree, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode

    # Partial builds still yield useful binaries, so collect regardless of rc.
    n = collect_binaries(tree, out_dir)
    shutil.rmtree(tree, ignore_errors=True)
    if n:
        (out_dir / ".done").write_text(f"rc={rc}\n")
    return f"rc={rc} binaries={n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", default="configs/projects.yaml")
    ap.add_argument("--work", default="work", type=Path)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.projects))
    for proj in spec["projects"]:
        if args.only and proj["name"] not in args.only:
            continue
        for cfg in spec["configs"]:
            try:
                status = build_one(proj, cfg, args.work.resolve())
            except subprocess.CalledProcessError as e:
                status = f"error: {e}"
            print(f"{proj['name']:<20} {cfg['name']:<10} {status}", flush=True)


if __name__ == "__main__":
    main()
