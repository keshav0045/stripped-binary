"""Stage 1c: join Ghidra output with ground-truth symbols into one dataset.

    python -m fnname.pair --work work --out data/functions.jsonl

Dropped (and counted, because the counts belong in the write-up):
  - no symbol at that address, or a runtime/CRT helper
  - Ghidra already knew the name (exported .dynsym symbols survive strip)
  - duplicate (name, body) within the same project+config, e.g. a static
    helper linked into several of the project's executables
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from .elf_symbols import function_symbols, offset_to_vaddr
from .names import clean_symbol
from .normalize import code_hash, normalize, string_literals


def pair_binary(unstripped, stripped, decomp_jsonl, stats):
    syms = function_symbols(unstripped)
    to_vaddr = offset_to_vaddr(stripped)
    for line in open(decomp_jsonl):
        rec = json.loads(line)
        stats["ghidra_functions"] += 1
        sym = syms.get(to_vaddr(rec["file_offset"]))
        if sym is None:
            stats["drop_no_symbol"] += 1
            continue
        name = clean_symbol(sym[0])
        if name is None:
            stats["drop_runtime_or_unusable"] += 1
            continue
        if not rec["ghidra_name"].startswith("FUN_"):
            stats["drop_name_visible_in_stripped"] += 1
            continue
        code = normalize(rec["code"], rec["ghidra_name"])
        yield {
            "name": name,
            "raw_symbol": sym[0],
            "code": code,
            "code_hash": code_hash(code),
            "n_instructions": rec["n_instructions"],
            # assert()/__func__ put the real name in .rodata; legit signal, but report it
            "name_in_strings": any(name in s for s in string_literals(code)),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work", type=Path)
    ap.add_argument("--out", default="data/functions.jsonl", type=Path)
    args = ap.parse_args()

    stats, seen = Counter(), set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as out:
        for decomp_jsonl in sorted((args.work / "decomp").rglob("*.jsonl")):
            project, config = decomp_jsonl.parts[-3], decomp_jsonl.parts[-2]
            binary = decomp_jsonl.name.removesuffix(".jsonl")
            bin_dir = args.work / "bins" / project / config
            for ex in pair_binary(bin_dir / binary, bin_dir / f"{binary}.stripped", decomp_jsonl, stats):
                key = (project, config, ex["name"], ex["code_hash"])
                if key in seen:
                    stats["drop_dup_within_project"] += 1
                    continue
                seen.add(key)
                ex.update(project=project, config=config, binary=binary)
                out.write(json.dumps(ex) + "\n")
                stats["kept"] += 1

    for k, v in sorted(stats.items()):
        print(f"{k:<32} {v}")


if __name__ == "__main__":
    main()
