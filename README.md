# fnname: predicting function names in stripped binaries

When a binary is stripped, Ghidra shows `FUN_00101a30` where the source said
`parse_http_header`. This project builds a dataset from open-source C code
compiled with and without symbols, then trains models that read Ghidra's
decompiler output and suggest the original name, with a confidence score.

```
open-source C ──► gcc/clang × -O0/-O2 ──► unstripped ─┐   (labels: .symtab)
                                       └► stripped ──► Ghidra headless ──► decompiled C
                                                                         │
            pair by address ◄────────────────────────────────────────────┘
                   │
       split by project + cross-project dedupe ──► train / val / test
                   │
    frequency · TF-IDF kNN · LoRA code LLM · zero-shot API model ──► evaluate ──► FastAPI + Ghidra plugin
```

## Results

> Placeholder until the full dataset is built. Every number here must come from `make eval` on held-out projects.

| model | exact@1 | exact@5 | subtoken F1 |
|---|---|---|---|
| frequency prior | – | – | – |
| TF-IDF kNN | – | – | – |
| zero-shot (Claude Opus 5.5) | – | – | – |
| Qwen2.5-Coder-1.5B + LoRA | – | – | – |

`make eval` also prints breakdowns by optimization level, by function size and
by compiler config, followed by sample failures.

## Design decisions (each one is material for the write-up)

**Labels for free, cleaned carefully.** Names come from `.symtab` of the
unstripped twin (`fnname/elf_symbols.py`). GCC clone suffixes (`.isra.0`,
`.constprop.1`, `.part.2`) are stripped, `.cold` fragments are dropped, and so
are CRT/runtime helpers (`_start`, `frame_dummy`, `main`, `__*`).

**Matching stripped to unstripped.** `strip` leaves PT_LOAD segments where they
were, so Ghidra reports each function's *file offset* in the stripped binary,
and `pair.py` converts it to a vaddr using the program headers. Ghidra's image
base for PIE binaries doesn't matter.

**Forcing optimization levels.** Build systems override `CFLAGS` all the time.
`build.py` puts a wrapper `cc` first on `PATH` that runs `gcc "$@" -O2`, and
the last `-O` flag wins, so each binary really has the level its config says.

**Normalization.** Ghidra embeds addresses everywhere. `FUN_00102000` and
`DAT_00104010` become positional `FUN_1` and `DAT_1`, the target function becomes
`FUNC`, and long hex constants become `ADDR`. Without this, a model can memorize
memory layout.

**Leakage, three ways.**
1. *Split by project, not by function.* Otherwise zlib's `inflate` at -O0 sits in
   train and the same `inflate` at -O2 sits in test.
2. *Cross-project dedupe.* Projects vendor shared code (gnulib, getopt, md5).
   `split.py` drops any val/test function whose normalized body also appears in
   train, and reports how many it dropped.
3. *Names that survive stripping.* Exported `.dynsym` symbols remain in stripped
   shared libraries, so Ghidra already knows those names. `pair.py` drops these
   functions. It also flags functions whose name appears in a string literal
   (from `assert` or `__func__`). That's a real signal, but report it separately.

**Prediction files are model-agnostic.** Every model writes
`{"id", "preds": [{"name", "score"}]}`, and `evaluate.py` scores them all the
same way. Use `--intersect` to compare everything on the zero-shot sample.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test                      # unit + synthetic end-to-end tests

# Stage 1: data (in the x86-64 pipeline container: gcc, clang, Ghidra)
make image
make shell                     # then, inside the container:
make build                     # configs/projects.yaml × 4 configs
make decompile JOBS=8          # Ghidra headless, ExportDecompiled.java
make dataset                   # pair + split + dedupe, prints drop statistics

# Stage 2: baselines
make baseline                  # preds/knn.jsonl, preds/frequency.jsonl, models/knn.pkl

# Stage 3: models  (pip install -r requirements-llm.txt)
make lora                      # needs a CUDA GPU with about 16 GB
make zeroshot                  # needs Anthropic API credentials; 2K-sample by default

# Stage 4 and 5
make eval
make serve                     # http://localhost:8000
docker build -t fnname . && docker run -p 8000:8000 fnname
```

**Ghidra plugin:** add `ghidra_scripts/` in Script Manager, then press
Ctrl-Shift-N inside a function (`SuggestName.java`). It sends the decompiled
code to the service, lets you pick a suggestion, renames the function and
records every suggestion in a comment.

**On Apple Silicon:** the pipeline image is amd64, so it runs emulated and is
slow. Building and decompiling about 300 projects is best done on an x86-64
Linux cloud VM.

## Scaling the dataset

`configs/projects.yaml` starts with 20 projects. Grow it toward about 300. For
each project, check `work/logs/<project>.<config>.log`. A project that builds
partially still contributes the binaries it produced. The data-engineering
story to tell is how many projects built cleanly, how many functions each stage
dropped and why, and how much vendored code the dedupe removed.

## Layout

```
configs/projects.yaml      projects + toolchain configs
docker/Dockerfile.pipeline gcc + clang + Ghidra image
ghidra_scripts/            ExportDecompiled.java (headless), SuggestName.java (plugin)
fnname/
  build.py decompile.py pair.py split.py   stage 1
  names.py normalize.py elf_symbols.py     labels, normalization, ground truth
  baselines/knn.py                         stage 2
  llm/                                     stage 3: prompt, LoRA train/predict, zero-shot
  metrics.py evaluate.py                   stage 4
serve/                     FastAPI app + web UI (stage 5)
tests/
```
