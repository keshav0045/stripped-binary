"""Function-name normalization and subtoken splitting.

Labels come straight from ELF symbol tables, so they need cleaning before they
are usable targets: compiler clone suffixes are stripped, runtime/CRT helpers
are dropped, and every name is canonicalized to lowercase snake_case subtokens.
"""

import re

# GCC/Clang clone suffixes: foo.isra.0, foo.constprop.3, foo.part.1, foo.lto_priv.0
_CLONE_SUFFIX = re.compile(r"\.(isra|constprop|part|lto_priv|llvm)\.\d+.*$")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# Functions every binary has, which say nothing about the program.
_RUNTIME = {
    "_start", "_init", "_fini", "main", "frame_dummy", "register_tm_clones",
    "deregister_tm_clones", "__do_global_dtors_aux", "__libc_csu_init",
    "__libc_csu_fini", "_dl_relocate_static_pie", "__x86.get_pc_thunk.bx",
}


def clean_symbol(raw):
    """Return the label for a raw symbol name, or None if it should be dropped."""
    if raw in _RUNTIME or raw.startswith("__"):
        return None
    if ".cold" in raw:  # split-off cold paths aren't whole functions
        return None
    name = _CLONE_SUFFIX.sub("", raw)
    if "." in name or "@" in name or not subtokens(name):
        return None
    return name


def subtokens(name):
    """parse_HTTPHeader2 -> ['parse', 'http', 'header', '2']"""
    out = []
    for part in re.split(r"[_\W]+", name):
        out.extend(t.lower() for t in _CAMEL.findall(part))
    return out


def canonical(name):
    """Comparison form used by exact-match: lowercase snake_case of subtokens."""
    return "_".join(subtokens(name))
