"""Normalization of Ghidra decompiler output.

Ghidra embeds addresses everywhere (FUN_00101a30, DAT_00104010, LAB_...). They
are meaningless across binaries and let a model memorize layout, so they're
renamed to positional placeholders. The function's own name becomes FUNC.
"""

import hashlib
import re

_ADDR_SYM = re.compile(r"\b(FUN|DAT|LAB|PTR|UNK|switchD|caseD|thunk_FUN)_([0-9a-fA-F]{4,16})\b")
# Other address-suffixed labels, e.g. s_usage:_%s_00102004 -> s_usage:_%s
_ADDR_SUFFIX = re.compile(r"\b(\w+?)_[0-9a-fA-F]{8,16}\b")
_HEX = re.compile(r"0x[0-9a-fA-F]{5,}")
_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_STRING = re.compile(r'"(?:\\.|[^"\\])*"')
_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|0x[0-9a-fA-F]+|\d+|==|!=|<=|>=|->|&&|\|\||<<|>>|\S')


def normalize(code, self_name=None):
    """Replace address-derived identifiers with stable placeholders."""
    code = _COMMENT.sub("", code)
    if self_name:
        code = re.sub(rf"\b{re.escape(self_name)}\b", "FUNC", code)
    seen, per_kind = {}, {}

    def rename(m):
        key, kind = m.group(0), m.group(1)
        if key not in seen:
            per_kind[kind] = per_kind.get(kind, 0) + 1
            seen[key] = f"{kind}_{per_kind[kind]}"
        return seen[key]

    code = _ADDR_SYM.sub(rename, code)
    code = _ADDR_SUFFIX.sub(r"\1", code)
    code = _HEX.sub("ADDR", code)
    code = re.sub(r"[ \t]+", " ", code)
    return re.sub(r"\n\s*\n+", "\n", code).strip()


def code_hash(normalized):
    """Body hash used for de-duplication across binaries and projects."""
    body = re.sub(r"\s+", "", normalized)
    return hashlib.sha1(body.encode()).hexdigest()[:16]


def tokens(code):
    return _TOKEN.findall(code)


def string_literals(code):
    return [s[1:-1] for s in _STRING.findall(code)]
