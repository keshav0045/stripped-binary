"""Prompt format shared by fine-tuning, local inference, and the zero-shot baseline."""

MAX_CODE_CHARS = 6000  # ~1.5-2K tokens; long functions keep their head (signature + setup)

TEMPLATE = "// Decompiled function (stripped binary)\n{code}\n// Original function name:"


def build_prompt(code):
    if len(code) > MAX_CODE_CHARS:
        code = code[:MAX_CODE_CHARS] + "\n// ... truncated"
    return TEMPLATE.format(code=code)
