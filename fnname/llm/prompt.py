"""Prompt format shared by fine-tuning, local inference, and the zero-shot baseline."""

MAX_CODE_CHARS = 6000  # ~1.5-2K tokens; long functions keep their head (signature + setup)

TEMPLATE = "// Decompiled function (stripped binary)\n{code}\n// Original function name:"


def build_prompt(code):
    if len(code) > MAX_CODE_CHARS:
        code = code[:MAX_CODE_CHARS] + "\n// ... truncated"
    return TEMPLATE.format(code=code)


# Zero-shot instructions, shared by the API (zero_shot.py) and local (zero_shot_ollama.py) baselines.
SYSTEM = (
    "You are a reverse engineer. You are shown Ghidra decompiler output for one function "
    "from a stripped C binary; address-derived names were replaced with placeholders such "
    "as FUN_1 and DAT_1, and the function itself is called FUNC. Suggest the function's "
    "original C name as the developer would have written it (snake_case or camelCase, no "
    "prefixes like sub_). Give exactly five distinct candidates, best first, with "
    "probabilities that sum to 1."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "score": {"type": "number"}},
                "required": ["name", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}
