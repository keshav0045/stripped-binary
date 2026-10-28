from fnname.metrics import score
from fnname.names import canonical, clean_symbol, subtokens
from fnname.normalize import code_hash, normalize


def test_subtokens():
    assert subtokens("parse_HTTPHeader2") == ["parse", "http", "header", "2"]
    assert subtokens("readFileContents") == ["read", "file", "contents"]
    assert canonical("ParseHttpHeader") == canonical("parse_http_header")


def test_clean_symbol():
    assert clean_symbol("inflate_fast.isra.0") == "inflate_fast"
    assert clean_symbol("deflate.part.3") == "deflate"
    assert clean_symbol("main") is None
    assert clean_symbol("__libc_csu_init") is None
    assert clean_symbol("foo.cold") is None
    assert clean_symbol("frame_dummy") is None


def test_normalize_placeholders_are_positional():
    a = "int FUN_00101a30(void) { FUN_00102000(DAT_00104010); return FUN_00102000(0x401234); }"
    b = "int FUN_00203b40(void) { FUN_00205000(DAT_00207010); return FUN_00205000(0x509abc); }"
    na = normalize(a, "FUN_00101a30")
    assert na == normalize(b, "FUN_00203b40")
    assert na.startswith("int FUNC(void)") and "FUN_1(DAT_1)" in na and "ADDR" in na
    assert code_hash(na) == code_hash(normalize(b, "FUN_00203b40"))


def test_normalize_strips_comments_and_string_label_addresses():
    code = "/* WARNING: x */\nvoid f(void) { puts(s_usage_00102004); }"
    assert normalize(code) == "void f(void) { puts(s_usage); }"


def test_metrics():
    m = score([
        ("parse_http_header", ["parseHttpHeader", "x"]),      # exact
        ("read_config_file", ["load_config", "read_config_file"]),  # top-5 only
        ("free_buffer", []),                                   # no prediction
    ])
    assert m["n"] == 3
    assert abs(m["exact@1"] - 1 / 3) < 1e-9
    assert abs(m["exact@5"] - 2 / 3) < 1e-9
    # tp: 3 + 1 (config); predicted: 3 + 2; true: 3 + 3 + 2
    assert abs(m["precision"] - 4 / 5) < 1e-9
    assert abs(m["recall"] - 4 / 8) < 1e-9
