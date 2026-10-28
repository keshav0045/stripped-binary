"""Split -> kNN -> evaluate -> serve on a small synthetic dataset."""

import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEMPLATES = {
    "parse_header": 'int FUNC(char *param_1) { char *p = strchr(param_1, 0x3a); if (p == 0) return -1; *p = 0; return strspn(p + 1, " \\t"); }',
    "read_file": 'long FUNC(char *param_1) { FILE *f = fopen(param_1, "rb"); fseek(f, 0, 2); long n = ftell(f); fread(DAT_1, 1, n, f); fclose(f); return n; }',
    "hash_string": "uint FUNC(byte *param_1) { uint h = 0x1505; while (*param_1) { h = h * 0x21 + *param_1; param_1++; } return h; }",
    "free_list": "void FUNC(long *param_1) { long *n; while (param_1) { n = (long *)*param_1; free(param_1); param_1 = n; } }",
    "print_usage": 'void FUNC(void) { fprintf(stderr, "usage: %s [options] file\\n", DAT_1); exit(1); }',
}


def make_dataset(path):
    rows = []
    for p in range(12):
        for cfg in ("gcc-O0", "gcc-O2", "clang-O0", "clang-O2"):
            for name, code in TEMPLATES.items():
                variant = code.replace("param_1", f"param_1 /*{p}*/") if p % 2 else code
                body = f"{variant}\n// {cfg} {p}"  # unique body per project/config
                rows.append({"name": name, "raw_symbol": name, "code": body, "code_hash": f"{p}{cfg}{name}",
                             "n_instructions": 20 + len(code) // 10, "name_in_strings": False,
                             "project": f"proj{p}", "config": cfg, "binary": "a"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def run(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def test_end_to_end(tmp_path):
    data = tmp_path / "functions.jsonl"
    make_dataset(data)
    run("fnname.split", "--data", str(data), "--out", str(tmp_path), "--test-projects", "proj0", "proj1")
    split = json.loads((tmp_path / "split_projects.json").read_text())
    assert split["proj0"] == "test"
    test_projects = {json.loads(l)["project"] for l in open(tmp_path / "test.jsonl")}
    train_projects = {json.loads(l)["project"] for l in open(tmp_path / "train.jsonl")}
    assert test_projects and not test_projects & train_projects

    model = tmp_path / "knn.pkl"
    run("fnname.baselines.knn", "--train", str(tmp_path / "train.jsonl"), "--test", str(tmp_path / "test.jsonl"),
        "--out", str(tmp_path / "knn.jsonl"), "--save", str(model))
    report = run("fnname.evaluate", "--test", str(tmp_path / "test.jsonl"), "--failures", "3",
                 str(tmp_path / "knn.jsonl"), str(tmp_path / "frequency.jsonl"))
    assert "## Overall" in report and "| knn" in report and "O2" in report
    knn_row = next(l for l in report.splitlines() if l.startswith("| knn "))
    assert float(knn_row.split("|")[4]) > 90  # exact@5 on near-duplicate templates

    namer = pickle.load(open(model, "rb"))
    preds = namer.predict([TEMPLATES["hash_string"]])[0]
    assert preds[0]["name"] == "hash_string" and abs(sum(p["score"] for p in preds) - 1) < 1e-6


def test_serve(tmp_path, monkeypatch):
    from fnname.baselines.knn import KNNNamer
    names = list(TEMPLATES) * 3
    model = KNNNamer().fit([TEMPLATES[n] for n in names], names)
    pickle.dump(model, open(tmp_path / "knn.pkl", "wb"))
    monkeypatch.setenv("FNNAME_MODEL", str(tmp_path / "knn.pkl"))
    sys.modules.pop("serve.app", None)
    from fastapi.testclient import TestClient
    from serve.app import app

    client = TestClient(app)
    raw = TEMPLATES["read_file"].replace("FUNC", "FUN_00101a30").replace("DAT_1", "DAT_00104010")
    r = client.post("/predict", json={"code": raw})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "knn" and body["candidates"][0]["name"] == "read_file"
    assert client.get("/").status_code == 200
