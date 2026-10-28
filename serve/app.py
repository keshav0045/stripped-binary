"""Stage 5: HTTP service for name suggestions.

    FNNAME_MODEL=models/knn.pkl uvicorn serve.app:app --port 8000
    FNNAME_MODEL=models/lora    uvicorn serve.app:app --port 8000   # needs torch+peft

POST /predict {"code": "<Ghidra decompiler output>"} -> top-5 names with confidence.
Raw Ghidra output is fine: it is normalized here exactly as in training.
"""

import os
import pickle
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fnname.normalize import normalize

MODEL_PATH = Path(os.environ.get("FNNAME_MODEL", "models/knn.pkl"))
STATIC = Path(__file__).parent / "static"
_SIGNATURE_NAME = re.compile(r"^[\w\s\*]*?\b(FUN_[0-9a-fA-F]+|sub_[0-9a-fA-F]+)\s*\(", re.M)


def load_model(path):
    if path.suffix == ".pkl":
        return pickle.load(open(path, "rb")), "knn"
    from fnname.llm.predict_lora import LoraNamer
    return LoraNamer(str(path)), "lora"


model, model_kind = load_model(MODEL_PATH)
app = FastAPI(title="fnname", description="Suggest names for stripped-binary functions")


class PredictRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=200_000)
    top: int = Field(5, ge=1, le=10)


class Candidate(BaseModel):
    name: str
    score: float


class PredictResponse(BaseModel):
    model: str
    candidates: list[Candidate]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    m = _SIGNATURE_NAME.search(req.code)
    code = normalize(req.code, m.group(1) if m else None)
    return {"model": model_kind, "candidates": model.predict([code], top=req.top)[0]}


@app.get("/health")
def health():
    return {"ok": True, "model": model_kind, "path": str(MODEL_PATH)}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
