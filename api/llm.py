"""
Minimal LLM endpoint that serves a small BERT-based question-answering model
using ONNX Runtime + tokenizers directly (no transformers/optimum/PyTorch).

Model: deepset/minilm-uncased-squad2  (~80 MB ONNX)
  - A distilled MiniLM fine-tuned on SQuAD 2.0 for extractive QA.
  - You provide a context paragraph and a question; the model extracts the answer.

Usage:
  POST /llm/ask
  {
    "question": "What is the capital of France?",
    "context": "France is a country in Europe. Its capital is Paris."
  }
"""

from __future__ import annotations

import math
import os
import time

# Use /tmp for caches in serverless environments (Vercel, Lambda, etc.)
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf_home/hub")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="LLM Question Answering API", version="1.0.0")

MODEL_REPO = "deepset/minilm-uncased-squad2"

# ---------------------------------------------------------------------------
# Lazy-loaded model (singleton -- only downloaded/loaded once per cold start)
# ---------------------------------------------------------------------------
_session = None
_tokenizer = None


def _load_model():
    """Download the ONNX model + tokenizer on first request."""
    global _session, _tokenizer
    if _session is not None:
        return _session, _tokenizer

    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="onnx/model.onnx",
    )

    _session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    tok_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="tokenizer.json",
    )
    _tokenizer = Tokenizer.from_file(tok_path)

    return _session, _tokenizer


def _softmax(logits: list[float]) -> list[float]:
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _run_qa(question: str, context: str) -> dict:
    """Tokenize, run ONNX inference, decode the answer span."""
    session, tokenizer = _load_model()

    encoding = tokenizer.encode(question, context)
    input_ids = encoding.ids
    attention_mask = encoding.attention_mask
    token_type_ids = encoding.type_ids

    import numpy as np

    feeds = {
        "input_ids": np.array([input_ids], dtype=np.int64),
        "attention_mask": np.array([attention_mask], dtype=np.int64),
        "token_type_ids": np.array([token_type_ids], dtype=np.int64),
    }

    start_logits, end_logits = session.run(None, feeds)
    start_logits = start_logits[0].tolist()
    end_logits = end_logits[0].tolist()

    # Mask out question tokens (type_id == 0) so the answer comes from context
    for i, tid in enumerate(token_type_ids):
        if tid == 0:
            start_logits[i] = -1e9
            end_logits[i] = -1e9

    start_probs = _softmax(start_logits)
    end_probs = _softmax(end_logits)

    start_idx = max(range(len(start_probs)), key=lambda i: start_probs[i])
    end_idx = max(
        range(start_idx, min(start_idx + 64, len(end_probs))),
        key=lambda i: end_probs[i],
    )

    score = start_probs[start_idx] * end_probs[end_idx]

    answer_ids = input_ids[start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    offsets = encoding.offsets
    char_start = offsets[start_idx][0] if start_idx < len(offsets) else 0
    char_end = offsets[end_idx][1] if end_idx < len(offsets) else 0

    return {
        "answer": answer if answer else "",
        "score": score,
        "start": char_start,
        "end": char_end,
    }


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=500, description="The question to answer"
    )
    context: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The context paragraph containing the answer",
    )


class AskResponse(BaseModel):
    answer: str
    score: float
    start: int
    end: int
    model: str
    inference_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/llm")
def llm_root():
    return {
        "service": "LLM Question Answering",
        "model": MODEL_REPO,
        "endpoints": {
            "POST /llm/ask": "Answer a question given a context paragraph",
            "GET  /llm/health": "Check if the model is loaded and ready",
        },
        "example": {
            "question": "What is the capital of France?",
            "context": "France is a country in Europe. Its capital is Paris.",
        },
    }


@app.post("/llm/ask", response_model=AskResponse)
def ask(body: AskRequest):
    """Extract an answer from the context using the BERT QA model."""
    try:
        t0 = time.perf_counter()
        result = _run_qa(body.question, body.context)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference failed: {exc}")

    return AskResponse(
        answer=result["answer"],
        score=round(result["score"], 4),
        start=result["start"],
        end=result["end"],
        model=MODEL_REPO,
        inference_ms=round(elapsed_ms, 2),
    )


@app.get("/llm/health")
def health():
    """Report whether the model is loaded."""
    return {
        "status": "ready" if _session is not None else "cold",
        "model": MODEL_REPO,
        "runtime": "onnxruntime (CPU)",
    }
