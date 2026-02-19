"""
Minimal LLM endpoint that serves a small BERT-based question-answering model
using ONNX Runtime for lightweight inference (no PyTorch required).

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

import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="LLM Question Answering API", version="1.0.0")

# ---------------------------------------------------------------------------
# Lazy-loaded pipeline (singleton so the model is only downloaded/loaded once
# per cold start).
# ---------------------------------------------------------------------------
_pipeline = None


def _get_pipeline():
    """Load the QA pipeline on first request, reuse afterwards."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from optimum.onnxruntime import ORTModelForQuestionAnswering
    from transformers import AutoTokenizer, pipeline

    model_name = "deepset/minilm-uncased-squad2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ORTModelForQuestionAnswering.from_pretrained(
        model_name,
        export=True,  # auto-export to ONNX if no ONNX weights exist
    )

    _pipeline = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
    )
    return _pipeline


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
        "model": "deepset/minilm-uncased-squad2",
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
        qa = _get_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {exc}")

    start_time = time.perf_counter()
    result = qa(question=body.question, context=body.context)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return AskResponse(
        answer=result["answer"],
        score=round(result["score"], 4),
        start=result["start"],
        end=result["end"],
        model="deepset/minilm-uncased-squad2",
        inference_ms=round(elapsed_ms, 2),
    )


@app.get("/llm/health")
def health():
    """Report whether the model is loaded."""
    return {
        "status": "ready" if _pipeline is not None else "cold",
        "model": "deepset/minilm-uncased-squad2",
        "runtime": "onnxruntime",
    }
