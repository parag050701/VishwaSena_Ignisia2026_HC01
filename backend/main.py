"""FastAPI app exposing enterprise RAG endpoints for HC01."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag_engine import RAGEngine


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    n_results: int = Field(default=3, ge=1, le=10)


class RAGResultItem(BaseModel):
    citation: str
    section: str
    text: str
    score: float


class RAGQueryResponse(BaseModel):
    results: List[RAGResultItem]
    query_time_ms: float


class RAGHealthResponse(BaseModel):
    status: str
    corpus_size: int
    embedding_model: str


rag_engine: RAGEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    rag_engine = RAGEngine(model_name="all-MiniLM-L6-v2")
    rag_engine.build_corpus()
    yield


app = FastAPI(title="HC01 RAG Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "null",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/rag/health", response_model=RAGHealthResponse)
def rag_health() -> RAGHealthResponse:
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    h = rag_engine.health()
    return RAGHealthResponse(
        status=h["status"],
        corpus_size=h["corpus_size"],
        embedding_model=h["embedding_model"],
    )


@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(payload: RAGQueryRequest) -> RAGQueryResponse:
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    t0 = time.perf_counter()
    results = rag_engine.query(payload.query, n=payload.n_results)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return RAGQueryResponse(results=[RAGResultItem(**r) for r in results], query_time_ms=round(elapsed_ms, 2))


@app.get("/rag/visualize")
def rag_visualize():
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return rag_engine.visualize()
