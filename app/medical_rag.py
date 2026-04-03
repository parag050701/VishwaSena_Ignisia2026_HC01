"""Medical guideline RAG with hybrid BM25 + FAISS retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from .clients import ollama


class MedicalRAG:
    """Build and query a local hybrid index of clinical guideline chunks."""

    def __init__(self, db_path: str, collection_name: str = "medical_guidelines") -> None:
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.db_path.mkdir(parents=True, exist_ok=True)

        self._meta_path = self.db_path / f"{collection_name}.chunks.json"
        self._emb_path = self.db_path / f"{collection_name}.embeddings.npy"
        self._faiss_path = self.db_path / f"{collection_name}.faiss.index"

        self._chunks: List[Dict] = []
        self._chunk_texts: List[str] = []
        self._tokenized: List[List[str]] = []
        self._bm25: BM25Okapi | None = None
        self._embeddings: np.ndarray | None = None
        self._faiss_index: faiss.IndexFlatIP | None = None

    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = await asyncio.gather(*(ollama.embed(t) for t in texts))
        return [v.tolist() for v in vectors]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in "".join(c.lower() if c.isalnum() else " " for c in text).split() if tok]

    @staticmethod
    def _normalize_rows(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return mat / norms

    def _load_artifacts(self) -> bool:
        if not (self._meta_path.exists() and self._emb_path.exists() and self._faiss_path.exists()):
            return False

        self._chunks = json.loads(self._meta_path.read_text(encoding="utf-8"))
        self._chunk_texts = [c["text"] for c in self._chunks]
        self._tokenized = [self._tokenize(t) for t in self._chunk_texts]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        self._embeddings = np.load(self._emb_path)
        self._faiss_index = faiss.read_index(str(self._faiss_path))
        return True

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
        text = " ".join((text or "").split())
        if not text:
            return []
        chunks: List[str] = []
        i = 0
        while i < len(text):
            chunks.append(text[i : i + chunk_size])
            if i + chunk_size >= len(text):
                break
            i += max(1, chunk_size - overlap)
        return chunks

    @staticmethod
    def _read_pdf_text(pdf_path: Path) -> str:
        reader = PdfReader(str(pdf_path))
        pages: List[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    async def build_from_pdf_dir(self, pdf_dir: str) -> int:
        """Ingest all PDFs and rebuild hybrid BM25 + FAISS artifacts."""
        base = Path(pdf_dir)
        if not base.exists():
            return 0

        pdfs = sorted(base.glob("*.pdf"))
        if not pdfs:
            return 0

        docs: List[str] = []
        chunk_meta: List[Dict] = []

        for pdf_path in pdfs:
            text = self._read_pdf_text(pdf_path)
            text_chunks = self._chunk_text(text)
            for idx, chunk in enumerate(text_chunks):
                stable = hashlib.md5(f"{pdf_path.name}:{idx}:{chunk[:120]}".encode("utf-8")).hexdigest()
                docs.append(chunk)
                chunk_meta.append(
                    {
                        "id": f"{pdf_path.stem}-{idx}-{stable[:8]}",
                        "source": pdf_path.name,
                        "section": f"chunk-{idx}",
                        "guideline_id": pdf_path.stem.lower().replace(" ", "-"),
                        "text": chunk,
                    }
                )

        if not docs:
            return 0

        embeddings = []
        batch_size = 24
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            embeddings.extend(await self._embed_texts(batch))

        emb = np.array(embeddings, dtype=np.float32)
        emb = self._normalize_rows(emb)

        faiss_index = faiss.IndexFlatIP(emb.shape[1])
        faiss_index.add(emb)

        self._meta_path.write_text(json.dumps(chunk_meta, ensure_ascii=False), encoding="utf-8")
        np.save(self._emb_path, emb)
        faiss.write_index(faiss_index, str(self._faiss_path))

        # Warm in-memory structures.
        self._chunks = chunk_meta
        self._chunk_texts = docs
        self._tokenized = [self._tokenize(t) for t in docs]
        self._bm25 = BM25Okapi(self._tokenized)
        self._embeddings = emb
        self._faiss_index = faiss_index
        return len(docs)

    async def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        if not query_text.strip():
            return []

        if self._faiss_index is None or self._bm25 is None:
            if not self._load_artifacts():
                return []

        if not self._chunks or self._faiss_index is None or self._bm25 is None:
            return []

        # Dense FAISS retrieval.
        q_emb = np.array(await self._embed_texts([query_text]), dtype=np.float32)
        q_emb = self._normalize_rows(q_emb)
        dense_k = min(max(top_k * 3, 10), len(self._chunks))
        sims, dense_idx = self._faiss_index.search(q_emb, dense_k)
        dense_pairs: List[Tuple[int, float]] = [
            (int(idx), float(score)) for idx, score in zip(dense_idx[0], sims[0]) if idx >= 0
        ]

        # Sparse BM25 retrieval.
        bm25_scores = self._bm25.get_scores(self._tokenize(query_text))
        sparse_k = min(max(top_k * 3, 10), len(self._chunks))
        sparse_idx = np.argsort(bm25_scores)[::-1][:sparse_k]
        sparse_pairs: List[Tuple[int, float]] = [(int(i), float(bm25_scores[i])) for i in sparse_idx]

        # Reciprocal Rank Fusion for robust hybrid ranking.
        fused: Dict[int, float] = {}
        rrf_k = 60.0
        for rank, (idx, _) in enumerate(dense_pairs, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (idx, _) in enumerate(sparse_pairs, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

        out: List[Dict] = []
        for idx, score in ranked:
            c = self._chunks[idx]
            out.append(
                {
                    "id": c.get("guideline_id", "pdf-guideline"),
                    "source": c.get("source", "medical-guidelines-pdf"),
                    "section": c.get("section", "chunk"),
                    "text": c.get("text", ""),
                    "score": round(float(score), 4),
                    "keywords": [],
                }
            )
        return out


_MEDICAL_RAG: MedicalRAG | None = None


def get_medical_rag(db_path: str) -> MedicalRAG:
    global _MEDICAL_RAG
    if _MEDICAL_RAG is None:
        _MEDICAL_RAG = MedicalRAG(db_path=db_path)
    return _MEDICAL_RAG
