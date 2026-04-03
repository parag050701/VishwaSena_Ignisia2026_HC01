"""ChromaDB + sentence-transformers RAG engine for HC01."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import chromadb
import numpy as np
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .corpus import GUIDELINE_CORPUS


@dataclass
class RAGResult:
    citation: str
    section: str
    text: str
    score: float


class RAGEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        import os
        _db_path = os.path.join(os.path.dirname(__file__), ".chroma_db")
        self.client = chromadb.PersistentClient(
            path=_db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="hc01_guidelines",
            metadata={"hnsw:space": "cosine"},
        )

        self._indexed = False
        self._cached_embeddings: np.ndarray | None = None
        self._cached_docs: List[Dict[str, str]] = []

    def build_corpus(self) -> int:
        docs = GUIDELINE_CORPUS
        if len(docs) < 5:
            raise ValueError(f"Corpus must contain at least 5 chunks, got {len(docs)}.")

        ids = [d["id"] for d in docs]
        documents = [d["text"] for d in docs]
        metadatas = [
            {
                "citation": d["citation"],
                "section": d["section"],
                "source_id": d["id"],
            }
            for d in docs
        ]

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        existing = self.collection.get(include=[])
        if existing.get("ids"):
            self.collection.delete(ids=existing["ids"])

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        self._cached_embeddings = embeddings
        self._cached_docs = docs
        self._indexed = True
        return len(docs)

    def query(self, text: str, n: int = 3) -> List[Dict[str, Any]]:
        if not self._indexed:
            self.build_corpus()

        n = max(1, min(int(n), 10))

        q_emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        res = self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        out: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            out.append(
                {
                    "citation": meta.get("citation", "Unknown Source"),
                    "section": meta.get("section", "Unknown Section"),
                    "text": doc,
                    "score": round(score, 4),
                }
            )

        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    def health(self) -> Dict[str, Any]:
        count = self.collection.count() if self.collection else 0
        return {
            "status": "ok",
            "corpus_size": count,
            "embedding_model": self.model_name,
            "device": self.device,
        }

    def visualize(self) -> Dict[str, Any]:
        if not self._indexed:
            self.build_corpus()

        if self._cached_embeddings is None:
            raise RuntimeError("Embeddings not cached.")

        points_2d: np.ndarray
        method = "umap"

        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=8,
                min_dist=0.2,
                metric="cosine",
                random_state=42,
            )
            points_2d = reducer.fit_transform(self._cached_embeddings)
        except Exception:
            method = "pca_fallback"
            from sklearn.decomposition import PCA

            points_2d = PCA(n_components=2, random_state=42).fit_transform(self._cached_embeddings)

        points = []
        for idx, item in enumerate(self._cached_docs):
            points.append(
                {
                    "id": item["id"],
                    "citation": item["citation"],
                    "section": item["section"],
                    "x": float(points_2d[idx, 0]),
                    "y": float(points_2d[idx, 1]),
                }
            )

        return {
            "status": "ok",
            "method": method,
            "points": points,
            "count": len(points),
        }
