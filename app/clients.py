"""HTTP clients for Ollama (local) and NVIDIA NIM (cloud)."""

import asyncio
import json
import logging
from typing import Callable, Dict, List, Optional

import httpx
import numpy as np

from .config import cfg

log = logging.getLogger("HC01.clients")

# ─── Shared persistent clients ─────────────────────────────────────────────
# Created once per process; avoids TCP handshake overhead on every call.
_ollama_http: Optional[httpx.AsyncClient] = None
_nim_http: Optional[httpx.AsyncClient] = None


def _get_ollama_client() -> httpx.AsyncClient:
    global _ollama_http
    if _ollama_http is None or _ollama_http.is_closed:
        _ollama_http = httpx.AsyncClient(
            base_url=cfg.OLLAMA_BASE,
            timeout=httpx.Timeout(cfg.OLLAMA_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _ollama_http


def _get_nim_client() -> httpx.AsyncClient:
    global _nim_http
    if _nim_http is None or _nim_http.is_closed:
        _nim_http = httpx.AsyncClient(
            base_url=cfg.NIM_BASE,
            timeout=httpx.Timeout(cfg.NIM_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _nim_http


# ─── Ollama ────────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self) -> None:
        self._online: Optional[bool] = None

    async def is_online(self) -> bool:
        """Quick liveness check (cached per object lifetime)."""
        if self._online is not None:
            return self._online
        try:
            client = _get_ollama_client()
            r = await client.get("/api/tags", timeout=3.0)
            self._online = r.status_code == 200
        except Exception:
            self._online = False
        return self._online

    def invalidate_cache(self) -> None:
        self._online = None

    async def available_models(self) -> List[str]:
        try:
            client = _get_ollama_client()
            r = await client.get("/api/tags", timeout=5.0)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception as exc:
            log.warning("Ollama available_models failed: %s", exc)
            return []

    async def chat(
        self,
        model: str,
        messages: List[Dict],
        max_tokens: int = 512,
        on_chunk: Optional[Callable] = None,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            # think=False disables chain-of-thought on qwen3 models (no effect on others)
            "think": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        }
        full = ""
        client = _get_ollama_client()
        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    delta = obj.get("message", {}).get("content", "")
                    if delta:
                        full += delta
                        if on_chunk:
                            await on_chunk(delta, full)
                    if obj.get("done"):
                        break
                except json.JSONDecodeError:
                    pass
        return full

    @staticmethod
    def _clean_for_embed(text: str) -> str:
        """Normalise text to avoid bge-m3 NaN instability.
        Replaces Unicode symbols with ASCII equivalents, then strips remaining
        non-ASCII, and truncates to 512 chars (safe token budget for bge-m3).
        """
        replacements = [
            ("\u2265", ">="), ("\u2264", "<="), ("\u2260", "!="),
            ("\u2248", "~"),  ("\u00d7", "x"),  ("\u2192", "->"),
            ("\u2190", "<-"), ("\u00b1", "+/-"), ("\u00b0", " deg"),
            ("\u00b5", "u"),  ("\u03b1", "alpha"), ("\u03b2", "beta"),
            ("\u2013", "-"),  ("\u2014", "-"),  ("\u00e7", "c"),
            ("\u00e9", "e"),  ("\u00e8", "e"),  ("\u00e0", "a"),
            ("\u00fc", "u"),  ("\u00f6", "o"),  ("\u00e4", "a"),
            ("\u00a7", "Section "), ("\u2019", "'"), ("\u201c", '"'),
            ("\u201d", '"'),  ("\u2026", "..."),
        ]
        for src, dst in replacements:
            text = text.replace(src, dst)
        # Strip any remaining non-ASCII
        text = "".join(c for c in text if ord(c) < 128).strip()
        # Truncate to safe token budget (~512 chars ≈ 128 tokens)
        return text[:512]

    async def embed(self, text: str) -> np.ndarray:
        # Ollama ≥0.3 uses /api/embed with {"input": ...}; older uses /api/embeddings with {"prompt": ...}
        clean = self._clean_for_embed(text)
        client = _get_ollama_client()
        try:
            r = await client.post("/api/embed", json={"model": cfg.EMBED_MODEL, "input": clean}, timeout=30.0)
            r.raise_for_status()
            data = r.json()
            emb = data.get("embeddings", [[]])[0]
            if emb:
                return np.array(emb, dtype=np.float32)
        except Exception:
            pass
        # Fallback to legacy endpoint
        r = await client.post("/api/embeddings", json={"model": cfg.EMBED_MODEL, "prompt": clean}, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        if "embedding" not in data:
            raise RuntimeError(f"Ollama embed: no embedding in response: {list(data.keys())}")
        return np.array(data["embedding"], dtype=np.float32)


# ─── NVIDIA NIM ────────────────────────────────────────────────────────────

class NIMClient:
    def __init__(self) -> None:
        pass

    async def chat(
        self,
        model: str,
        messages: List[Dict],
        api_key: str,
        max_tokens: int = 1024,
        on_chunk: Optional[Callable] = None,
        temperature: float = 0.3,
    ) -> str:
        if not api_key:
            raise RuntimeError(
                "NIM API key not configured. Set NIM_API_KEY_CHIEF or "
                "NIM_API_KEY_FALLBACK in .env"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        last_exc: Exception = RuntimeError("NIM request not attempted")
        for attempt in range(cfg.NIM_RETRY_ATTEMPTS + 1):
            try:
                return await self._stream_request(headers, payload, on_chunk)
            except (httpx.RemoteProtocolError, httpx.ReadTimeout) as exc:
                last_exc = exc
                if attempt < cfg.NIM_RETRY_ATTEMPTS:
                    wait = 2 ** attempt
                    log.warning("NIM attempt %d failed (%s), retrying in %ds…", attempt + 1, exc, wait)
                    await asyncio.sleep(wait)
            except Exception as exc:
                raise  # Don't retry on auth errors, 4xx, etc.

        raise last_exc

    async def _stream_request(
        self,
        headers: Dict,
        payload: Dict,
        on_chunk: Optional[Callable],
    ) -> str:
        full = ""
        client = _get_nim_client()
        async with client.stream(
            "POST", "/chat/completions", json=payload, headers=headers
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                try:
                    detail = body.decode("utf-8", errors="replace")[:400]
                except Exception:
                    detail = repr(body[:200])
                raise RuntimeError(f"NIM HTTP {resp.status_code}: {detail}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        full += delta
                        if on_chunk:
                            await on_chunk(delta, full)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
        return full

    async def embed(
        self,
        texts: List[str],
        api_key: str,
        input_type: str = "passage",   # "passage" for docs, "query" for queries
    ) -> List[np.ndarray]:
        """
        NVIDIA NIM embeddings via nvidia/nv-embedqa-e5-v5 (1024-dim, ~0.5s/batch).
        input_type: "passage" when indexing documents, "query" when embedding a search query.
        """
        if not api_key:
            raise RuntimeError("NIM API key not configured for embeddings")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
        }
        client = _get_nim_client()
        resp = await client.post("/embeddings", json=payload, headers=headers, timeout=30.0)
        if resp.status_code != 200:
            raise RuntimeError(f"NIM embeddings HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json().get("data", [])
        if not data:
            raise RuntimeError("NIM embeddings response missing data")
        vectors = [np.array(item.get("embedding", []), dtype=np.float32) for item in data]
        if any(v.size == 0 for v in vectors):
            raise RuntimeError("NIM embeddings returned empty vector")
        return vectors


# ─── Singletons ────────────────────────────────────────────────────────────
ollama = OllamaClient()
nim = NIMClient()
