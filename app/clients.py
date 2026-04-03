import json
from typing import Callable, Dict, List, Optional

import httpx
import numpy as np

from .config import cfg


class OllamaClient:
    def __init__(self):
        self.base = cfg.OLLAMA_BASE
        self._online: Optional[bool] = None

    async def is_online(self) -> bool:
        if self._online is not None:
            return self._online
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.base}/api/tags")
                self._online = r.status_code == 200
        except Exception:
            self._online = False
        return self._online

    async def available_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.base}/api/tags")
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    async def chat(self, model: str, messages: List[Dict], max_tokens: int = 512, on_chunk: Optional[Callable] = None) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        }
        full = ""
        async with httpx.AsyncClient(timeout=cfg.OLLAMA_TIMEOUT) as client:
            async with client.stream("POST", f"{self.base}/api/chat", json=payload) as resp:
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

    async def embed(self, text: str) -> np.ndarray:
        payload = {"model": cfg.EMBED_MODEL, "prompt": text}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self.base}/api/embeddings", json=payload)
            r.raise_for_status()
            return np.array(r.json()["embedding"], dtype=np.float32)


class NIMClient:
    def __init__(self):
        self.base = cfg.NIM_BASE

    async def chat(self, model: str, messages: List[Dict], api_key: str, max_tokens: int = 1024, on_chunk: Optional[Callable] = None) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens, "stream": True}
        full = ""
        async with httpx.AsyncClient(timeout=cfg.NIM_TIMEOUT) as client:
            async with client.stream("POST", f"{self.base}/chat/completions", json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise RuntimeError(f"NIM {resp.status_code}: {body[:200].decode()}")
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


ollama = OllamaClient()
nim = NIMClient()
