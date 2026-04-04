"""
Voice Workflow Engine for HC01.
Pipeline: Audio → STT → Intent → Patient Lookup → Diagnosis → Response → TTS
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .audit import log_voice_session
from .clients import nim, ollama
from .config import cfg

log = logging.getLogger("HC01.voice")


# ─── STT: NIM Whisper Large v3 (cloud) with faster-whisper fallback ──────────

_stt_model = None
_stt_lock  = asyncio.Lock()


async def _get_stt_model():
    """Lazy-load local faster-whisper (only used if NIM STT key absent)."""
    global _stt_model
    if _stt_model is not None:
        return _stt_model
    async with _stt_lock:
        if _stt_model is not None:
            return _stt_model
        try:
            from faster_whisper import WhisperModel
            device = cfg.STT_DEVICE if cfg.STT_DEVICE == "cpu" else "auto"
            ctype  = "float16" if device != "cpu" else "int8"
            log.info("Loading faster-whisper %s on %s…", cfg.STT_MODEL, device)
            _stt_model = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: WhisperModel(cfg.STT_MODEL, device=device, compute_type=ctype),
            )
            log.info("faster-whisper loaded")
        except Exception as exc:
            log.warning("faster-whisper not available: %s", exc)
            _stt_model = None
    return _stt_model


async def _nim_transcribe(audio_bytes: bytes) -> Optional[str]:
    """Transcribe via NVIDIA NIM Whisper Large v3 API."""
    import httpx
    headers = {"Authorization": f"Bearer {cfg.NIM_STT_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{cfg.NIM_SPEECH_BASE}/audio/transcriptions",
                headers=headers,
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"language": cfg.STT_LANGUAGE or "en"},
            )
            if resp.status_code != 200:
                log.warning("NIM STT HTTP %d: %s", resp.status_code, resp.text[:200])
                return None
            text = resp.json().get("text", "").strip()
            log.info("NIM STT → %d chars", len(text))
            return text or None
    except Exception as exc:
        log.error("NIM STT failed: %s", exc)
        return None


async def transcribe_audio(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe audio bytes to text.
    Primary: NVIDIA NIM Whisper Large v3 (if NIM_STT_API_KEY set).
    Fallback: local faster-whisper.
    """
    # ── NIM cloud path ────────────────────────────────────────────────────
    if cfg.NIM_STT_API_KEY:
        result = await _nim_transcribe(audio_bytes)
        if result:
            return result
        log.warning("NIM STT returned nothing, trying local fallback")

    # ── Local faster-whisper fallback ─────────────────────────────────────
    model = await _get_stt_model()
    if model is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(audio_bytes)
    try:
        segments, info = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.transcribe(
                tmp_path,
                language=cfg.STT_LANGUAGE,
                beam_size=5,
                vad_filter=True,
                word_timestamps=False,
            ),
        )
        text = " ".join(seg.text for seg in segments).strip()
        log.info("local STT: %.1fs audio → %d chars", info.duration, len(text))
        return text if text else None
    except Exception as exc:
        log.error("local STT transcription failed: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─── TTS: NIM Magpie TTS (cloud) with edge-tts / pyttsx3 fallback ────────────

_tts_lock = asyncio.Lock()


async def _nim_synthesize(text: str, voice: str = "female") -> Optional[bytes]:
    """Synthesize via NVIDIA NIM Magpie TTS Multilingual API. Returns WAV bytes."""
    import httpx
    nim_voice = cfg.NIM_TTS_VOICE  # e.g. "Magpie-Multilingual.EN-US.Aria"
    if "male" in voice and "female" not in voice:
        # Pick a male voice — Jason is available in EN-US
        nim_voice = "Magpie-Multilingual.EN-US.Jason"
    headers = {"Authorization": f"Bearer {cfg.NIM_TTS_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{cfg.NIM_SPEECH_BASE}/audio/synthesize",
                headers=headers,
                files={
                    "text":     (None, text),
                    "language": (None, "en-US"),
                    "voice":    (None, nim_voice),
                },
            )
            if resp.status_code != 200:
                log.warning("NIM TTS HTTP %d: %s", resp.status_code, resp.text[:200])
                return None
            audio = resp.content
            if audio:
                log.info("NIM TTS: synthesized %d bytes (WAV)", len(audio))
                return audio
    except Exception as exc:
        log.error("NIM TTS failed: %s", exc)
    return None


async def synthesize_speech(
    text: str,
    voice: str = "female",
    speed: float = 1.0,
) -> Optional[bytes]:
    """
    Synthesize text to audio bytes.
    Primary: NVIDIA NIM Magpie TTS Multilingual (WAV, if NIM_TTS_API_KEY set).
    Fallback 1: edge-tts (Microsoft neural, MP3, free).
    Fallback 2: pyttsx3 (local, WAV).
    """
    async with _tts_lock:
        # ── NIM Magpie TTS ────────────────────────────────────────────────
        if cfg.NIM_TTS_API_KEY:
            audio = await _nim_synthesize(text, voice)
            if audio:
                return audio
            log.warning("NIM TTS returned nothing, falling back to edge-tts")

        # ── edge-tts fallback (Microsoft Neural, MP3) ─────────────────────
        try:
            import edge_tts  # type: ignore
            import io
            voice_name = "en-US-AriaNeural" if "female" in voice else "en-US-GuyNeural"
            rate_pct = f"+{int((speed - 1.0) * 100)}%" if speed >= 1.0 else f"{int((speed - 1.0) * 100)}%"
            communicate = edge_tts.Communicate(text, voice=voice_name, rate=rate_pct)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            audio_bytes = buf.getvalue()
            if audio_bytes:
                log.info("edge-tts: synthesized %d bytes", len(audio_bytes))
                return audio_bytes
        except ImportError:
            pass
        except Exception as exc:
            log.warning("edge-tts TTS failed: %s", exc)

        # ── pyttsx3 last-resort fallback ──────────────────────────────────
        try:
            import pyttsx3  # type: ignore
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            def _synth():
                engine = pyttsx3.init()
                rate   = int(engine.getProperty("rate") * speed)
                engine.setProperty("rate", rate)
                voices = engine.getProperty("voices")
                if voices:
                    for v in voices:
                        if "female" in v.name.lower() and "female" in voice:
                            engine.setProperty("voice", v.id)
                            break
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()

            await asyncio.get_event_loop().run_in_executor(None, _synth)
            with open(tmp_path, "rb") as fh:
                audio_bytes = fh.read()
            os.unlink(tmp_path)
            return audio_bytes if audio_bytes else None
        except ImportError:
            pass
        except Exception as exc:
            log.warning("pyttsx3 TTS failed: %s", exc)

    log.info("No TTS engine available — returning text-only response")
    return None


def _clinical_tts_text(full_response: str) -> str:
    """
    Extract the most important clinical snippet for voice (≤200 words).
    Prefers SHIFT HANDOVER BRIEF, then CLINICAL ASSESSMENT, then first 200 words.
    """
    import re
    for section in ("SHIFT HANDOVER BRIEF", "CLINICAL ASSESSMENT"):
        m = re.search(rf"{section}[^:]*:\s*([\s\S]+?)(?:\n\n|\n[A-Z]|\Z)", full_response, re.I)
        if m:
            text = m.group(1).strip()
            words = text.split()
            return " ".join(words[:200])
    words = full_response.split()
    return " ".join(words[:200])


# ─── Session management ───────────────────────────────────────────────────────

@dataclass
class VoiceMessage:
    role:    str      # "user" | "assistant"
    content: str
    ts:      float = field(default_factory=time.time)


@dataclass
class VoiceSession:
    session_id: str
    patient_id: Optional[str]   = None
    messages:   List[VoiceMessage] = field(default_factory=list)
    context:    Dict[str, Any]  = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add(self, role: str, content: str) -> None:
        self.messages.append(VoiceMessage(role, content))

    def to_llm_history(self, max_turns: int = 6) -> List[Dict]:
        msgs = self.messages[-max_turns * 2:]
        return [{"role": m.role, "content": m.content} for m in msgs]


_sessions: Dict[str, VoiceSession] = {}
_sessions_lock = asyncio.Lock()


async def get_or_create_session(
    session_id: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> VoiceSession:
    async with _sessions_lock:
        if session_id and session_id in _sessions:
            return _sessions[session_id]
        sid = session_id or str(uuid.uuid4())
        s   = VoiceSession(session_id=sid, patient_id=patient_id)
        _sessions[sid] = s
        log_voice_session(sid, "start", patient_id)
        return s


async def close_session(session_id: str) -> None:
    async with _sessions_lock:
        if session_id in _sessions:
            s = _sessions.pop(session_id)
            log_voice_session(session_id, "end", s.patient_id,
                              {"messages": len(s.messages)})


# ─── Intent recognition + clinical query ────────────────────────────────────

_SYSTEM_PROMPT = """You are HC01, an AI clinical assistant for an ICU ward.
You have access to real-time patient data, lab trends, medication safety checks, and clinical guidelines.
Answer CONCISELY — your responses will be read aloud to a clinician.
Limit responses to 3-5 sentences unless asked for detail.
Always cite specific values (SOFA score, lab values, alert levels) when available.
If patient data is available in context, use it. If not, ask the clinician to specify a patient."""


async def voice_query(
    session: VoiceSession,
    user_text: str,
    patient_data_json: Optional[str] = None,
    nim_key: Optional[str] = None,
) -> str:
    """
    Process a voice query and return a clinical text response.
    Optionally includes patient data as context.
    """
    from .audit import log_ai_inference

    context_block = ""
    if patient_data_json:
        context_block = f"\n\n[PATIENT CONTEXT]\n{patient_data_json[:2000]}\n[END CONTEXT]\n"

    session.add("user", user_text)

    system = _SYSTEM_PROMPT + context_block
    messages = [{"role": "system", "content": system}] + session.to_llm_history()

    key = nim_key or cfg.nim_key("fallback") or cfg.nim_key("chief")
    response = "[HC01 voice model unavailable]"

    if key:
        try:
            response = await nim.chat(
                cfg.FALLBACK_MODEL,
                messages,
                key,
                max_tokens=400,
                temperature=0.3,
            )
            log_ai_inference(
                session.patient_id or "unknown",
                cfg.FALLBACK_MODEL,
                "voice_response",
            )
        except Exception as exc:
            log.error("Voice NIM query failed: %s", exc)
            # Try Ollama as fallback
            try:
                if await ollama.is_online():
                    response = await ollama.chat(cfg.NOTE_MODEL, messages, max_tokens=300)
            except Exception as exc2:
                log.error("Voice Ollama fallback failed: %s", exc2)
    else:
        try:
            if await ollama.is_online():
                response = await ollama.chat(cfg.NOTE_MODEL, messages, max_tokens=300)
        except Exception as exc:
            log.error("Voice Ollama query failed: %s", exc)

    session.add("assistant", response)
    log_voice_session(session.session_id, "response", session.patient_id,
                      {"chars": len(response)})
    return response


# ─── WebSocket voice handler (used by main.py) ───────────────────────────────

async def handle_voice_websocket(
    websocket,
    session_id: Optional[str] = None,
    nim_key: Optional[str] = None,
) -> None:
    """
    Full voice WebSocket handler.

    Client → Server messages:
      {"type": "audio",  "data": "<base64 WAV/WebM>", "patient_id": "..."}
      {"type": "text",   "content": "...",             "patient_id": "..."}
      {"type": "ping"}

    Server → Client messages:
      {"type": "session_id",   "session_id": "..."}
      {"type": "transcript",   "text": "..."}
      {"type": "thinking"}
      {"type": "response_text","text": "..."}
      {"type": "audio",        "data": "<base64 WAV>", "available": true/false}
      {"type": "error",        "message": "..."}
    """
    from fastapi import WebSocketDisconnect

    sid = session_id or str(uuid.uuid4())
    await websocket.send_json({"type": "session_id", "session_id": sid})
    log.info("Voice WS session %s started", sid)

    session: Optional[VoiceSession] = None

    try:
        while True:
            raw = await websocket.receive_json()
            msg_type   = raw.get("type", "")
            patient_id = raw.get("patient_id")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if session is None:
                session = await get_or_create_session(sid, patient_id)

            # Resolve patient context for the session
            patient_ctx_json: Optional[str] = None
            patient_data_raw = raw.get("patient_data")  # Client can send PatientData
            if patient_data_raw:
                patient_ctx_json = json.dumps(patient_data_raw) if isinstance(patient_data_raw, dict) else str(patient_data_raw)
                session.context["patient_data"] = patient_ctx_json

            # Use cached patient if not re-sent
            if not patient_ctx_json and session.context.get("patient_data"):
                patient_ctx_json = session.context["patient_data"]

            # ── Audio input ────────────────────────────────────────────
            if msg_type == "audio":
                audio_b64 = raw.get("data", "")
                if not audio_b64:
                    await websocket.send_json({"type": "error", "message": "No audio data"})
                    continue

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Invalid base64 audio"})
                    continue

                # STT
                transcript = await transcribe_audio(audio_bytes)
                if not transcript:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Could not transcribe audio. Please check microphone or use text input.",
                    })
                    continue

                await websocket.send_json({"type": "transcript", "text": transcript})
                log_voice_session(sid, "transcript", patient_id, {"chars": len(transcript)})

                # LLM query
                await websocket.send_json({"type": "thinking"})
                response_text = await voice_query(session, transcript, patient_ctx_json, nim_key)
                await websocket.send_json({"type": "response_text", "text": response_text})

                # TTS
                tts_text  = _clinical_tts_text(response_text)
                audio_out = await synthesize_speech(tts_text)
                if audio_out:
                    await websocket.send_json({
                        "type":      "audio",
                        "data":      base64.b64encode(audio_out).decode(),
                        "available": True,
                    })
                else:
                    await websocket.send_json({"type": "audio", "available": False,
                                               "message": "TTS unavailable — text response only"})

            # ── Text input ─────────────────────────────────────────────
            elif msg_type == "text":
                content = raw.get("content", "").strip()
                if not content:
                    await websocket.send_json({"type": "error", "message": "Empty text"})
                    continue

                await websocket.send_json({"type": "thinking"})
                response_text = await voice_query(session, content, patient_ctx_json, nim_key)
                await websocket.send_json({"type": "response_text", "text": response_text})

                tts_text  = _clinical_tts_text(response_text)
                audio_out = await synthesize_speech(tts_text)
                if audio_out:
                    await websocket.send_json({
                        "type":      "audio",
                        "data":      base64.b64encode(audio_out).decode(),
                        "available": True,
                    })
                else:
                    await websocket.send_json({"type": "audio", "available": False})

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type!r}",
                })

    except WebSocketDisconnect:
        log.info("Voice WS session %s disconnected", sid)
    except Exception as exc:
        log.exception("Voice WS error in session %s", sid)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if session:
            await close_session(sid)
