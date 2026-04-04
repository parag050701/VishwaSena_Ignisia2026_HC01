"""
Voice Workflow Engine for HC01.
Pipeline: Audio → STT → Intent → Patient Lookup → Diagnosis → Response → TTS
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import wave
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .audit import log_voice_session
from .clients import nim, ollama
from .config import cfg

log = logging.getLogger("HC01.voice")

RIVA_CALL_TIMEOUT_SECONDS = 20


# ─── STT: Parakeet primary with cloud/local fallbacks ───────────────────────

_stt_model = None
_parakeet_asr = None
_stt_lock  = asyncio.Lock()
_parakeet_lock = asyncio.Lock()
_riva_services = None
_riva_lock = asyncio.Lock()
_coqui_tts = None
_coqui_lock = asyncio.Lock()


def _to_wav_bytes(audio_bytes: bytes) -> bytes:
    """Normalize audio to mono 16-bit 16 kHz WAV for STT."""
    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        segment = segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        buf = io.BytesIO()
        segment.export(buf, format="wav")
        return buf.getvalue()
    except Exception:
        return audio_bytes


def _looks_like_wav(audio_bytes: bytes) -> bool:
    return len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


async def _get_riva_services():
    """Build cached Riva ASR/TTS clients if the package and keys are available."""
    global _riva_services
    if _riva_services is not None:
        return _riva_services

    async with _riva_lock:
        if _riva_services is not None:
            return _riva_services

        try:
            import riva.client

            auth_stt = riva.client.Auth(
                use_ssl=True,
                uri=cfg.RIVA_SPEECH_BASE,
                metadata_args=(
                    ("function-id", cfg.RIVA_STT_FUNCTION_ID),
                    ("authorization", f"Bearer {cfg.NIM_STT_API_KEY}"),
                ),
                options=[
                    ("grpc.max_receive_message_length", 32 * 1024 * 1024),
                    ("grpc.max_send_message_length", 32 * 1024 * 1024),
                ],
            )
            auth_tts = riva.client.Auth(
                use_ssl=True,
                uri=cfg.RIVA_SPEECH_BASE,
                metadata_args=(
                    ("function-id", cfg.RIVA_TTS_FUNCTION_ID),
                    ("authorization", f"Bearer {cfg.NIM_TTS_API_KEY}"),
                ),
                options=[
                    ("grpc.max_receive_message_length", 32 * 1024 * 1024),
                    ("grpc.max_send_message_length", 32 * 1024 * 1024),
                ],
            )
            _riva_services = (
                riva.client.ASRService(auth_stt),
                riva.client.SpeechSynthesisService(auth_tts),
                riva.client.RecognitionConfig,
                riva.client.AudioEncoding,
            )
            log.info("Riva gRPC clients initialized")
        except Exception as exc:
            log.warning("Riva gRPC unavailable: %s", exc)
            _riva_services = None

    return _riva_services


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
            fallback_model = cfg.STT_FALLBACK_MODEL or "base"
            log.info("Loading faster-whisper %s on %s…", fallback_model, device)
            _stt_model = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: WhisperModel(fallback_model, device=device, compute_type=ctype),
            )
            log.info("faster-whisper loaded")
        except Exception as exc:
            log.warning("faster-whisper not available: %s", exc)
            _stt_model = None
    return _stt_model


async def _get_parakeet_asr():
    """Lazy-load NVIDIA NeMo Parakeet ASR model for local STT."""
    global _parakeet_asr
    if _parakeet_asr is not None:
        return _parakeet_asr

    async with _parakeet_lock:
        if _parakeet_asr is not None:
            return _parakeet_asr

        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["NVIDIA_VISIBLE_DEVICES"] = ""
            import torch
            torch.cuda.is_available = lambda: False  # type: ignore[assignment]
            torch.cuda.current_device = lambda: None  # type: ignore[assignment]

            model_id = cfg.STT_MODEL or "nvidia/parakeet-tdt-0.6b-v2"
            log.info("Loading Parakeet STT model via NeMo: %s", model_id)
            _parakeet_asr = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: __import__("nemo.collections.asr", fromlist=["models"]).models.ASRModel.from_pretrained(model_name=model_id),
            )
            log.info("Parakeet STT pipeline loaded")
        except Exception as exc:
            log.warning("Parakeet STT unavailable: %s", exc)
            _parakeet_asr = None

    return _parakeet_asr


async def _parakeet_transcribe(wav_bytes: bytes) -> Optional[str]:
    """Transcribe with local HF Parakeet model."""
    asr = await _get_parakeet_asr()
    if asr is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(wav_bytes)

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, lambda: asr.transcribe([tmp_path], timestamps=False))
        transcript = ""
        if isinstance(result, tuple) and result:
            transcripts = result[0]
            if transcripts:
                first = transcripts[0]
                transcript = getattr(first, "text", "") or str(first)
        elif isinstance(result, list) and result:
            first = result[0]
            transcript = getattr(first, "text", "") or str(first)
        elif isinstance(result, dict):
            transcript = str(result.get("text", "")).strip()

        transcript = transcript.strip()
        if transcript:
            log.info("Parakeet STT -> %d chars", len(transcript))
            return transcript
    except Exception as exc:
        log.warning("Parakeet STT transcription failed: %s", exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return None


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
    Primary: NVIDIA Riva gRPC (Whisper Large v3, confirmed working).
    Fallback: local faster-whisper.
    """
    wav_bytes = _to_wav_bytes(audio_bytes)

    # ── Riva gRPC PRIMARY (Whisper Large v3) ──────────────────────────────
    services = await _get_riva_services()
    if services:
        asr_service, _, recognition_config_cls, audio_encoding_cls = services
        try:
            config = recognition_config_cls(
                encoding=audio_encoding_cls.LINEAR_PCM,
                language_code="en-US",
                max_alternatives=1,
                enable_automatic_punctuation=True,
                sample_rate_hertz=16000,
                audio_channel_count=1,
            )
            response_future = asyncio.get_event_loop().run_in_executor(
                None,
                lambda: asr_service.offline_recognize(wav_bytes, config),
            )
            response = await asyncio.wait_for(response_future, timeout=15.0)
            pieces = [
                alternatives[0].transcript.strip()
                for result in getattr(response, "results", [])
                for alternatives in [getattr(result, "alternatives", [])]
                if alternatives and getattr(alternatives[0], "transcript", "").strip()
            ]
            transcript = " ".join(pieces).strip()
            if transcript:
                log.info("Riva STT → %d chars", len(transcript))
                return transcript
        except Exception as exc:
            log.warning("Riva STT failed: %s", exc)

    # ── faster-whisper fallback ───────────────────────────────────────────
    model = await _get_stt_model()
    if model is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(wav_bytes)
    try:
        segments, info = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.transcribe(tmp_path, language=cfg.STT_LANGUAGE,
                                     beam_size=5, vad_filter=True, word_timestamps=False),
        )
        text = " ".join(seg.text for seg in segments).strip()
        log.info("faster-whisper STT: %.1fs → %d chars", info.duration, len(text))
        return text if text else None
    except Exception as exc:
        log.error("faster-whisper STT failed: %s", exc)
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


async def _get_coqui_tts():
    """Lazy-load Coqui XTTS-v2 model files and runtime objects."""
    global _coqui_tts
    if _coqui_tts is not None:
        return _coqui_tts

    async with _coqui_lock:
        if _coqui_tts is not None:
            return _coqui_tts

        try:
            os.environ["COQUI_TOS_AGREED"] = "1"
            from huggingface_hub import snapshot_download
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts

            model_repo = "coqui/XTTS-v2"
            cache_dir = Path.home() / ".cache" / "hc01" / "xtts_v2"
            cache_dir.mkdir(parents=True, exist_ok=True)
            log.info("Downloading Coqui XTTS checkpoint files from %s", model_repo)
            model_dir = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=model_repo,
                    local_dir=str(cache_dir),
                    local_dir_use_symlinks=False,
                    allow_patterns=[
                        "config.json",
                        "model.pth",
                        "vocab.json",
                        "speakers_xtts.pth",
                        "dvae.pth",
                        "mel_stats.pth",
                        "hash.md5",
                    ],
                ),
            )

            config = XttsConfig()
            config.load_json(str(Path(model_dir) / "config.json"))
            model = Xtts.init_from_config(config)
            model.load_checkpoint(
                config,
                checkpoint_dir=str(model_dir),
                vocab_path=str(Path(model_dir) / "vocab.json"),
                speaker_file_path=str(Path(model_dir) / "speakers_xtts.pth"),
                eval=True,
                use_deepspeed=False,
            )

            device = cfg.TTS_DEVICE if cfg.TTS_DEVICE in {"cpu", "cuda"} else "cpu"
            if device == "cuda":
                model.cuda()
            else:
                model.cpu()

            _coqui_tts = {"model": model, "config": config, "model_dir": model_dir, "device": device}
            log.info("Coqui XTTS loaded")
        except Exception as exc:
            log.warning("Coqui TTS unavailable: %s", exc)
            _coqui_tts = None

    return _coqui_tts


async def _coqui_synthesize(text: str, voice: str = "female", speed: float = 1.0) -> Optional[bytes]:
    """Synthesize speech with Coqui XTTS-v2, returning WAV bytes."""
    tts = await _get_coqui_tts()
    if tts is None:
        return None

    model = tts["model"]
    speaker_wav = cfg.COQUI_SPEAKER_WAV or None
    language = cfg.COQUI_LANGUAGE or "en"

    coqui_speaker = None
    if not speaker_wav:
        # Use built-in XTTS speaker bank to avoid decoding external reference audio.
        coqui_speaker = "Ana Florence" if "female" in voice.lower() else "Damien Black"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    try:
        def _run_xtts():
            if speaker_wav:
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=[speaker_wav],
                    gpt_cond_len=6,
                    gpt_cond_chunk_len=6,
                    load_sr=22050,
                )
                out = model.inference(
                    text,
                    language,
                    gpt_cond_latent,
                    speaker_embedding,
                    temperature=0.7,
                    speed=speed,
                    enable_text_splitting=True,
                )
            else:
                out = model.synthesize(
                    text=text,
                    config=None,
                    speaker=coqui_speaker,
                    language=language,
                    speed=speed,
                )
            import numpy as np
            import soundfile as sf

            wav = np.asarray(out["wav"], dtype="float32")
            sf.write(out_path, wav, 24000)

        await asyncio.get_event_loop().run_in_executor(None, _run_xtts)
        with open(out_path, "rb") as fh:
            audio_bytes = fh.read()
        if audio_bytes:
            log.info("Coqui XTTS: synthesized %d bytes", len(audio_bytes))
            return audio_bytes
    except Exception as exc:
        log.warning("Coqui synthesis failed: %s", exc)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass

    return None


async def synthesize_speech(
    text: str,
    voice: str = "female",
    speed: float = 1.0,
) -> Optional[bytes]:
    """
    Synthesize text to audio bytes.
    Primary: edge-tts (Microsoft Neural, MP3, confirmed working).
    Fallback: pyttsx3 (local, WAV — last resort only).
    """
    async with _tts_lock:
        # ── edge-tts PRIMARY (Microsoft Neural voices, fast, free) ───────────
        try:
            import edge_tts  # type: ignore
            voice_name = "en-US-AriaNeural" if "female" in voice else "en-US-GuyNeural"
            rate_pct = f"+{int((speed - 1.0) * 100)}%" if speed >= 1.0 else f"{int((speed - 1.0) * 100)}%"
            communicate = edge_tts.Communicate(text, voice=voice_name, rate=rate_pct)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            audio_bytes = buf.getvalue()
            if audio_bytes:
                log.info("edge-tts: synthesized %d bytes (MP3)", len(audio_bytes))
                return audio_bytes
        except Exception as exc:
            log.warning("edge-tts failed: %s — trying pyttsx3", exc)

        # ── pyttsx3 last-resort ───────────────────────────────────────────────
        try:
            import pyttsx3  # type: ignore
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            def _synth():
                engine = pyttsx3.init()
                engine.setProperty("rate", int(engine.getProperty("rate") * speed))
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
