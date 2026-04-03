"""
Note Input Engine — OCR and Speech-to-Note parsing for HC01.

Two input modes:
  1. OCR:    image (JPEG/PNG base64) → pytesseract extracts text → NOTE_PARSER agent
  2. Speech: audio (WAV/WebM base64) → faster-whisper transcribes → NOTE_PARSER agent

Both return the same structured ParsedNote schema.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import io
from typing import Any, Dict, Optional

from .clients import nim, ollama
from .config import cfg
from .voice_workflow import transcribe_audio

log = logging.getLogger("HC01.note_input")


# ─── OCR via pytesseract ─────────────────────────────────────────────────────

async def ocr_image(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """
    Extract text from a clinical note image using pytesseract (Tesseract OCR).
    Runs in a thread pool to avoid blocking the event loop.
    Returns raw extracted text.
    """
    def _run_ocr(data: bytes) -> str:
        from PIL import Image
        import pytesseract

        img = Image.open(io.BytesIO(data))

        # Convert to RGB if needed (handles RGBA/palette modes)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Tesseract config: treat as single block of text, optimise for mixed content
        custom_config = r"--oem 3 --psm 6"
        text = pytesseract.image_to_string(img, config=custom_config, lang="eng")
        return text.strip()

    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(None, _run_ocr, image_bytes)
    except Exception as exc:
        # PIL.UnidentifiedImageError, TesseractNotFound, etc → 400-level errors
        err = str(exc)
        if "cannot identify image" in err or "UnidentifiedImage" in err:
            raise ValueError("Cannot decode image — send a valid JPEG or PNG.")
        if "tesseract" in err.lower() and "not found" in err.lower():
            raise RuntimeError("Tesseract OCR engine not installed on server.")
        raise RuntimeError(f"OCR failed: {err}")
    log.info("Tesseract OCR: extracted %d chars", len(text))
    if not text:
        raise ValueError("OCR produced no text — check image quality or send a clearer image.")
    return text


# ─── Note parser (shared logic for both OCR and speech paths) ────────────────

async def _parse_note_text(raw_text: str, nim_key: str = "") -> Dict[str, Any]:
    """
    Run structured NOTE_PARSER extraction on raw text.
    Returns the same JSON schema as agent_note_parser.
    """
    sys_prompt = (
        "You are a clinical NLP agent. Parse the clinical note and return ONLY a JSON object: "
        '{"symptoms":[],"medications_mentioned":[],"timeline_events":[{"time":"","event":""}],'
        '"vital_concerns":[],"active_problems":[],"infection_sources":[],"raw_text":""}. '
        "No preamble. No explanation."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"/no_think\n\nParse this clinical note:\n\n{raw_text[:3000]}"},
    ]

    # Try local qwen3:4b first, then NIM fallback
    result_text = ""
    try:
        if await ollama.is_online():
            result_text = await ollama.chat(cfg.NOTE_MODEL, messages, max_tokens=800)
    except Exception as exc:
        log.warning("Local note parse failed: %s", exc)

    if not result_text.strip() and nim_key:
        try:
            result_text = await nim.chat(cfg.FALLBACK_MODEL, messages, nim_key, max_tokens=800)
        except Exception as exc:
            log.warning("NIM note parse failed: %s", exc)

    # Extract JSON from response
    m = re.search(r'\{[\s\S]*\}', result_text)
    if m:
        try:
            parsed = json.loads(m.group())
            parsed["raw_text"] = raw_text
            return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text only
    return {
        "symptoms": [],
        "medications_mentioned": [],
        "timeline_events": [],
        "vital_concerns": [],
        "active_problems": [],
        "infection_sources": [],
        "raw_text": raw_text,
        "parse_warning": "JSON extraction failed — raw text preserved",
    }


# ─── Public API functions ─────────────────────────────────────────────────────

async def note_from_image(
    image_b64: str,
    mime: str = "image/jpeg",
    nim_key: str = "",
) -> Dict[str, Any]:
    """
    Full pipeline: base64 image → OCR → structured note parsing.

    Returns:
        {
          "input_type": "ocr",
          "ocr_text": "<raw extracted text>",
          "parsed": { symptoms, medications_mentioned, timeline_events, ... }
        }
    """
    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as exc:
        raise ValueError(f"Invalid base64 image: {exc}")

    ocr_text = await ocr_image(image_bytes, mime)
    parsed   = await _parse_note_text(ocr_text, nim_key)

    return {
        "input_type": "ocr",
        "ocr_text":   ocr_text,
        "parsed":     parsed,
    }


async def note_from_speech(
    audio_b64: str,
    nim_key: str = "",
) -> Dict[str, Any]:
    """
    Full pipeline: base64 audio → faster-whisper STT → structured note parsing.

    Returns:
        {
          "input_type": "speech",
          "transcript": "<transcribed text>",
          "parsed": { symptoms, medications_mentioned, timeline_events, ... }
        }
    """
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as exc:
        raise ValueError(f"Invalid base64 audio: {exc}")

    transcript = await transcribe_audio(audio_bytes)
    if not transcript:
        raise RuntimeError(
            "Speech transcription failed — faster-whisper unavailable or audio unrecognisable."
        )

    parsed = await _parse_note_text(transcript, nim_key)

    return {
        "input_type": "speech",
        "transcript": transcript,
        "parsed":     parsed,
    }
