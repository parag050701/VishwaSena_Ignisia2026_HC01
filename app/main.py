"""HC01 — ICU Diagnostic Risk Assistant  |  FastAPI application entry point."""

import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import data
from .agents import master_orchestrate, preembed_guidelines
from .clients import ollama, nim
from .config import cfg
from .audit import get_recent_entries, log_patient_access
from .data_loader import (
    get_demo_patients,
    get_loader,
    get_synthetic_patient,
    list_synthetic_patients,
)
from .ehr import FHIRClient, get_fhir_client
from .fhir_local import get_local_store
from .llm_council import LLMCouncilError, ask_council, build_council_question
from .voice_workflow import handle_voice_websocket, transcribe_audio, synthesize_speech
from .models import AgentContext, PatientData, PatientSummary

logging.basicConfig(
    level=logging.DEBUG if cfg.DEBUG else logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("HC01")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("HC01 Backend starting…")
    # Warm up data loader cache
    try:
        loader = get_loader()
        patients = loader.list_patients()
        log.info("MIMIC data loaded: %d ICU stays available", len(patients))
    except Exception as exc:
        log.warning("MIMIC data loader warm-up failed: %s", exc)

    # Preload local FHIR bundle
    try:
        store = get_local_store()
        summaries = store.list_patient_summaries()
        log.info("Local FHIR bundle: %d patients preloaded", len(summaries))
    except Exception as exc:
        log.warning("Local FHIR preload failed: %s", exc)

    await preembed_guidelines()

    # Pre-warm faster-whisper STT model (avoids 30s delay on first voice call)
    try:
        from .voice_workflow import _get_stt_model
        stt = await _get_stt_model()
        if stt:
            log.info("STT model (faster-whisper) pre-warmed")
        else:
            log.warning("STT model not available — voice transcription disabled")
    except Exception as exc:
        log.warning("STT pre-warm failed: %s", exc)

    log.info("HC01 ready.")
    yield
    log.info("HC01 Backend shutting down.")


app = FastAPI(
    title="HC01 — ICU Diagnostic Risk Assistant",
    version="2.1.0",
    description="Multi-agent clinical decision support for ICU patients.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Serve frontend ──────────────────────────────────────────────────────────

_FRONTEND = Path(__file__).resolve().parent.parent / "hc01-icu-assistant.html"

@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(_FRONTEND)


# ─── Health & status ─────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health() -> Dict[str, Any]:
    online = await ollama.is_online()
    models = await ollama.available_models() if online else []
    nim_configured = cfg.has_nim_key("chief") or cfg.has_nim_key("fallback")
    return {
        "status": "ok",
        "ollama_online":           online,
        "ollama_models":           models,
        "nim_configured":          nim_configured,
        "nim_chief_key_set":       cfg.has_nim_key("chief"),
        "nim_fallback_key_set":    cfg.has_nim_key("fallback"),
        "llm_council_enabled":     cfg.ENABLE_LLM_COUNCIL,
        "llm_council_space":       cfg.LLM_COUNCIL_SPACE,
        "guideline_embeds_ready":  len(data._guideline_embeddings) == len(data.GUIDELINES),
        "guidelines_count":        len(data.GUIDELINES),
        "datasets": {
            "mimic_iv":          data.MIMIC_IV["citation"],
            "eicu":              data.EICU["citation"],
            "physionet_sepsis":  data.PHYSIONET_SEPSIS["citation"],
        },
    }


@app.get("/api/mimic-stats", tags=["System"])
async def mimic_stats() -> Dict[str, Any]:
    return {
        "mimic_iv":          data.MIMIC_IV,
        "eicu":              data.EICU,
        "physionet_sepsis":  data.PHYSIONET_SEPSIS,
    }


# ─── Patient endpoints ────────────────────────────────────────────────────────

@app.get("/api/patients", tags=["Patients"], response_model=List[PatientSummary])
async def list_patients() -> List[Dict]:
    """Return all ICU patients available in the MIMIC CSV dataset."""
    try:
        loader = get_loader()
        return loader.list_patients()
    except Exception as exc:
        log.exception("list_patients error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/patient/{subject_id}/{hadm_id}", tags=["Patients"])
async def get_patient(subject_id: int, hadm_id: int) -> Dict[str, Any]:
    """
    Build and return a full PatientData record from MIMIC CSV files.
    This is the data payload the frontend should send to /ws/diagnose.
    """
    try:
        loader = get_loader()
        patient = loader.build_patient_data(subject_id, hadm_id)
        if patient is None:
            raise HTTPException(status_code=404, detail=f"Patient {subject_id}/{hadm_id} not found")
        return patient.model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("get_patient error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/demo-patients", tags=["Patients"])
async def demo_patients() -> List[Dict[str, Any]]:
    """
    Return 3 pre-built demo ICU patients with rich multi-timepoint data.
    These showcase all pipeline agents: sepsis, ARDS, and MODS.
    """
    try:
        return [p.model_dump() for p in get_demo_patients()]
    except Exception as exc:
        log.exception("demo_patients error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/patients/{subject_id}", tags=["Patients"])
async def get_patient_stays(subject_id: int) -> List[Dict]:
    """List all ICU admissions for a given subject_id."""
    try:
        loader = get_loader()
        all_patients = loader.list_patients()
        stays = [p for p in all_patients if p["subject_id"] == subject_id]
        if not stays:
            raise HTTPException(status_code=404, detail=f"No stays found for subject {subject_id}")
        return stays
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("get_patient_stays error")
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Synchronous REST diagnose endpoint (no streaming) ───────────────────────

class DiagnoseHTTPRequest(BaseModel):
    patient: PatientData
    nim_api_key: Optional[str] = ""
    include_council: bool = False


class DiagnoseHTTPResponse(BaseModel):
    synthesis:    str
    handover:     str
    alert_level:  str
    sofa_total:   int
    news2_total:  int
    sofa:         Optional[Dict] = None
    news2:        Optional[Dict] = None
    disease_timeline: List[Dict] = []
    timeline_str: str = ""
    lab_findings: List[Dict]
    alerts:       List[Dict]
    med_conflicts: List[Dict]
    outliers:     List[Dict]
    trajectory:   Dict
    parsed_notes: Optional[Dict] = None
    family_communication: Optional[Dict] = None
    diagnosis_hold: bool = False
    hold_reasons: List[str] = []
    council_consult: Optional[str] = None
    council_status: Optional[str] = None
    safety_disclaimer: str = ""


async def _maybe_council_consult(
    include_council: bool,
    patient: PatientData,
    ctx: AgentContext,
) -> Dict[str, Optional[str]]:
    """Optionally request a second-opinion council review."""
    if not include_council:
        return {"council_consult": None, "council_status": None}
    if not cfg.ENABLE_LLM_COUNCIL:
        return {"council_consult": None, "council_status": "disabled"}

    question = build_council_question(
        patient_name=patient.name,
        admit_diag=patient.admitDiag,
        alert_level=ctx.alert_level,
        sofa_total=ctx.sofa.get("total", 0) if ctx.sofa else 0,
        news2_total=ctx.news2.get("total", 0) if ctx.news2 else 0,
        timeline_str=ctx.timeline_str,
        synthesis=ctx.synthesis,
    )
    try:
        return {
            "council_consult": await ask_council(question),
            "council_status": "ok",
        }
    except LLMCouncilError as exc:
        return {"council_consult": None, "council_status": f"error: {exc}"}


@app.post("/api/diagnose", tags=["Diagnosis"], response_model=DiagnoseHTTPResponse)
async def diagnose_http(req: DiagnoseHTTPRequest) -> Dict[str, Any]:
    """
    Run the full diagnostic pipeline and return the result as JSON (no streaming).
    Use /ws/diagnose for real-time streaming output.
    """
    nim_key = req.nim_api_key or cfg.nim_key("chief")
    ctx = AgentContext(req.patient, nim_key, ws=None)
    await master_orchestrate(ctx)
    council_data = await _maybe_council_consult(req.include_council, req.patient, ctx)

    return {
        "synthesis":        ctx.synthesis,
        "handover":         ctx.handover,
        "alert_level":      ctx.alert_level,
        "sofa_total":       ctx.sofa.get("total", 0) if ctx.sofa else 0,
        "news2_total":      ctx.news2.get("total", 0) if ctx.news2 else 0,
        "sofa":             ctx.sofa,
        "news2":            ctx.news2,
        "disease_timeline": ctx.disease_timeline,
        "timeline_str":     ctx.timeline_str,
        "lab_findings":     ctx.lab_findings,
        "alerts":           ctx.alert_events,
        "med_conflicts":    ctx.med_conflicts,
        "outliers":         ctx.outliers,
        "trajectory":       ctx.trajectory,
        "parsed_notes":     ctx.parsed_notes,
        "family_communication": ctx.family_communication,
        "diagnosis_hold":   ctx.diagnosis_hold,
        "hold_reasons":     ctx.hold_reasons,
        "council_consult":  council_data["council_consult"],
        "council_status":   council_data["council_status"],
        "safety_disclaimer": (
            "DECISION-SUPPORT ONLY. All outputs must be verified by a qualified clinician. "
            "This system does NOT provide clinical diagnoses. Held diagnoses require confirmed laboratory redraws."
        ),
    }


# ─── WebSocket streaming diagnose ─────────────────────────────────────────────

@app.websocket("/ws/diagnose")
async def ws_diagnose(websocket: WebSocket) -> None:
    await websocket.accept()
    log.info("WebSocket client connected")
    try:
        raw = await websocket.receive_json()
        if raw.get("action") != "diagnose":
            await websocket.send_json({"type": "error", "message": "Expected action=diagnose"})
            return

        patient = PatientData(**raw["patient"])

        # Resolve NIM key: from request > .env chief > .env fallback
        nim_key = (
            raw.get("nim_api_key", "").strip() or
            cfg.nim_key("chief")
        )

        ctx = AgentContext(patient, nim_key, websocket)
        await master_orchestrate(ctx)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as exc:
        log.exception("WebSocket orchestration error")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# ─── Synthetic dataset endpoints ──────────────────────────────────────────────

@app.get("/api/synthetic-patients", tags=["Synthetic Dataset"])
async def list_synthetic(
    case_type:  Optional[str] = None,
    trajectory: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List 120 synthetic ICU cases.
    Filter: ?case_type=sepsis|ards|aki|cardiac|neuro|stable
            &trajectory=worsening|improving|stable
    """
    return list_synthetic_patients(case_type=case_type, trajectory=trajectory)


@app.get("/api/synthetic-patients/{patient_id}", tags=["Synthetic Dataset"])
async def get_synthetic(patient_id: str) -> Dict[str, Any]:
    """Return full PatientData for a synthetic patient (e.g. P001)."""
    pt = get_synthetic_patient(patient_id)
    if pt is None:
        raise HTTPException(status_code=404, detail=f"Synthetic patient {patient_id!r} not found")
    log_patient_access(patient_id, "read", "synthetic")
    return pt.model_dump()


# ─── FHIR / EHR endpoints ────────────────────────────────────────────────────

class FHIRSearchRequest(BaseModel):
    fhir_id:      Optional[str] = None
    patient_name: Optional[str] = None
    mrn:          Optional[str] = None
    server_url:   Optional[str] = None   # override .env server
    client_id:    Optional[str] = None
    client_secret: Optional[str] = None
    oauth_url:    Optional[str] = None


@app.post("/api/ehr/patient", tags=["EHR / FHIR"])
async def ehr_search_patient(req: FHIRSearchRequest) -> Dict[str, Any]:
    """
    Search a FHIR server for a patient and return their data as PatientData.
    Supports OAuth2 client-credentials (SMART on FHIR) or anonymous access.

    If no server_url is provided, uses FHIR_SERVER_URL from .env,
    falling back to the public HAPI FHIR sandbox.
    """
    client = FHIRClient(
        server_url=req.server_url or cfg.FHIR_SERVER_URL,
        client_id=req.client_id or cfg.FHIR_CLIENT_ID,
        client_secret=req.client_secret or cfg.FHIR_CLIENT_SECRET,
        oauth_url=req.oauth_url or cfg.FHIR_OAUTH_URL,
    )

    if not (req.fhir_id or req.patient_name or req.mrn):
        raise HTTPException(status_code=400, detail="Provide fhir_id, patient_name, or mrn")

    fhir_resource = await client.get_patient_resource(
        fhir_id=req.fhir_id, name=req.patient_name, mrn=req.mrn
    )
    if not fhir_resource:
        raise HTTPException(status_code=404, detail="Patient not found in FHIR server")

    fhir_id  = fhir_resource.get("id", req.fhir_id or "unknown")
    pat_data = await client.to_patient_data(fhir_id)
    if not pat_data:
        raise HTTPException(status_code=502, detail="Could not convert FHIR resources to PatientData")

    log_patient_access(fhir_id, "ehr_pull", "fhir")
    return {
        "source":     "fhir",
        "fhir_id":    fhir_id,
        "server":     client.server_url,
        "patient":    pat_data.model_dump(),
    }


@app.get("/api/ehr/patient/{fhir_id}", tags=["EHR / FHIR"])
async def ehr_get_patient(fhir_id: str) -> Dict[str, Any]:
    """
    Pull a patient by their FHIR ID from the configured (or public HAPI) server.
    Returns full PatientData ready for the diagnostic pipeline.
    """
    client   = get_fhir_client()
    pat_data = await client.to_patient_data(fhir_id)
    if not pat_data:
        raise HTTPException(status_code=404, detail=f"FHIR patient {fhir_id!r} not found")
    log_patient_access(fhir_id, "ehr_pull", "fhir")
    return {
        "source":  "fhir",
        "fhir_id": fhir_id,
        "server":  client.server_url,
        "patient": pat_data.model_dump(),
    }


@app.get("/api/ehr/capability", tags=["EHR / FHIR"])
async def ehr_capability() -> Dict[str, Any]:
    """Check FHIR server capability statement."""
    client = get_fhir_client()
    cap    = await client.capability_statement()
    return {
        "server":  client.server_url,
        "is_public": client.is_public,
        "status":  "ok" if cap else "unreachable",
        "fhir_version": (cap or {}).get("fhirVersion", "unknown"),
        "software": (cap or {}).get("software", {}),
    }


# ─── Voice endpoints ──────────────────────────────────────────────────────────

class TranscribeRequest(BaseModel):
    audio_b64: str          # base64-encoded WAV/WebM audio
    language:  str = "en"


class SynthesizeRequest(BaseModel):
    text:   str
    voice:  str   = "female"   # female | male
    speed:  float = 1.0


@app.post("/api/voice/transcribe", tags=["Voice"])
async def voice_transcribe(req: TranscribeRequest) -> Dict[str, Any]:
    """
    Transcribe base64-encoded audio to text using faster-whisper.
    Returns the transcript or an error if STT is unavailable.
    """
    try:
        audio_bytes = base64.b64decode(req.audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    from .voice_workflow import _get_stt_model
    model = await _get_stt_model()
    if model is None:
        return {"available": False, "text": None,
                "message": "STT engine not available on server."}
    transcript = await transcribe_audio(audio_bytes)
    if not transcript:
        return {"available": True, "text": None,
                "message": "No speech detected — speak clearly and try again."}
    return {"available": True, "text": transcript}


@app.post("/api/voice/synthesize", tags=["Voice"])
async def voice_synthesize(req: SynthesizeRequest) -> Dict[str, Any]:
    """
    Synthesize text to speech.
    Returns base64-encoded WAV audio, or {available: false} if no TTS engine.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    audio = await synthesize_speech(req.text, voice=req.voice, speed=req.speed)
    if audio is None:
        return {"available": False, "audio_b64": None,
                "message": "No TTS engine available. Install kokoro or pyttsx3."}
    return {"available": True, "audio_b64": base64.b64encode(audio).decode()}


# ─── Note input: OCR + Speech ─────────────────────────────────────────────────

class NoteOCRRequest(BaseModel):
    image_b64: str          # base64-encoded JPEG/PNG of the clinical note
    mime: str = "image/jpeg"
    nim_api_key: str = ""


class NoteSpeechRequest(BaseModel):
    audio_b64: str          # base64-encoded WAV/WebM/MP3 dictation
    nim_api_key: str = ""


@app.post("/api/notes/ocr", tags=["Notes"])
async def notes_ocr(req: NoteOCRRequest) -> Dict[str, Any]:
    """
    OCR pipeline: base64 image of a clinical note → qwen3-vl vision model extracts text
    → NOTE_PARSER agent returns structured JSON (symptoms, timeline, medications, problems).

    Use this to parse handwritten or printed clinical notes by photo.
    Requires qwen3-vl:4b (or qwen3-vl:2b) to be pulled in Ollama.
    """
    from .note_input import note_from_image
    try:
        nim_key = req.nim_api_key or cfg.nim_key("fallback")
        result  = await note_from_image(req.image_b64, req.mime, nim_key)
        return {"success": True, **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/notes/speech", tags=["Notes"])
async def notes_speech(req: NoteSpeechRequest) -> Dict[str, Any]:
    """
    Speech-to-note pipeline: base64 audio (WAV/WebM) → faster-whisper transcribes
    → NOTE_PARSER agent returns structured JSON (symptoms, timeline, medications, problems).

    Use this to dictate clinical observations verbally.
    Requires faster-whisper to be installed (already available in hc01 env).
    """
    from .note_input import note_from_speech
    try:
        nim_key = req.nim_api_key or cfg.nim_key("fallback")
        result  = await note_from_speech(req.audio_b64, nim_key)
        return {"success": True, **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.websocket("/ws/voice/{session_id}")
async def ws_voice(websocket: WebSocket, session_id: str) -> None:
    """
    Real-time voice conversation WebSocket.
    Accepts audio or text input, returns transcript + text response + TTS audio.

    Message protocol:
      → {"type":"audio", "data":"<base64>", "patient_data": {...}}
      → {"type":"text",  "content":"...",   "patient_data": {...}}
      ← {"type":"transcript",   "text":"..."}
      ← {"type":"thinking"}
      ← {"type":"response_text","text":"..."}
      ← {"type":"audio",        "data":"<base64>", "available": true}
    """
    await websocket.accept()
    nim_key = cfg.nim_key("fallback")
    await handle_voice_websocket(websocket, session_id=session_id, nim_key=nim_key)


@app.websocket("/ws/voice")
async def ws_voice_anon(websocket: WebSocket) -> None:
    """Same as /ws/voice/{session_id} but auto-generates a session ID."""
    await websocket.accept()
    nim_key = cfg.nim_key("fallback")
    await handle_voice_websocket(websocket, nim_key=nim_key)


# ─── Local FHIR bundle endpoints (offline / demo mode) ───────────────────────

@app.get("/api/fhir-local/patients", tags=["EHR / FHIR"])
async def fhir_local_list() -> List[Dict[str, Any]]:
    """List all patients from the local FHIR bundle (hc01_synthetic_fhir_bundle.json)."""
    return get_local_store().list_patient_summaries()


@app.get("/api/fhir-local/patient/{fhir_id}", tags=["EHR / FHIR"])
async def fhir_local_patient(fhir_id: str) -> Dict[str, Any]:
    """
    Return PatientData built from the local FHIR bundle.
    fhir_id format: hc01-patient-001 … hc01-patient-120
    """
    store = get_local_store()
    pt    = store.to_patient_data(fhir_id)
    if pt is None:
        raise HTTPException(status_code=404, detail=f"Local FHIR patient {fhir_id!r} not found")
    log_patient_access(fhir_id, "ehr_pull", "fhir_local")
    return {"source": "fhir_local", "fhir_id": fhir_id, "patient": pt.model_dump()}


@app.post("/api/fhir-local/diagnose/{fhir_id}", tags=["EHR / FHIR"])
async def fhir_local_diagnose(
    fhir_id: str,
    nim_api_key: str = "",
    include_council: bool = False,
) -> Dict[str, Any]:
    """
    One-shot: load patient from local FHIR bundle → run full diagnostic pipeline.
    This is the key demo endpoint.
    """
    store = get_local_store()
    pt    = store.to_patient_data(fhir_id)
    if pt is None:
        raise HTTPException(status_code=404, detail=f"Patient {fhir_id!r} not found in local bundle")

    nim_key = nim_api_key or cfg.nim_key("chief")
    ctx     = AgentContext(pt, nim_key, ws=None)
    await master_orchestrate(ctx)
    log_patient_access(fhir_id, "diagnose", "fhir_local")
    council_data = await _maybe_council_consult(include_council, pt, ctx)

    return {
        "source":            "fhir_local",
        "fhir_id":           fhir_id,
        "patient_name":      pt.name,
        "alert_level":       ctx.alert_level,
        "sofa":              ctx.sofa,
        "news2":             ctx.news2,
        # Disease progression timeline from TEMPORAL_LAB_MAPPER agent
        "disease_timeline":  ctx.disease_timeline,
        "timeline_str":      ctx.timeline_str,
        # Agent outputs
        "lab_findings":      ctx.lab_findings,
        "alerts":            ctx.alert_events,
        "med_conflicts":     ctx.med_conflicts,
        "outliers":          ctx.outliers,
        "trajectory":        ctx.trajectory,
        "retrieved_guidelines": [
            {"source": g.get("source", ""), "section": g["section"],
             "score": g.get("score", 0), "text": g["text"]}
            for g in (ctx.retrieved_guidelines or [])
        ],
        # Chief synthesis output
        "synthesis":         ctx.synthesis,
        "handover":          ctx.handover,
        "council_consult":   council_data["council_consult"],
        "council_status":    council_data["council_status"],
        "family_communication": ctx.family_communication,
        "diagnosis_hold":    ctx.diagnosis_hold,
        "hold_reasons":      ctx.hold_reasons,
        "safety_disclaimer": (
            "DECISION-SUPPORT ONLY. All outputs must be verified by a qualified clinician. "
            "This system does NOT provide clinical diagnoses. Held diagnoses require confirmed laboratory redraws."
        ),
    }


# ─── Priority queue (SOFA-ranked) ────────────────────────────────────────────

@app.get("/api/priority-queue", tags=["Clinical"])
async def priority_queue() -> List[Dict[str, Any]]:
    """
    Return all patients ranked by SOFA score (highest = most critical).
    Each entry: fhir_id, name, sofa, news2, alert_level, admit_diag, mortality_pct.
    """
    from .scoring import calc_sofa, calc_news2
    from .data import MIMIC_IV

    store = get_local_store()
    summaries = store.list_patient_summaries()
    ranked = []
    for s in summaries:
        fhir_id = s["fhir_id"]
        pt = store.to_patient_data(fhir_id)
        if pt is None:
            continue
        sofa  = calc_sofa(pt)
        news2 = calc_news2(pt)
        mortality = MIMIC_IV.get("sofa_mortality", {}).get(str(min(sofa["total"], 15)), "?")
        alert = (
            "CRITICAL" if sofa["total"] >= 10 or news2["level"] == "HIGH" else
            "WARNING"  if sofa["total"] >= 6  or news2["level"] == "MEDIUM" else
            "STABLE"
        )
        ranked.append({
            "fhir_id":      fhir_id,
            "name":         pt.name,
            "age":          pt.age,
            "sex":          pt.sex,
            "admit_diag":   pt.admitDiag,
            "sofa":         sofa["total"],
            "news2":        news2["total"],
            "news2_level":  news2["level"],
            "alert_level":  alert,
            "mortality_pct": mortality,
            "days_icu":     pt.daysInICU,
        })
    ranked.sort(key=lambda x: (x["sofa"], x["news2"]), reverse=True)
    return ranked


# ─── Natural-language assistant query ────────────────────────────────────────

class AssistantQueryRequest(BaseModel):
    query: str
    context_patient: Optional[str] = None   # fhir_id of currently viewed patient
    current_diagnostics: Optional[Dict] = None  # live diagnostic results from frontend

class AssistantQueryResponse(BaseModel):
    intent:      str
    answer:      str
    action:      Optional[str] = None        # "load_patient" | "show_priority" | "run_diagnostics"
    action_data: Optional[Dict] = None
    tts_text:    str = ""

@app.post("/api/assistant/query", tags=["Assistant"])
async def assistant_query(req: AssistantQueryRequest) -> AssistantQueryResponse:
    """
    Natural-language ward intelligence query.
    Covers all 120 patients with disease-specific filtering + live diagnostic context.
    """
    import re as _re
    import json as _json
    from .scoring import calc_sofa, calc_news2

    store   = get_local_store()
    pts     = store.list_patient_summaries()
    nim_key = cfg.nim_key("fallback")
    log.info("ASST: query=%r nim_key_set=%s has_diag=%s", req.query[:40], bool(nim_key), bool(req.current_diagnostics))

    # ── Build full ward index (all patients) ───────────────────────────────
    all_pts = []
    for s in pts:
        pt = store.to_patient_data(s["fhir_id"])
        if pt is None:
            continue
        sofa  = calc_sofa(pt)
        news2 = calc_news2(pt)
        alert = ("CRITICAL" if sofa["total"] >= 10 or news2["level"] == "HIGH"
                 else "WARNING" if sofa["total"] >= 6 else "STABLE")
        all_pts.append({
            "fhir_id": s["fhir_id"],
            "name":    pt.name,
            "age":     pt.age,
            "diag":    pt.admitDiag,
            "sofa":    sofa["total"],
            "news2":   news2["total"],
            "alert":   alert,
            "bp":      f"{pt.vitals.bpSys}/{pt.vitals.bpDia}",
            "hr":      pt.vitals.hr,
            "spo2":    pt.vitals.spo2,
            "lactate": next((v["v"] for v in reversed(pt.labs.get("lactate",[]) or []) if isinstance(v, dict)), None),
        })
    all_pts.sort(key=lambda x: x["sofa"], reverse=True)

    # ── Ward-level statistics ───────────────────────────────────────────────
    n_critical = sum(1 for p in all_pts if p["alert"] == "CRITICAL")
    n_warning  = sum(1 for p in all_pts if p["alert"] == "WARNING")

    # ── Disease-specific filtering based on query keywords ─────────────────
    q_low = req.query.lower()
    disease_keywords = {
        "sepsis":       ["sepsis", "septic"],
        "cardiac":      ["cardiac", "heart", "mi ", "infarct", "arrest", "cardio"],
        "respiratory":  ["respiratory", "pneumonia", "ards", "breathing", "ventilat", "pulmonary"],
        "renal":        ["renal", "kidney", "aki", "dialysis"],
        "trauma":       ["trauma", "injury", "fracture", "hemorrhage", "bleed"],
        "neurological": ["neuro", "stroke", "seizure", "brain", "encephalop"],
        "liver":        ["liver", "hepat", "cirrhosis"],
        "diabetic":     ["diabet", "dka", "glucose"],
    }
    matched_disease = None
    for disease, kws in disease_keywords.items():
        if any(kw in q_low for kw in kws):
            matched_disease = disease
            break

    # Build context sections
    top10_str = "\n".join(
        f"  {i+1}. {r['name']} ({r['fhir_id']}) — SOFA {r['sofa']} / NEWS2 {r['news2']}"
        f" — {r['alert']} — {r['diag']} — BP {r['bp']} HR {r['hr']} SpO2 {r['spo2']}%"
        + (f" Lactate {r['lactate']} mmol/L" if r['lactate'] else "")
        for i, r in enumerate(all_pts[:10])
    )

    disease_str = ""
    if matched_disease:
        kws = disease_keywords[matched_disease]
        matched_pts = [p for p in all_pts if any(kw in p["diag"].lower() for kw in kws)]
        if matched_pts:
            disease_str = f"\n\nPatients with {matched_disease} ({len(matched_pts)} found, SOFA-ranked):\n" + "\n".join(
                f"  - {p['name']} ({p['fhir_id']}) SOFA {p['sofa']} {p['alert']} — {p['diag']}"
                for p in matched_pts[:15]
            )

    stats_str = (f"\nWard summary: {len(all_pts)} patients total — "
                 f"{n_critical} CRITICAL, {n_warning} WARNING, "
                 f"{len(all_pts)-n_critical-n_warning} STABLE")

    # ── Current-patient diagnostic context ─────────────────────────────────
    diag_block = ""
    if req.current_diagnostics and req.context_patient:
        d = req.current_diagnostics
        parts = []
        # sofa can be a number or dict from frontend
        sofa_val = d.get("sofa")
        if sofa_val is not None:
            if isinstance(sofa_val, dict):
                parts.append(f"SOFA {sofa_val.get('total','?')} "
                             f"(Resp {sofa_val.get('resp','?')}, Coag {sofa_val.get('coag','?')}, "
                             f"Liver {sofa_val.get('liver','?')}, CV {sofa_val.get('cv','?')}, "
                             f"CNS {sofa_val.get('cns','?')}, Renal {sofa_val.get('renal','?')})")
            else:
                parts.append(f"SOFA {sofa_val}")
        news2_val = d.get("news2")
        if news2_val is not None:
            if isinstance(news2_val, dict):
                parts.append(f"NEWS2 {news2_val.get('total','?')} ({news2_val.get('level','?')})")
            else:
                parts.append(f"NEWS2 {news2_val}")
        if d.get("alertLevel"):
            parts.append(f"Alert level: {d['alertLevel']}")
        if d.get("handover"):
            parts.append(f"Clinical summary: {str(d['handover'])[:400]}")
        if isinstance(d.get("outliers"), list) and d["outliers"]:
            strs = [f"{o.get('name','?')} {o.get('value','?')} ({o.get('severity','?')})"
                    for o in d["outliers"][:5] if isinstance(o, dict)]
            if strs: parts.append(f"Lab outliers: {', '.join(strs)}")
        if isinstance(d.get("medConflicts"), list) and d["medConflicts"]:
            strs = [f"{m.get('drug','?')}: {m.get('issue',m.get('conflict','?'))}"
                    for m in d["medConflicts"][:3] if isinstance(m, dict)]
            if strs: parts.append(f"Med conflicts: {'; '.join(strs)}")
        if isinstance(d.get("alerts"), list) and d["alerts"]:
            strs = [str(a.get("message","")) for a in d["alerts"][:3] if isinstance(a, dict)]
            if strs: parts.append(f"Clinical alerts: {'; '.join(strs)}")
        if parts:
            diag_block = f"\n\nLive diagnostics for current patient ({req.context_patient}):\n" + "\n".join(f"  - {p}" for p in parts)

    # ── LLM call ────────────────────────────────────────────────────────────
    messages = [
        {"role": "system", "content": (
            "You are HC01 Ward Assistant — a clinical AI for an ICU. "
            "Answer using the ward data. Be specific: cite patient names, SOFA scores, diagnoses. "
            "For the current patient, use the live diagnostics block.\n"
            "Respond ONLY with this JSON structure, no markdown, no extra text:\n"
            '{"intent":"answer_question","answer":"<your answer here, max 2 sentences>","action":null,"action_data":{"fhir_id":null},"tts_text":"<spoken version, 1 sentence>"}\n'
            "Rules: intent must be exactly one of: answer_question, load_patient, show_priority, run_diagnostics. "
            "action must be exactly one of: load_patient, show_priority, run_diagnostics, or null. "
            "Keep answer under 100 words."
        )},
        {"role": "user", "content": (
            f"Top 10 critical patients (SOFA-ranked):\n{top10_str}\n"
            f"{stats_str}"
            f"{disease_str}"
            f"{diag_block}\n\n"
            f"Currently viewing: {req.context_patient or 'none'}\n"
            f"Query: {req.query}"
        )},
    ]

    raw = ""
    # Try fast model (Llama 3 8B) first; fall back to Qwen 2.5 7B
    fast_key = cfg.NIM_API_KEY_FAST
    if fast_key:
        try:
            raw = await nim.chat(cfg.FAST_MODEL, messages, fast_key, max_tokens=800, temperature=0.1)
            log.info("ASST: fast model → %d chars", len(raw))
        except Exception as e:
            log.warning("ASST: fast model failed (%s), trying fallback", e)
    if not raw and nim_key:
        try:
            raw = await nim.chat(cfg.FALLBACK_MODEL, messages, nim_key, max_tokens=800, temperature=0.1)
            log.info("ASST: fallback model → %d chars", len(raw))
        except Exception as e:
            log.warning("ASST: fallback model failed: %s", e)

    # ── Strip any <think>...</think> blocks before parsing ──────────────────
    raw_clean = _re.sub(r'<think>[\s\S]*?</think>', '', raw, flags=_re.IGNORECASE).strip()

    # ── Parse JSON — robust, handles truncated responses ───────────────────
    # Try full match first; if truncated, recover key fields with regex
    obj = None
    m = _re.search(r'\{[\s\S]*\}', raw_clean)
    if m:
        try:
            obj = _json.loads(m.group())
        except _json.JSONDecodeError:
            pass

    if obj is None:
        # Partial JSON recovery: extract fields individually
        try:
            def _extract(key: str) -> Optional[str]:
                pat = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)'
                hit = _re.search(pat, raw_clean)
                return hit.group(1) if hit else None

            answer = _extract("answer")
            tts    = _extract("tts_text") or answer
            intent_m = _re.search(r'"intent"\s*:\s*"(\w+)"', raw_clean)
            action_m = _re.search(r'"action"\s*:\s*"(\w+)"', raw_clean)
            fhir_m   = _re.search(r'"fhir_id"\s*:\s*"(hc01-patient-\d+)"', raw_clean)
            if answer:
                obj = {
                    "intent": intent_m.group(1) if intent_m else "answer_question",
                    "answer": answer,
                    "action": action_m.group(1) if action_m and action_m.group(1) != "null" else None,
                    "action_data": {"fhir_id": fhir_m.group(1)} if fhir_m else None,
                    "tts_text": tts or "",
                }
                log.info("ASST: partial JSON recovery succeeded, answer len=%d", len(answer))
        except Exception as recovery_err:
            log.warning("ASST: partial JSON recovery failed: %s", recovery_err)

    if obj:
        try:
            return AssistantQueryResponse(
                intent      = str(obj.get("intent") or "answer_question"),
                answer      = str(obj.get("answer") or ""),
                action      = obj.get("action") if obj.get("action") not in (None, "null", "") else None,
                action_data = obj.get("action_data") if isinstance(obj.get("action_data"), dict) else None,
                tts_text    = str(obj.get("tts_text") or obj.get("answer") or ""),
            )
        except Exception as parse_err:
            log.warning("ASST: response build failed: %s", parse_err)

    # ── Rule-based fallbacks (skip if we have live diagnostic context — LLM handles that) ──
    has_diag_context = bool(req.current_diagnostics and diag_block)
    if not has_diag_context and any(w in q_low for w in ["critical", "worst", "most critical", "highest sofa"]):
        top = all_pts[0] if all_pts else {}
        ans = (f"Most critical: {top.get('name')} (SOFA {top.get('sofa')}, {top.get('alert')}) "
               f"— {top.get('diag')}.")
        return AssistantQueryResponse(intent="answer_question", answer=ans, tts_text=ans,
                                      action="load_patient", action_data={"fhir_id": top.get("fhir_id")})

    if any(w in q_low for w in ["priority", "list", "all patients", "ward"]):
        ans = (f"Ward has {len(all_pts)} patients: {n_critical} CRITICAL, {n_warning} WARNING. "
               f"Top: {', '.join(p['name'] + ' SOFA ' + str(p['sofa']) for p in all_pts[:3])}.")
        return AssistantQueryResponse(intent="show_priority", answer=ans, tts_text=ans, action="show_priority")

    if matched_disease:
        kws = disease_keywords[matched_disease]
        matched_pts = [p for p in all_pts if any(kw in p["diag"].lower() for kw in kws)]
        if matched_pts:
            ans = (f"{len(matched_pts)} {matched_disease} patient(s): "
                   + ", ".join(f"{p['name']} SOFA {p['sofa']}" for p in matched_pts[:5])
                   + ("…" if len(matched_pts) > 5 else "."))
            top_match = matched_pts[0]
            return AssistantQueryResponse(intent="answer_question", answer=ans, tts_text=ans,
                                          action="load_patient", action_data={"fhir_id": top_match["fhir_id"]})

    ans = (f"Ward: {len(all_pts)} patients, {n_critical} critical. "
           f"Top priority: {all_pts[0]['name']} SOFA {all_pts[0]['sofa']}." if all_pts else "No patients loaded.")
    return AssistantQueryResponse(intent="answer_question", answer=ans, tts_text=ans)


# ─── Audit log endpoint ───────────────────────────────────────────────────────

@app.get("/api/audit", tags=["Compliance"])
async def audit_log(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent audit log entries (HIPAA compliance trail)."""
    return get_recent_entries(min(limit, 200))


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
