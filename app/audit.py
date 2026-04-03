"""
HIPAA-aligned audit logging for HC01.
Append-only, structured JSON records for every patient data access and clinical action.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("HC01.audit")

_AUDIT_PATH = Path(os.getenv("HC01_AUDIT_LOG", "./audit.log"))


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _write(record: Dict[str, Any]) -> None:
    """Append one JSON record to the audit log (one record per line)."""
    try:
        with _AUDIT_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        # Never let audit failure break clinical workflow
        log.error("Audit write failed: %s", exc)


# ─── Public API ──────────────────────────────────────────────────────────────

def log_patient_access(
    patient_id: str,
    action: str,                    # "read" | "diagnose" | "voice_query" | "ehr_pull"
    source: str,                    # "mimic" | "synthetic" | "fhir" | "demo"
    user_id: str = "anonymous",
    ip_address: str = "unknown",
    details: Optional[Dict] = None,
) -> None:
    """Record a patient data access event."""
    _write({
        "event":       "PATIENT_ACCESS",
        "timestamp":   _now_iso(),
        "patient_id":  patient_id,
        "action":      action,
        "data_source": source,
        "user_id":     user_id,
        "ip_address":  ip_address,
        "details":     details or {},
    })


def log_ehr_query(
    fhir_server: str,
    resource_type: str,
    query_params: Dict,
    result_count: int,
    user_id: str = "anonymous",
) -> None:
    """Record a FHIR EHR query for compliance."""
    _write({
        "event":         "EHR_QUERY",
        "timestamp":     _now_iso(),
        "fhir_server":   fhir_server,
        "resource_type": resource_type,
        "query_params":  query_params,
        "result_count":  result_count,
        "user_id":       user_id,
    })


def log_ai_inference(
    patient_id: str,
    model: str,
    action: str,            # "diagnose" | "rag_explain" | "note_parse" | "voice_response"
    tokens_approx: int = 0,
    user_id: str = "anonymous",
) -> None:
    """Record an AI inference event — required for clinical AI governance."""
    _write({
        "event":        "AI_INFERENCE",
        "timestamp":    _now_iso(),
        "patient_id":   patient_id,
        "model":        model,
        "action":       action,
        "tokens_approx": tokens_approx,
        "user_id":      user_id,
    })


def log_voice_session(
    session_id: str,
    event: str,             # "start" | "transcript" | "response" | "end"
    patient_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """Record voice session lifecycle events."""
    _write({
        "event":      "VOICE_SESSION",
        "timestamp":  _now_iso(),
        "session_id": session_id,
        "sub_event":  event,
        "patient_id": patient_id or "none",
        "metadata":   metadata or {},
    })


def get_recent_entries(n: int = 50) -> list:
    """Read last N audit entries (for /api/audit endpoint)."""
    if not _AUDIT_PATH.exists():
        return []
    try:
        lines = _AUDIT_PATH.read_text(encoding="utf-8").strip().splitlines()
        tail  = lines[-n:] if len(lines) > n else lines
        return [json.loads(l) for l in reversed(tail)]
    except Exception:
        return []
