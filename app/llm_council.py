"""Optional wrapper for querying the Karpathy LLM Council Gradio endpoint."""

import asyncio
import importlib
import logging
from typing import Optional

from .config import cfg

log = logging.getLogger("HC01.llm_council")


class LLMCouncilError(RuntimeError):
    """Raised when LLM council invocation fails."""


def _call_council_sync(question: str) -> str:
    try:
        gradio_client_mod = importlib.import_module("gradio_client")
        client_cls = getattr(gradio_client_mod, "Client")
    except Exception as exc:
        raise LLMCouncilError(
            "gradio_client not installed. Run: pip install gradio_client"
        ) from exc

    client = client_cls(cfg.LLM_COUNCIL_SPACE)
    result = client.predict(question=question, api_name="/ask_council")
    return str(result)


async def ask_council(question: str) -> str:
    """Query the council asynchronously while using the sync Gradio client under the hood."""
    if not cfg.ENABLE_LLM_COUNCIL:
        raise LLMCouncilError("LLM council is disabled by configuration")

    q = (question or "").strip()
    if not q:
        raise LLMCouncilError("Council question cannot be empty")

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_call_council_sync, q),
            timeout=cfg.LLM_COUNCIL_TIMEOUT,
        )
    except asyncio.TimeoutError as exc:
        raise LLMCouncilError(
            f"Council request timed out after {cfg.LLM_COUNCIL_TIMEOUT:.0f}s"
        ) from exc
    except Exception as exc:
        log.exception("LLM council request failed")
        raise LLMCouncilError(str(exc)) from exc


def build_council_question(
    patient_name: str,
    admit_diag: str,
    alert_level: str,
    sofa_total: Optional[int],
    news2_total: Optional[int],
    timeline_str: str,
    synthesis: str,
) -> str:
    """Create a bounded second-opinion prompt for the external council."""
    timeline = (timeline_str or "Timeline unavailable")[:1800]
    synth = (synthesis or "Synthesis unavailable")[:2500]
    return (
        "You are an ICU second-opinion council. Review this decision-support output and return: "
        "(1) key agreement points, (2) key risks/omissions, (3) whether outlier-hold behavior appears safe, "
        "(4) confidence in current risk level. Keep concise and clinically grounded.\n\n"
        f"Patient: {patient_name}\n"
        f"Admission: {admit_diag}\n"
        f"Current alert level: {alert_level}\n"
        f"SOFA: {sofa_total}\n"
        f"NEWS2: {news2_total}\n"
        "Disease timeline:\n"
        f"{timeline}\n\n"
        "Chief synthesis report:\n"
        f"{synth}\n"
    )
