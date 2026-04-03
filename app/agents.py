"""Multi-agent clinical orchestration pipeline for HC01."""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List

from . import data
from .clients import nim, ollama
from .config import cfg
from .medical_rag import get_medical_rag
from .models import AgentContext
from .outlier_detection import detect_lab_outliers
from .scoring import calc_news2, calc_sofa, cosine_sim, keyword_score

log = logging.getLogger("HC01.agents")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _latest_lab(labs: Dict, key: str) -> float:
    arr = labs.get(key, [])
    if not arr:
        return 0.0
    entry = arr[-1]
    return float(entry["v"] if isinstance(entry, dict) else entry)


async def _ollama_or_nim(ctx: AgentContext, messages: List[Dict], max_tokens: int = 700) -> str:
    """Try Ollama first; fall back to NIM fallback model if Ollama is unavailable."""
    try:
        if await ollama.is_online():
            return await ollama.chat(cfg.NOTE_MODEL, messages, max_tokens=max_tokens)
        raise RuntimeError("Ollama offline")
    except Exception as e_oll:
        key = ctx.nim_key or cfg.nim_key("fallback")
        if not key:
            return f"[Local model unavailable, no NIM key configured: {e_oll}]"
        try:
            return await nim.chat(cfg.FALLBACK_MODEL, messages, key, max_tokens=max_tokens)
        except Exception as e_nim:
            return f"[Both models failed. Ollama: {e_oll} | NIM: {e_nim}]"


# ─── Wave 1 agents ────────────────────────────────────────────────────────────

_SYMPTOM_KW = [
    "fever","chills","tachycardia","hypotension","confusion","altered","dyspnea","shortness of breath",
    "oliguria","hypoxia","hypoxemia","pain","nausea","vomiting","sepsis","shock","acidosis",
    "decreased urine","agitation","lethargy","distension","diarrhea","cough","wheeze",
]
_PROBLEM_KW = [
    "sepsis","septic shock","ards","aki","pneumonia","uti","bacteremia","endocarditis",
    "heart failure","respiratory failure","renal failure","liver failure","coagulopathy",
    "meningitis","encephalopathy","pancreatitis","peritonitis","ileus",
]
_INFECTION_KW = [
    "pneumonia","lung","uti","urinary","blood","bacteremia","skin","wound","catheter",
    "line","abdomen","abdominal","gi tract","endocarditis","meningitis","cns",
]
_VITAL_KW = [
    "tachycardia","bradycardia","hypotension","hypertension","fever","hypothermia",
    "low bp","high bp","low heart rate","high heart rate","dropping","rising","falling",
    "spo2","oxygen","gcs","confused","unresponsive",
]


async def agent_note_parser(ctx: AgentContext) -> None:
    """Rule-based clinical note parser — no LLM, sub-second execution."""
    orch = "CLINICAL"
    await ctx.set_agent_status(orch, "NOTE_PARSER", "active")
    try:
        notes = ctx.patient.notes or []
        if not notes:
            ctx.parsed_notes = {"raw": "No clinical notes available"}
            await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "NOTE_PARSER", "data": ctx.parsed_notes})
            await ctx.set_agent_status(orch, "NOTE_PARSER", "done")
            return

        symptoms: list        = []
        meds_mentioned: list  = []
        timeline_events: list = []
        vital_concerns: list  = []
        active_problems: list = []
        infection_sources: list = []
        seen: set = set()

        for note in notes:
            txt  = note.get("text", "")
            time = note.get("time", "?")
            low  = txt.lower()

            # timeline event per note
            timeline_events.append({"time": time, "event": txt[:120]})

            for kw in _SYMPTOM_KW:
                if kw in low and kw not in seen:
                    symptoms.append(kw)
                    seen.add(kw)

            for kw in _PROBLEM_KW:
                if kw in low and kw not in seen:
                    active_problems.append(kw)
                    seen.add(kw)

            for kw in _INFECTION_KW:
                if kw in low and ("source" in low or "infection" in low or kw in low) and kw not in seen:
                    infection_sources.append(kw)
                    seen.add(kw)

            for kw in _VITAL_KW:
                if kw in low and kw not in seen:
                    vital_concerns.append(kw)
                    seen.add(kw)

        # Medications from patient data (already structured)
        meds_mentioned = list(ctx.patient.medications or [])

        ctx.parsed_notes = {
            "symptoms": symptoms,
            "medications_mentioned": meds_mentioned,
            "timeline_events": timeline_events,
            "vital_concerns": vital_concerns,
            "active_problems": active_problems,
            "infection_sources": infection_sources,
        }
        await ctx.log("NOTE_PARSER",
            f"Rule-based: {len(symptoms)} symptoms, {len(active_problems)} problems, {len(timeline_events)} events",
            "info")

    except Exception as exc:
        log.exception("NOTE_PARSER error")
        ctx.parsed_notes = {"error": str(exc)}

    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "NOTE_PARSER", "data": ctx.parsed_notes})
    await ctx.set_agent_status(orch, "NOTE_PARSER", "done")


async def agent_outlier_detector(ctx: AgentContext) -> None:
    orch = "SAFETY"
    await ctx.set_agent_status(orch, "OUTLIER_DETECTOR", "active")
    try:
        outliers = detect_lab_outliers(ctx.patient.labs, z_threshold=cfg.OUTLIER_Z)
        ctx.outliers = outliers
        if outliers:
            summary = ", ".join(f"{o['lab']}(Z={o['z']})" for o in outliers)
            await ctx.log("OUTLIER_DETECTOR", f"[!] {len(outliers)} outlier(s): {summary}", "warn")
        else:
            await ctx.log("OUTLIER_DETECTOR", "No statistical outliers detected", "info")
    except Exception as exc:
        log.exception("OUTLIER_DETECTOR error")
        ctx.outliers = []

    await ctx.send({"type": "agent_result", "orchestrator": "SAFETY", "agent": "OUTLIER_DETECTOR", "data": ctx.outliers})
    await ctx.set_agent_status("SAFETY", "OUTLIER_DETECTOR", "done")


async def agent_med_safety(ctx: AgentContext) -> None:
    orch = "SAFETY"
    await ctx.set_agent_status(orch, "MED_SAFETY", "active")
    try:
        meds = ctx.patient.medications or []
        meds_lower = [m.lower() for m in meds]
        cr_latest = _latest_lab(ctx.patient.labs, "creatinine")
        aki = cr_latest > 1.5

        has_piptazo = any(
            "piperacillin" in m or "pip-tazo" in m or "tazobactam" in m
            for m in meds_lower
        )

        conflicts = []
        for med in meds:
            ml = med.lower()
            status, conflict, severity = "ok", None, None

            if "vancomycin" in ml and has_piptazo and aki:
                status, severity = "conflict", "HIGH"
                conflict = (
                    f"Vanco + Pip-Tazo: RR 3.7x AKI (MIMIC-IV 2023). "
                    f"Creatinine {cr_latest} mg/dL — nephrology review NOW."
                )
            elif "vancomycin" in ml and aki:
                status, severity = "warn", "MEDIUM"
                conflict = f"Vancomycin AKI risk elevated (Cr {cr_latest}). Switch to AUC/MIC monitoring."
            elif any(x in ml for x in ["gentamicin", "amikacin", "tobramycin"]):
                status, severity = "warn", "MEDIUM"
                conflict = "Aminoglycoside: nephrotoxicity + ototoxicity risk. Daily drug levels required."
            elif any(x in ml for x in ["ibuprofen", "ketorolac", "nsaid"]) and aki:
                status, severity = "conflict", "HIGH"
                conflict = "NSAIDs contraindicated in AKI — prostaglandin-mediated renal perfusion inhibition."

            conflicts.append({"med": med, "status": status, "conflict": conflict, "severity": severity})

        ctx.med_conflicts = conflicts
        high = [c for c in conflicts if c["severity"] == "HIGH"]
        if high:
            await ctx.log("MED_SAFETY", f"[!] {len(high)} HIGH-severity medication conflicts", "warn")
        else:
            await ctx.log("MED_SAFETY", f"Checked {len(meds)} medications — no critical conflicts", "info")
    except Exception as exc:
        log.exception("MED_SAFETY error")
        ctx.med_conflicts = []

    await ctx.send({"type": "agent_result", "orchestrator": "SAFETY", "agent": "MED_SAFETY", "data": ctx.med_conflicts})
    await ctx.set_agent_status("SAFETY", "MED_SAFETY", "done")


async def agent_alert_escalation(ctx: AgentContext) -> None:
    orch = "SAFETY"
    await ctx.set_agent_status(orch, "ALERT_ESCALATION", "active")
    try:
        alerts = []
        pt = ctx.patient
        v  = pt.vitals
        labs = pt.labs

        lactate = _latest_lab(labs, "lactate")   or None
        cr      = _latest_lab(labs, "creatinine") or None
        wbc     = _latest_lab(labs, "wbc")        or None
        pct     = _latest_lab(labs, "procalcitonin") or None

        # Zero means "not measured" in our synthesiser
        if lactate == 0.0: lactate = None
        if cr == 0.0:      cr      = None
        if wbc == 0.0:     wbc     = None

        if lactate and lactate >= 4.0:
            alerts.append({"level": "CRITICAL", "code": "LACTATE_CRITICAL",
                "message": f"Lactate {lactate} mmol/L — critical tissue hypoperfusion. SSC 2021: resuscitation escalation + ICU attending NOW.",
                "intervention": "Fluid bolus 30 mL/kg + vasopressor titration + repeat lactate in 1h"})
        elif lactate and lactate >= 2.0:
            alerts.append({"level": "WARNING", "code": "LACTATE_ELEVATED",
                "message": f"Lactate {lactate} mmol/L — tissue hypoperfusion threshold reached (SSC 2021 §3.1).",
                "intervention": "Guide resuscitation by serial lactate; target clearance ≥10%"})

        if v.bpSys <= 90:
            alerts.append({"level": "CRITICAL", "code": "HYPOTENSION",
                "message": f"SBP {v.bpSys} mmHg — vasopressor threshold. Septic shock criteria met.",
                "intervention": "Norepinephrine 0.01-0.5 mcg/kg/min; MAP target ≥65"})

        if v.spo2 < 90:
            alerts.append({"level": "CRITICAL", "code": "HYPOXEMIA",
                "message": f"SpO2 {v.spo2}% — severe hypoxemia. ARDS evaluation required.",
                "intervention": "Increase FiO2; consider NIV/intubation; prone positioning if P/F <150"})

        if cr and cr >= 3.5:
            alerts.append({"level": "WARNING", "code": "AKI_STAGE3",
                "message": f"Creatinine {cr} mg/dL — KDIGO AKI Stage 3 criteria met.",
                "intervention": "Nephrology consult; hold nephrotoxins; consider RRT"})

        if pct and pct >= 10.0:
            alerts.append({"level": "CRITICAL", "code": "PCT_CRITICAL",
                "message": f"Procalcitonin {pct} ng/mL — severe sepsis/septic shock biomarker.",
                "intervention": "Escalate antimicrobial coverage; reassess source control"})

        if wbc and wbc > 20.0:
            alerts.append({"level": "WARNING", "code": "LEUKOCYTOSIS",
                "message": f"WBC {wbc} x10^3/uL — severe leukocytosis. Bacterial sepsis or haematological process.",
                "intervention": "Blood cultures x2 if not done; haematology review if no infectious source"})

        meds_lower = [m.lower() for m in (pt.medications or [])]
        has_abx = any(
            x in m for m in meds_lower
            for x in ["cefazolin", "pip-tazo", "piperacillin", "meropenem", "vancomycin",
                       "amoxicillin", "azithromycin", "levofloxacin"]
        )
        if lactate and lactate >= 2.0 and not has_abx:
            alerts.append({"level": "CRITICAL", "code": "ANTIBIOTICS_DUE",
                "message": "Sepsis criteria met but broad-spectrum antimicrobials not confirmed (SSC 2021: within 1 hour).",
                "intervention": "Start pip-tazo +/- vancomycin NOW; blood cultures before if possible"})

        ctx.alert_level = (
            "CRITICAL" if any(a["level"] == "CRITICAL" for a in alerts) else
            "WARNING"  if any(a["level"] == "WARNING"  for a in alerts) else
            "WATCH"
        )
        ctx.alert_events = alerts
        await ctx.log("ALERT_ESCALATION", f"Alert level: {ctx.alert_level} | {len(alerts)} active alerts", "info")

    except Exception as exc:
        log.exception("ALERT_ESCALATION error")
        ctx.alert_level = "WATCH"
        ctx.alert_events = []

    await ctx.send({"type": "agent_result", "orchestrator": "SAFETY", "agent": "ALERT_ESCALATION",
                    "data": {"level": ctx.alert_level, "alerts": ctx.alert_events}})
    await ctx.set_agent_status("SAFETY", "ALERT_ESCALATION", "done")


async def agent_trend_classifier(ctx: AgentContext) -> None:
    orch = "TEMPORAL"
    await ctx.set_agent_status(orch, "TREND_CLASSIFIER", "active")
    findings = []
    try:
        labs = ctx.patient.labs
        for key, arr in labs.items():
            if not isinstance(arr, list) or len(arr) < 2:
                continue
            try:
                vals = [float(x["v"] if isinstance(x, dict) else x) for x in arr]
            except (KeyError, TypeError, ValueError):
                continue

            n = len(vals)
            first_v, latest_v = vals[0], vals[-1]
            ref = data.MIMIC_IV["lab_refs"].get(key, {})

            # Least-squares slope
            if n >= 3:
                xs = list(range(n))
                xm = sum(xs) / n
                ym = sum(vals) / n
                num = sum((xs[i] - xm) * (vals[i] - ym) for i in range(n))
                den = sum((xs[i] - xm) ** 2 for i in range(n))
                slope = num / den if den > 1e-10 else 0.0
            else:
                slope = vals[-1] - vals[0]

            pct_change  = ((latest_v - first_v) / max(abs(first_v), 0.01)) * 100
            direction   = "RISING" if slope > 0.05 else "FALLING" if slope < -0.05 else "STABLE"
            is_abnormal = bool(ref) and (
                latest_v > ref.get("high", float("inf")) or
                latest_v < ref.get("low", float("-inf"))
            )

            finding = {
                "lab": key, "direction": direction,
                "slope": round(slope, 3), "pct_change": round(pct_change, 1),
                "latest": latest_v, "first": first_v, "abnormal": is_abnormal,
            }

            if direction == "RISING" and is_abnormal:
                crit_high = ref.get("critical", ref.get("critical_high", float("inf")))
                severity  = "critical" if latest_v > crit_high else "elevated"
                message   = f"{key.upper()} rising {first_v}→{latest_v} ({pct_change:+.1f}%)"
                if key == "lactate" and latest_v >= 2.0:
                    message += " — SSC 2021 lactate threshold breached"
                elif key == "creatinine" and first_v > 0 and (latest_v / first_v) >= 1.5:
                    stage = 1 if (latest_v / first_v) < 2.0 else 2 if (latest_v / first_v) < 3.0 else 3
                    message += f" — KDIGO AKI Stage {stage}"
                findings.append({"lab": key, "severity": severity, "message": message, "finding": finding})

    except Exception as exc:
        log.exception("TREND_CLASSIFIER error")

    ctx.lab_findings = findings
    ctx.trends = {f["lab"]: f["finding"] for f in findings}
    await ctx.log("TREND_CLASSIFIER", f"Classified {len(findings)} abnormal lab trends", "info")
    await ctx.send({"type": "agent_result", "orchestrator": "TEMPORAL", "agent": "TREND_CLASSIFIER", "data": findings})
    await ctx.set_agent_status("TEMPORAL", "TREND_CLASSIFIER", "done")


async def agent_trajectory_predictor(ctx: AgentContext) -> None:
    orch = "TEMPORAL"
    await ctx.set_agent_status(orch, "TRAJECTORY_PREDICTOR", "active")
    predictions = {}
    try:
        labs = ctx.patient.labs
        for key, arr in labs.items():
            if not isinstance(arr, list) or len(arr) < 2:
                continue
            try:
                vals = [float(x["v"] if isinstance(x, dict) else x) for x in arr]
            except (KeyError, TypeError, ValueError):
                continue

            # Weighted slope: recent delta gets 2x weight vs older
            n = len(vals)
            if n >= 3:
                recent_slope = vals[-1] - vals[-2]
                longer_slope = (vals[-1] - vals[0]) / max(n - 1, 1)
                slope = 0.7 * recent_slope + 0.3 * longer_slope
            else:
                slope = vals[-1] - vals[0]

            pred_6h  = round(max(0, vals[-1] + slope * 1), 3)
            pred_12h = round(max(0, vals[-1] + slope * 2), 3)

            ref  = data.MIMIC_IV["lab_refs"].get(key, {})
            high = ref.get("high", float("inf"))
            crit = ref.get("critical", ref.get("critical_high", float("inf")))

            status = (
                "CRITICAL_WITHIN_6H"  if pred_6h  > crit else
                "WARNING_WITHIN_6H"   if pred_6h  > high else
                "NORMALIZING"         if pred_12h <= high < vals[-1] else
                "STABLE"
            )
            predictions[key] = {
                "current": vals[-1], "pred_6h": pred_6h, "pred_12h": pred_12h,
                "trajectory_status": status,
            }

    except Exception as exc:
        log.exception("TRAJECTORY_PREDICTOR error")

    ctx.trajectory = predictions
    critical_labs = [k for k, v in predictions.items() if v["trajectory_status"] == "CRITICAL_WITHIN_6H"]
    if critical_labs:
        await ctx.log("TRAJECTORY_PREDICTOR", f"Projected critical in 6h: {', '.join(critical_labs)}", "warn")
    await ctx.send({"type": "agent_result", "orchestrator": "TEMPORAL", "agent": "TRAJECTORY_PREDICTOR", "data": predictions})
    await ctx.set_agent_status("TEMPORAL", "TRAJECTORY_PREDICTOR", "done")


# ─── Temporal Lab Mapper ─────────────────────────────────────────────────────

async def agent_temporal_lab_mapper(ctx: AgentContext) -> None:
    """Builds a chronological disease progression timeline from vitals + labs + notes."""
    orch = "TEMPORAL"
    await ctx.set_agent_status(orch, "TEMPORAL_LAB_MAPPER", "active")
    try:
        pt   = ctx.patient
        labs = pt.labs or {}

        # Collect all timestamped events
        events: List[Dict] = []

        # Lab time-series events
        for lab_name, arr in labs.items():
            if not isinstance(arr, list):
                continue
            ref  = data.MIMIC_IV["lab_refs"].get(lab_name, {})
            high = ref.get("high", float("inf"))
            low  = ref.get("low", float("-inf"))
            crit = ref.get("critical", ref.get("critical_high", float("inf")))
            unit = ref.get("unit", "")
            for entry in arr:
                if not isinstance(entry, dict):
                    continue
                v, t = entry.get("v"), entry.get("t", "")
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                flag = (
                    "CRITICAL" if fv > crit else
                    "ABNORMAL" if fv > high or fv < low else
                    "NORMAL"
                )
                events.append({
                    "time":   t,
                    "type":   "LAB",
                    "label":  lab_name.upper(),
                    "value":  fv,
                    "unit":   unit,
                    "flag":   flag,
                })

        # Vitals snapshot as a single event at latest time
        v = pt.vitals
        vitals_flags = []
        if v.hr and (v.hr > 100 or v.hr < 60):     vitals_flags.append(f"HR {v.hr}")
        if v.bpSys and v.bpSys <= 90:               vitals_flags.append(f"SBP {v.bpSys} (HYPOTENSION)")
        if v.spo2 and v.spo2 < 92:                  vitals_flags.append(f"SpO2 {v.spo2}% (HYPOXEMIA)")
        if v.rr and v.rr > 20:                      vitals_flags.append(f"RR {v.rr} (TACHYPNOEA)")
        if v.temp and v.temp > 38.3:                vitals_flags.append(f"Temp {v.temp}C (FEVER)")
        if v.gcs and v.gcs < 15:                    vitals_flags.append(f"GCS {v.gcs}/15 (ALTERED)")
        if vitals_flags:
            events.append({
                "time":  "current",
                "type":  "VITALS",
                "label": "VITAL SIGNS",
                "value": "; ".join(vitals_flags),
                "unit":  "",
                "flag":  "ABNORMAL",
            })

        # Note timeline events from NOTE_PARSER output
        if ctx.parsed_notes:
            for ev in ctx.parsed_notes.get("timeline_events", []):
                if ev.get("event"):
                    events.append({
                        "time":  ev.get("time", "?"),
                        "type":  "NOTE",
                        "label": "CLINICAL NOTE",
                        "value": ev["event"],
                        "unit":  "",
                        "flag":  "INFO",
                    })

        # Sort: put timestamped items first, then "current"
        def sort_key(e):
            t = e.get("time", "")
            return ("z" + t) if t == "current" else (t or "")

        events.sort(key=sort_key)

        # Build human-readable timeline string
        timeline_lines = []
        for e in events:
            flag_tag = f"[{e['flag']}]" if e["flag"] not in ("NORMAL", "INFO") else ""
            if e["type"] == "LAB":
                line = f"  {e['time'][:16]:16} | {e['type']:7} | {e['label']:16} {e['value']} {e['unit']} {flag_tag}"
            else:
                line = f"  {'current':16} | {e['type']:7} | {e['value']} {flag_tag}"
            timeline_lines.append(line)

        ctx.disease_timeline = events
        ctx.timeline_str = "\n".join(timeline_lines) if timeline_lines else "  No timestamped data available"

        await ctx.log("TEMPORAL_LAB_MAPPER",
                      f"Timeline built: {len(events)} events "
                      f"({sum(1 for e in events if e['flag'] not in ('NORMAL','INFO'))} abnormal)",
                      "info")
    except Exception as exc:
        log.exception("TEMPORAL_LAB_MAPPER error")
        ctx.disease_timeline = []
        ctx.timeline_str = "Timeline unavailable"

    await ctx.send({
        "type": "agent_result", "orchestrator": orch, "agent": "TEMPORAL_LAB_MAPPER",
        "data": {"event_count": len(ctx.disease_timeline), "timeline": ctx.timeline_str},
    })
    await ctx.set_agent_status(orch, "TEMPORAL_LAB_MAPPER", "done")


# ─── Wave 2 agents ────────────────────────────────────────────────────────────

async def agent_semantic_retriever(ctx: AgentContext) -> List[Dict]:
    orch = "EVIDENCE"
    await ctx.set_agent_status(orch, "SEMANTIC_RETRIEVER", "active")
    retrieved: List[Dict] = []

    try:
        finding_msgs = [f["message"] for f in (ctx.lab_findings or [])]
        alert_msgs   = [a["message"] for a in (ctx.alert_events or [])]
        query_parts  = [
            ctx.patient.admitDiag,
            *finding_msgs[:3],
            *ctx.patient.medications[:5],
            f"sepsis lactate {_latest_lab(ctx.patient.labs, 'lactate')} mmol/L",
            "SOFA organ failure septic shock",
        ] + alert_msgs[:2]
        query = ". ".join(str(x) for x in query_parts if x)

        # First preference: local medical vector DB built from guideline PDFs.
        try:
            rag = get_medical_rag(cfg.MEDICAL_RAG_DB_DIR)
            retrieved = await rag.query(query, top_k=cfg.TOP_K_GUIDELINES)
            if retrieved:
                await ctx.log("SEMANTIC_RETRIEVER", f"Medical RAG DB hit: {len(retrieved)} chunks", "info")
        except Exception as exc:
            await ctx.log("SEMANTIC_RETRIEVER", f"Medical RAG DB unavailable ({exc}), falling back", "warn")

        # NIM semantic retrieval: embed query + cosine similarity against pre-embedded guidelines (~0.5s)
        if data._guideline_embeddings and not retrieved:
            nim_key = ctx.nim_key or cfg.nim_key("fallback") or cfg.nim_key("chief")
            if nim_key:
                try:
                    await ctx.log("SEMANTIC_RETRIEVER", "NIM query embedding (nv-embedqa-e5-v5)…", "info")
                    q_vecs = await nim.embed([query[:512]], nim_key, input_type="query")
                    q_emb  = q_vecs[0]
                    scored = [
                        {**g, "score": round(float(cosine_sim(q_emb, data._guideline_embeddings[i])), 4)}
                        for i, g in enumerate(data.GUIDELINES)
                        if data._guideline_embeddings[i] is not None
                    ]
                    scored.sort(key=lambda x: x["score"], reverse=True)
                    retrieved = scored[:cfg.TOP_K_GUIDELINES]
                    await ctx.log("SEMANTIC_RETRIEVER",
                        f"Semantic retrieval done. Top score: {retrieved[0]['score']:.4f}", "info")
                except Exception as e:
                    await ctx.log("SEMANTIC_RETRIEVER", f"NIM embed failed ({e}), keyword fallback", "warn")

        if not retrieved:
            lower = query.lower()
            retrieved = sorted(
                [{**g, "score": keyword_score(lower, g["keywords"])} for g in data.GUIDELINES],
                key=lambda x: x["score"], reverse=True,
            )[:cfg.TOP_K_GUIDELINES]
            await ctx.log("SEMANTIC_RETRIEVER", "Using keyword fallback retrieval", "warn")

    except Exception as exc:
        log.exception("SEMANTIC_RETRIEVER error")
        retrieved = data.GUIDELINES[:cfg.TOP_K_GUIDELINES]

    ctx.retrieved_guidelines = retrieved
    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "SEMANTIC_RETRIEVER",
                    "data": [{"source": g.get("source", g.get("citation", "")),
                               "section": g["section"], "score": g.get("score", 0), "text": g["text"]}
                              for g in retrieved]})
    await ctx.set_agent_status(orch, "SEMANTIC_RETRIEVER", "done")
    return retrieved


# ─── Explicit PS roles ───────────────────────────────────────────────────────

async def run_note_parser_agent(ctx: AgentContext) -> None:
    await ctx.set_orch_status("NOTE_PARSER_AGENT", "active")
    await agent_note_parser(ctx)
    await ctx.set_orch_status("NOTE_PARSER_AGENT", "done")


async def run_temporal_lab_mapper_agent(ctx: AgentContext) -> None:
    await ctx.set_orch_status("TEMPORAL_LAB_MAPPER_AGENT", "active")
    await asyncio.gather(
        agent_outlier_detector(ctx),
        agent_med_safety(ctx),
        agent_alert_escalation(ctx),
        agent_trend_classifier(ctx),
        agent_trajectory_predictor(ctx),
        return_exceptions=True,
    )
    await agent_temporal_lab_mapper(ctx)
    await ctx.set_orch_status("TEMPORAL_LAB_MAPPER_AGENT", "done")


async def run_guideline_rag_agent(ctx: AgentContext) -> None:
    await ctx.set_orch_status("GUIDELINE_RAG_AGENT", "active")
    guidelines = await agent_semantic_retriever(ctx)
    await agent_rag_explainer(ctx, guidelines)
    await ctx.set_orch_status("GUIDELINE_RAG_AGENT", "done")


async def run_chief_synthesis_agent(ctx: AgentContext) -> None:
    await ctx.set_orch_status("CHIEF_SYNTHESIS_AGENT", "active")
    await agent_synthesis(ctx)
    await ctx.set_orch_status("CHIEF_SYNTHESIS_AGENT", "done")


async def agent_rag_explainer(ctx: AgentContext, guidelines: List[Dict]) -> None:
    orch = "EVIDENCE"
    await ctx.set_agent_status(orch, "RAG_EXPLAINER", "active")
    explanation = ""

    try:
        guideline_ctx = "\n\n".join(
            f"[{g.get('source', g.get('citation', ''))} {g['section']}]: {g['text']}"
            for g in (guidelines or [])
        )
        findings_str  = "; ".join(f["message"] for f in (ctx.lab_findings or [])) or "No major lab trends"
        outlier_str   = "; ".join(
            f"OUTLIER {o['lab'].upper()} value {o['value']} (Z={o['z']}) vs 72h mean {o['mean']}"
            for o in (ctx.outliers or [])
        ) or "None"

        messages = [
            {"role": "system",
             "content": (
                 "You are a clinical evidence agent. Output ONLY 3 bullet points. "
                 "Each bullet must be <= 24 words, clinically actionable, and end with one citation in [Source §Section] format. "
                 "No preamble, no rationale section, no chain-of-thought."
             )},
            {"role": "user",
             "content": (
                 f"Patient: {ctx.patient.name}, {ctx.patient.age}{ctx.patient.sex}. "
                 f"{ctx.patient.admitDiag}.\n"
                 f"Lab findings: {findings_str}\nOutliers: {outlier_str}\n\n"
                 f"Guidelines:\n{guideline_ctx}\n\nReturn exactly 3 bullets."
             )},
        ]
        # Prefer NIM fallback model here to minimize verbose reasoning leakage.
        fallback_key = ctx.nim_key or cfg.nim_key("fallback")
        if fallback_key:
            explanation = await nim.chat(cfg.FALLBACK_MODEL, messages, fallback_key, max_tokens=350)
        else:
            explanation = await _ollama_or_nim(ctx, messages, max_tokens=350)
        # Keep only concise final bullets if model emits extra sections.
        lines = [ln.strip() for ln in explanation.splitlines() if ln.strip()]
        bullet_lines = [ln for ln in lines if ln.startswith("-") or ln.startswith("*")]
        if bullet_lines:
            compact = []
            for ln in bullet_lines[:3]:
                text = ln.lstrip("-* ").strip().replace('"', "")
                words = text.split()
                if len(words) > 24:
                    text = " ".join(words[:24]) + "..."
                compact.append(f"- {text}")
            explanation = "\n".join(compact)
        else:
            compact = []
            for ln in lines[:3]:
                words = ln.split()
                compact.append(" ".join(words[:24]) + ("..." if len(words) > 24 else ""))
            explanation = "\n".join(compact)
    except Exception as exc:
        log.exception("RAG_EXPLAINER error")
        explanation = f"RAG explanation unavailable: {exc}"

    ctx.rag_explanation = explanation
    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "RAG_EXPLAINER",
                    "data": {"explanation": explanation}})
    await ctx.set_agent_status(orch, "RAG_EXPLAINER", "done")


# ─── Wave 3 — Chief synthesis ─────────────────────────────────────────────────

async def agent_synthesis(ctx: AgentContext) -> None:
    orch = "SYNTHESIS"
    await ctx.set_agent_status(orch, "CHIEF_AGENT", "active")

    pt     = ctx.patient
    sofa   = ctx.sofa or {}
    news2  = ctx.news2 or {}
    labs   = pt.labs or {}

    # Build lab trajectory string
    lab_traj_lines = []
    for k, v in labs.items():
        if not isinstance(v, list):
            continue
        ref  = data.MIMIC_IV["lab_refs"].get(k, {})
        unit = ref.get("unit", "")
        vals = " -> ".join(str(x.get("v", x)) for x in v)
        lab_traj_lines.append(f"  {k.upper()}: {vals} {unit}")
    lab_traj = "\n".join(lab_traj_lines) or "  No lab data available"

    # Outlier note
    outlier_note = "\n".join(
        f"!! OUTLIER HELD: {o['lab'].upper()} {o['value']} at {o['timepoint']} "
        f"(Z={o['z']} vs 72h mean {o['mean']}+/-{o['std']}) — DIAGNOSIS HELD, REDRAW REQUIRED"
        for o in (ctx.outliers or [])
    )

    # Medication conflicts
    med_conflicts_str = "\n".join(
        f"  - {c['med']}: {c['conflict']}"
        for c in (ctx.med_conflicts or []) if c.get("conflict")
    ) or "  No conflicts identified"

    # Guideline context
    guideline_ctx = "\n".join(
        f"[{g.get('source', g.get('citation', ''))} {g['section']}]: {g['text']}"
        for g in (ctx.retrieved_guidelines or [])
    )

    # Alert summary
    alert_summary = "\n".join(
        f"  [{a['level']}] {a['message']}"
        for a in (ctx.alert_events or [])
    ) or "  No active alerts"

    # Trajectory summary
    traj_summary = "\n".join(
        f"  {k.upper()}: pred 6h={v.get('pred_6h','?')} 12h={v.get('pred_12h','?')} [{v.get('trajectory_status','?')}]"
        for k, v in (ctx.trajectory or {}).items()
    ) or "  Insufficient data"

    # Temporal findings
    findings_str = "\n".join(
        f"  - {f['message']}" for f in (ctx.lab_findings or [])
    ) or "  None"

    sofa_str = (
        f"{sofa.get('total','?')}/24 "
        f"(Resp:{sofa.get('resp','?')} Coag:{sofa.get('coag','?')} "
        f"Liver:{sofa.get('liver','?')} Cardio:{sofa.get('cardio','?')} "
        f"CNS:{sofa.get('cns','?')} Renal:{sofa.get('renal','?')}, "
        f"MIMIC-IV predicted mortality {sofa.get('mortality_pct','?')}%)"
    )

    # Build outlier safety block — critical for PS compliance
    outlier_safety_block = ""
    if ctx.outliers:
        outlier_safety_block = "OUTLIER SAFETY BLOCK — DIAGNOSIS HELD ON THESE LABS:\n" + "\n".join(
            f"  !! {o['lab'].upper()} = {o['value']} at {o['timepoint']} "
            f"(Z-score {o['z']} vs 72h mean {o['mean']}+/-{o['std']}) "
            f"— PROBABLE LAB ERROR. Chief Agent must NOT update diagnosis on this value. Confirmed redraw required."
            for o in ctx.outliers
        )

    # Timeline string from TEMPORAL_LAB_MAPPER
    timeline_block = ctx.timeline_str or "Timeline not available"

    prompt = f"""You are the Chief Synthesis Agent for HC01 — ICU Diagnostic Risk Assistant.
In the ICU, hours — sometimes minutes — determine survival outcomes. Your role is to integrate all agent outputs below and produce a structured Diagnostic Risk Report.

CRITICAL RULES:
1. Any lab value flagged in the OUTLIER SAFETY BLOCK below is a PROBABLE LAB ERROR. You MUST flag it as held, cite the Z-score, and REFUSE to alter the diagnosis based on that value until a confirmed redraw is received.
2. Every flagged risk MUST cite the specific clinical guideline or dataset that supports it.
3. End with the mandatory safety disclaimer.
Respond ONLY with the structured output — no preamble, no chain-of-thought before CLINICAL ASSESSMENT.

PATIENT: {pt.name}, {pt.age}{pt.sex}, Day {pt.daysInICU:.1f} ICU, {pt.weight} kg
ADMISSION: {pt.admitDiag}

VITALS: HR {pt.vitals.hr} bpm | BP {pt.vitals.bpSys}/{pt.vitals.bpDia} mmHg | MAP {pt.vitals.map} mmHg | RR {pt.vitals.rr}/min | SpO2 {pt.vitals.spo2}% | Temp {pt.vitals.temp}C | GCS {pt.vitals.gcs}/15 | FiO2 {pt.vitals.fio2}

DISEASE PROGRESSION TIMELINE (chronological):
{timeline_block}

LAB TRAJECTORIES (72h summary):
{lab_traj}

SOFA: {sofa_str}
NEWS2: {news2.get('total','?')}/20 — {news2.get('level','?')} RISK

TEMPORAL LAB FLAGS (from Temporal Lab Mapper Agent):
{findings_str}

TRAJECTORY PREDICTIONS (6h/12h):
{traj_summary}

{outlier_safety_block}

ACTIVE ALERTS ({ctx.alert_level}) — from Alert Escalation Agent:
{alert_summary}

MEDICATION SAFETY — from Med Safety Agent:
{med_conflicts_str}

GUIDELINE RAG EVIDENCE — from Guideline RAG Agent:
{guideline_ctx}

RAG AGENT SYNTHESIS:
{ctx.rag_explanation or 'Not available'}

PARSED NOTE FINDINGS — from Note Parser Agent:
{json.dumps(ctx.parsed_notes)[:600] if ctx.parsed_notes else 'N/A'}

DATASETS: MIMIC-IV v2.2 (73,141 ICU stays), eICU-CRD (200,859 encounters), PhysioNet Sepsis Challenge 2019 (60,000+ records)

Produce EXACTLY this structure:

CLINICAL ASSESSMENT:
[3 sentences: current clinical picture, trajectory from timeline, immediate survival risk]

DISEASE PROGRESSION TIMELINE SUMMARY:
[2-3 sentences narrating how key vitals and labs shifted chronologically — when did deterioration begin, what changed, what is the current trajectory]

DIFFERENTIAL DIAGNOSIS:
1. [Diagnosis] — [XX]% confidence — [cite specific guideline §section or MIMIC-IV stat]
2. [Diagnosis] — [XX]% confidence — [cite guideline]
3. [Diagnosis] — [XX]% confidence — [cite guideline]

KEY CONCERNS (with guideline citations):
- [Concern] [Source §Section]
- [Concern] [Source §Section]

OUTLIER FLAGS (from Outlier Detection Module):
[List each held lab with Z-score, reason flagged as probable error, and required action — OR "No outliers detected"]

RECOMMENDED ACTIONS:
- [Action] — [IMMEDIATE / URGENT / ROUTINE]

SHIFT HANDOVER BRIEF (30 seconds):
[2-3 plain English sentences for the incoming doctor]

---
SAFETY DISCLAIMER: This report is generated by an AI decision-support system (HC01). All outputs are decision-support only and do NOT constitute a clinical diagnosis. All flagged risks, drug interactions, and recommended actions must be verified and acted upon by a qualified clinician. Held diagnoses must not be confirmed until laboratory redraws are obtained and reviewed by clinical staff."""

    messages  = [
        {"role": "system", "content": (
            "You are the Chief Synthesis Agent for HC01. "
            "Output ONLY the structured clinical report. "
            "No reasoning. No meta-commentary. No thinking. "
            "Start your response immediately with 'CLINICAL ASSESSMENT:'"
        )},
        {"role": "user", "content": "/no_think\n\n" + prompt},
    ]
    full_text = ""

    async def on_chunk(delta: str, accumulated: str) -> None:
        nonlocal full_text
        full_text = accumulated
        await ctx.send({"type": "stream_chunk", "orchestrator": orch, "agent": "CHIEF_AGENT", "content": delta})

    nim_key = ctx.nim_key or cfg.nim_key("chief")

    # Synthesis: NIM Chief (best quality) → NIM Fallback → local qwen3 → rule-based
    try:
        if not nim_key:
            raise RuntimeError("No NIM API key configured")
        await ctx.log("CHIEF_AGENT", "NIM Chief synthesis starting…", "info")
        full_text = await nim.chat(cfg.CHIEF_MODEL, messages, nim_key, max_tokens=1200, on_chunk=on_chunk)
    except Exception as e_chief:
        await ctx.log("CHIEF_AGENT", f"NIM Chief failed ({e_chief}), trying NIM fallback…", "warn")
        fallback_key = ctx.nim_key or cfg.nim_key("fallback")
        try:
            if not fallback_key:
                raise RuntimeError("No NIM fallback key configured")
            full_text = await nim.chat(cfg.FALLBACK_MODEL, messages, fallback_key, max_tokens=1500, on_chunk=on_chunk)
        except Exception as e_fallback:
            await ctx.log("CHIEF_AGENT", f"NIM fallback failed ({e_fallback}), trying local model…", "warn")
            try:
                if await ollama.is_online():
                    full_text = await ollama.chat(cfg.NOTE_MODEL, messages, max_tokens=1200, on_chunk=on_chunk)
                else:
                    raise RuntimeError("Ollama offline")
            except Exception as e_local:
                await ctx.log("CHIEF_AGENT", f"All LLM endpoints failed — using rule-based: {e_local}", "error")
                full_text = _rule_based_synthesis(ctx, sofa, news2)
                await on_chunk(full_text, full_text)

    # Extract structured sections — strip reasoning/chain-of-thought preamble
    # First try to find CLINICAL ASSESSMENT as entry point
    ca_match = re.search(r'(CLINICAL ASSESSMENT:[\s\S]+)', full_text, re.I)
    structured = ca_match.group(1).strip() if ca_match else full_text

    # Strip think blocks if model included them anyway
    structured = re.sub(r'<think>[\s\S]*?</think>', '', structured, flags=re.I).strip()

    # Clean up lines that are pure reasoning meta-commentary (heuristic: lines starting with "But ", "Now ", "Let's ", "We ")
    lines = structured.split('\n')
    clean_lines = []
    in_section = False
    section_headers = {'CLINICAL ASSESSMENT', 'DISEASE PROGRESSION', 'DIFFERENTIAL DIAGNOSIS',
                       'KEY CONCERNS', 'OUTLIER FLAGS', 'RECOMMENDED ACTIONS', 'SHIFT HANDOVER BRIEF',
                       'SAFETY DISCLAIMER', '---'}
    for line in lines:
        stripped = line.strip()
        is_header = any(stripped.upper().startswith(h) for h in section_headers)
        is_bullet = stripped.startswith(('-', '*', '1.', '2.', '3.', '4.', '5.'))
        is_blank  = not stripped
        # Skip pure reasoning lines that aren't section content
        is_reasoning = (
            not in_section and
            any(stripped.startswith(p) for p in ("But ", "Now ", "Let's ", "We need", "We can", "We must", "So we", "Given ", "Looking at", "I think"))
        )
        if is_header:
            in_section = True
        if not is_reasoning or in_section or is_bullet or is_blank:
            clean_lines.append(line)

    ctx.synthesis = '\n'.join(clean_lines).strip()
    hm = re.search(r'SHIFT HANDOVER BRIEF[^:]*:\s*([\s\S]+?)(?:\n\n|\n---|\n#|$)', ctx.synthesis, re.I)
    ctx.handover  = hm.group(1).strip() if hm else ""

    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "CHIEF_AGENT",
                    "data": {"synthesis": full_text, "handover": ctx.handover}})
    await ctx.set_agent_status(orch, "CHIEF_AGENT", "done")


def _rule_based_synthesis(ctx: AgentContext, sofa: Dict, news2: Dict) -> str:
    """Deterministic fallback when all LLM endpoints fail."""
    pt    = ctx.patient
    total = sofa.get("total", "?")
    mort  = sofa.get("mortality_pct", "?")
    n2    = news2.get("total", "?")
    level = news2.get("level", "?")
    alerts = "\n".join(f"- [{a['level']}] {a['message']}" for a in (ctx.alert_events or []))
    meds   = "\n".join(f"- {c['med']}: {c['conflict']}" for c in (ctx.med_conflicts or []) if c.get("conflict")) or "- None"
    return f"""CLINICAL ASSESSMENT:
{pt.name} ({pt.age}{pt.sex}) on Day {pt.daysInICU:.1f} ICU with {pt.admitDiag}. SOFA {total}/24 (predicted mortality {mort}%), NEWS2 {n2}/20 ({level} risk). Alert status: {ctx.alert_level}.

ACTIVE ALERTS:
{alerts or "None"}

MEDICATION SAFETY:
{meds}

SHIFT HANDOVER BRIEF (30 seconds):
Patient admitted for {pt.admitDiag}. SOFA {total}, NEWS2 {n2}. Alert level {ctx.alert_level}. Review active alerts and medication conflicts above. NIM AI synthesis unavailable — clinical team review required.
"""


# ─── Master orchestrator ──────────────────────────────────────────────────────

async def master_orchestrate(ctx: AgentContext) -> None:
    start = time.time()
    pt    = ctx.patient
    await ctx.log("HC01", f"=== PIPELINE START: {pt.name} {pt.age}{pt.sex} ===", "info")
    await ctx.send({"type": "orchestrator_status", "orchestrator": "MASTER", "status": "active"})

    # Scoring (synchronous, no LLM)
    try:
        ctx.sofa  = calc_sofa(pt)
        ctx.news2 = calc_news2(pt)
    except Exception as exc:
        log.exception("SOFA/NEWS2 calculation failed")
        ctx.sofa  = {"total": 0, "mortality_pct": 0}
        ctx.news2 = {"total": 0, "level": "LOW"}

    await ctx.send({"type": "agent_result", "orchestrator": "SCORING", "agent": "SOFA_NEWS2",
                    "data": {"sofa": ctx.sofa, "news2": ctx.news2}})

    # Explicit 4-role pipeline required by PS.
    # NOTE_PARSER and the safety/temporal agents are independent — run in parallel.
    # agent_temporal_lab_mapper needs parsed_notes so it runs after both complete.
    await ctx.set_orch_status("NOTE_PARSER_AGENT", "active")
    await ctx.set_orch_status("TEMPORAL_LAB_MAPPER_AGENT", "active")
    await asyncio.gather(
        agent_note_parser(ctx),
        asyncio.gather(
            agent_outlier_detector(ctx),
            agent_med_safety(ctx),
            agent_alert_escalation(ctx),
            agent_trend_classifier(ctx),
            agent_trajectory_predictor(ctx),
            return_exceptions=True,
        ),
        return_exceptions=True,
    )
    await agent_temporal_lab_mapper(ctx)
    await ctx.set_orch_status("NOTE_PARSER_AGENT", "done")
    await ctx.set_orch_status("TEMPORAL_LAB_MAPPER_AGENT", "done")
    await run_guideline_rag_agent(ctx)
    await run_chief_synthesis_agent(ctx)

    elapsed = round((time.time() - start) * 1000)
    await ctx.send({
        "type":          "pipeline_complete",
        "duration_ms":   elapsed,
        "alert_level":   ctx.alert_level,
        "sofa_total":    ctx.sofa.get("total", 0),
        "news2_total":   ctx.news2.get("total", 0),
    })
    await ctx.log("HC01", f"=== PIPELINE COMPLETE in {elapsed} ms ===", "info")


async def preembed_guidelines() -> None:
    """Pre-compute NIM bge embeddings for all guidelines at startup (single batch ~0.6s)."""
    nim_key = cfg.nim_key("fallback") or cfg.nim_key("chief")
    if not nim_key:
        log.info("No NIM key — skipping guideline pre-embedding (keyword fallback active)")
        return
    try:
        texts = [
            f"{g.get('source', g.get('citation', ''))} {g['section']}: {g['text']}"
            for g in data.GUIDELINES
        ]
        vectors = await nim.embed(texts, nim_key, input_type="passage")
        data._guideline_embeddings = vectors
        log.info("NIM pre-embedded %d guidelines (nvidia/nv-embedqa-e5-v5)", len(vectors))
    except Exception as exc:
        log.warning("NIM guideline pre-embedding failed (%s) — keyword fallback active", exc)
