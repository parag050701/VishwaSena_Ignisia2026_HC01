import asyncio
import json
import math
import re
import time
from typing import Dict, List

from . import data
from .clients import nim, ollama
from .config import cfg
from .models import AgentContext
from .scoring import calc_news2, calc_sofa, cosine_sim, keyword_score


async def agent_note_parser(ctx: AgentContext):
    orch = "CLINICAL"
    await ctx.set_agent_status(orch, "NOTE_PARSER", "active")
    notes_text = "\n\n".join(f"[{n['time']}] {n['author']}: {n['text']}" for n in ctx.patient.notes)
    sys_prompt = (
        "You are a clinical NLP agent. Parse ICU notes and return ONLY a JSON object: "
        "{symptoms:[], medications_mentioned:[], timeline_events:[{time,event}], "
        "vital_concerns:[], active_problems:[], infection_sources:[]}. No preamble."
    )
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Parse:\n\n{notes_text}"}]
    raw = ""
    try:
        if await ollama.is_online():
            await ctx.log("NOTE_PARSER", f"Calling {cfg.NOTE_MODEL} LOCAL (Ollama)...")
            raw = await ollama.chat(cfg.NOTE_MODEL, messages, max_tokens=700)
        else:
            raise RuntimeError("Ollama offline")
    except Exception as e:
        await ctx.log("NOTE_PARSER", f"Ollama failed ({e}), falling back to NIM...", "warn")
        try:
            raw = await nim.chat(cfg.FALLBACK_MODEL, messages, ctx.nim_key, max_tokens=600)
        except Exception as e2:
            await ctx.log("NOTE_PARSER", f"NIM also failed: {e2}", "error")

    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        try:
            ctx.parsed_notes = json.loads(m.group())
            problems = len(ctx.parsed_notes.get("active_problems", []))
            symptoms = len(ctx.parsed_notes.get("symptoms", []))
            await ctx.log("NOTE_PARSER", f"Extracted {symptoms} symptoms, {problems} active problems", "info")
        except Exception:
            ctx.parsed_notes = {"raw": raw[:800]}
    else:
        ctx.parsed_notes = {"raw": raw[:800]}

    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "NOTE_PARSER", "data": ctx.parsed_notes})
    await ctx.set_agent_status(orch, "NOTE_PARSER", "done")


async def agent_outlier_detector(ctx: AgentContext):
    orch = "SAFETY"
    await ctx.set_agent_status(orch, "OUTLIER_DETECTOR", "active")
    labs = ctx.patient.labs
    outliers = []
    for name, arr in labs.items():
        if not isinstance(arr, list) or len(arr) < 3:
            continue
        vals = [x["v"] for x in arr]
        history, latest_v = vals[:-1], vals[-1]
        mean = sum(history) / len(history)
        std = math.sqrt(sum((x - mean) ** 2 for x in history) / len(history))
        if std == 0:
            continue
        z = abs((latest_v - mean) / std)
        if z > cfg.OUTLIER_Z:
            outliers.append({
                "lab": name,
                "value": latest_v,
                "mean": round(mean, 2),
                "std": round(std, 2),
                "z": round(z, 2),
                "timepoint": arr[-1]["t"],
                "action": "HOLD DIAGNOSIS — Recommend confirmed redraw",
            })
    ctx.outliers = outliers
    if outliers:
        await ctx.log("OUTLIER_DETECTOR", f"⚠ {len(outliers)} outlier(s): {', '.join(o['lab'] + '(Z=' + str(o['z']) + ')' for o in outliers)}", "warn")
    else:
        await ctx.log("OUTLIER_DETECTOR", "No statistical outliers detected", "info")
    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "OUTLIER_DETECTOR", "data": outliers})
    await ctx.set_agent_status(orch, "OUTLIER_DETECTOR", "done")


async def agent_med_safety(ctx: AgentContext):
    orch = "SAFETY"
    await ctx.set_agent_status(orch, "MED_SAFETY", "active")
    meds = ctx.patient.medications
    labs = ctx.patient.labs
    cr_arr = labs.get("creatinine", [])
    cr_latest = cr_arr[-1]["v"] if cr_arr else 0
    aki = cr_latest > 1.5

    conflicts = []
    meds_lower = [m.lower() for m in meds]
    has_piptazo = any("piperacillin" in m or "pip-tazo" in m or "tazobactam" in m for m in meds_lower)

    for med in meds:
        ml = med.lower()
        status = "ok"
        conflict = None
        severity = None
        if "vancomycin" in ml and has_piptazo and aki:
            status = "conflict"
            conflict = f"Vanco + Pip-Tazo: RR 3.7× AKI (MIMIC-IV 2023). Creatinine {cr_latest} mg/dL — nephrology review NOW."
            severity = "HIGH"
        elif "vancomycin" in ml and aki:
            status = "warn"
            conflict = f"Vancomycin AKI risk elevated (Cr {cr_latest}). Switch to AUC/MIC monitoring."
            severity = "MEDIUM"
        elif any(x in ml for x in ["gentamicin", "amikacin", "tobramycin"]):
            status = "warn"
            conflict = "Aminoglycoside: cumulative nephrotoxicity + ototoxicity. Daily drug levels required."
            severity = "MEDIUM"
        elif any(x in ml for x in ["ibuprofen", "ketorolac"]) and aki:
            status = "conflict"
            conflict = "NSAIDs contraindicated in AKI — prostaglandin-mediated renal perfusion inhibition."
            severity = "HIGH"
        conflicts.append({"med": med, "status": status, "conflict": conflict, "severity": severity})

    ctx.med_conflicts = conflicts
    high_conflicts = [c for c in conflicts if c["severity"] == "HIGH"]
    if high_conflicts:
        await ctx.log("MED_SAFETY", f"⚠ {len(high_conflicts)} HIGH-severity medication conflicts", "warn")
    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "MED_SAFETY", "data": conflicts})
    await ctx.set_agent_status(orch, "MED_SAFETY", "done")


async def agent_alert_escalation(ctx: AgentContext):
    orch = "SAFETY"
    await ctx.set_agent_status(orch, "ALERT_ESCALATION", "active")

    alerts = []
    pt = ctx.patient
    v = pt.vitals
    labs = pt.labs

    def latest_v(key):
        arr = labs.get(key, [])
        return arr[-1]["v"] if arr else None

    lactate = latest_v("lactate")
    cr = latest_v("creatinine")
    wbc = latest_v("wbc")
    pct = latest_v("procalcitonin")

    if lactate and lactate >= 4.0:
        alerts.append({"level": "CRITICAL", "code": "LACTATE_CRITICAL", "message": f"Lactate {lactate} mmol/L — critical tissue hypoperfusion. SSC 2021: resuscitation escalation + ICU attending NOW.", "intervention": "Fluid bolus 30mL/kg + vasopressor titration + repeat lactate in 1h"})
    elif lactate and lactate >= 2.0:
        alerts.append({"level": "WARNING", "code": "LACTATE_ELEVATED", "message": f"Lactate {lactate} mmol/L — tissue hypoperfusion threshold reached (SSC 2021 §3.1).", "intervention": "Guide resuscitation by serial lactate; target clearance ≥10%"})

    if v.bpSys <= 90:
        alerts.append({"level": "CRITICAL", "code": "HYPOTENSION", "message": f"SBP {v.bpSys} mmHg — vasopressor threshold. Septic shock criteria met.", "intervention": "Norepinephrine 0.01-0.5 mcg/kg/min; MAP target ≥65"})

    if v.spo2 < 90:
        alerts.append({"level": "CRITICAL", "code": "HYPOXEMIA", "message": f"SpO2 {v.spo2}% — severe hypoxemia. ARDS evaluation required (PaO2/FiO2 ratio).", "intervention": "Increase FiO2; consider NIV/intubation; prone positioning if PaO2/FiO2 <150"})

    if cr and cr >= 3.5:
        alerts.append({"level": "WARNING", "code": "AKI_STAGE3", "message": f"Creatinine {cr} mg/dL — KDIGO AKI Stage 3 criteria met.", "intervention": "Nephrology consult; hold nephrotoxins; consider RRT if clinically indicated"})

    if pct and pct >= 10.0:
        alerts.append({"level": "CRITICAL", "code": "PCT_CRITICAL", "message": f"Procalcitonin {pct} ng/mL — severe sepsis/septic shock biomarker threshold.", "intervention": "Escalate antimicrobial coverage; reassess source control"})

    if wbc and wbc > 20.0:
        alerts.append({"level": "WARNING", "code": "LEUKOCYTOSIS", "message": f"WBC {wbc} ×10³/μL — severe leukocytosis. Bacterial sepsis or haematological process.", "intervention": "Blood cultures ×2 if not done; consider haematology if no infectious source"})

    has_abx = any(m.lower() for m in pt.medications if any(x in m.lower() for x in ["cefazolin", "pip-tazo", "piperacillin", "meropenem", "vancomycin"]))
    if lactate and lactate >= 2.0 and not has_abx:
        alerts.append({"level": "CRITICAL", "code": "ANTIBIOTICS_DUE", "message": "Sepsis criteria met but broad-spectrum antimicrobials not confirmed. SSC 2021: administer within 1 hour.", "intervention": "Start pip-tazo ± vancomycin NOW; blood cultures before if possible"})

    if any(a["level"] == "CRITICAL" for a in alerts):
        ctx.alert_level = "CRITICAL"
    elif any(a["level"] == "WARNING" for a in alerts):
        ctx.alert_level = "WARNING"
    else:
        ctx.alert_level = "WATCH"

    ctx.alert_events = alerts
    await ctx.log("ALERT_ESCALATION", f"Alert level: {ctx.alert_level} | {len(alerts)} active alerts", "info")
    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "ALERT_ESCALATION", "data": {"level": ctx.alert_level, "alerts": alerts}})
    await ctx.set_agent_status(orch, "ALERT_ESCALATION", "done")


async def agent_trend_classifier(ctx: AgentContext):
    orch = "TEMPORAL"
    await ctx.set_agent_status(orch, "TREND_CLASSIFIER", "active")
    labs = ctx.patient.labs
    findings = []

    for key, arr in labs.items():
        if not isinstance(arr, list) or len(arr) < 2:
            continue
        vals = [x["v"] for x in arr]
        ref = data.MIMIC_IV["lab_refs"].get(key, {})
        n = len(vals)
        latest_v = vals[-1]
        first_v = vals[0]
        if n >= 3:
            xs = list(range(n))
            xm = sum(xs) / n
            ym = sum(vals) / n
            num = sum((xs[i] - xm) * (vals[i] - ym) for i in range(n))
            den = sum((xs[i] - xm) ** 2 for i in range(n))
            slope = num / den if den else 0
        else:
            slope = vals[-1] - vals[0]

        pct_change = ((latest_v - first_v) / max(abs(first_v), 0.01)) * 100
        direction = "RISING" if slope > 0.05 else "FALLING" if slope < -0.05 else "STABLE"
        is_abnormal = ref and (latest_v > ref.get("high", 999) or latest_v < ref.get("low", 0))

        finding = {"lab": key, "direction": direction, "slope": round(slope, 3), "pct_change": round(pct_change, 1), "latest": latest_v, "first": first_v, "abnormal": is_abnormal}

        if direction == "RISING" and is_abnormal:
            message = f"{key.upper()} rising {first_v}→{latest_v} ({pct_change:+.1f}%)"
            if key == "lactate" and latest_v >= 2.0:
                message += " — SSC 2021 lactate threshold breached"
            elif key == "creatinine" and (latest_v / max(first_v, 0.01)) >= 1.5:
                stage = 1 if (latest_v / first_v) < 2.0 else 2 if (latest_v / first_v) < 3.0 else 3
                message += f" — KDIGO AKI Stage {stage} criteria"
            findings.append({"lab": key, "severity": "critical" if latest_v > ref.get("critical_high", 99999) or latest_v > ref.get("critical", 99999) else "elevated", "message": message, "finding": finding})

    ctx.lab_findings = findings
    ctx.trends = {f["lab"]: f["finding"] for f in findings}
    await ctx.log("TREND_CLASSIFIER", f"Classified {len(findings)} abnormal lab trends", "info")
    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "TREND_CLASSIFIER", "data": findings})
    await ctx.set_agent_status(orch, "TREND_CLASSIFIER", "done")


async def agent_trajectory_predictor(ctx: AgentContext):
    orch = "TEMPORAL"
    await ctx.set_agent_status(orch, "TRAJECTORY_PREDICTOR", "active")
    labs = ctx.patient.labs
    predictions = {}

    for key, arr in labs.items():
        if not isinstance(arr, list) or len(arr) < 2:
            continue
        vals = [x["v"] for x in arr]
        n = len(vals)
        if n >= 2:
            slope = vals[-1] - vals[-2]
            pred_6h = round(vals[-1] + slope * 1, 2)
            pred_12h = round(vals[-1] + slope * 2, 2)
            ref = data.MIMIC_IV["lab_refs"].get(key, {})
            high = ref.get("high", float("inf"))
            crit = ref.get("critical", ref.get("critical_high", float("inf")))
            predictions[key] = {
                "current": vals[-1],
                "pred_6h": pred_6h,
                "pred_12h": pred_12h,
                "trajectory_status": (
                    "CRITICAL_WITHIN_6H" if pred_6h > crit else
                    "WARNING_WITHIN_6H" if pred_6h > high else
                    "NORMALIZING" if pred_12h <= high < vals[-1] else "STABLE"
                ),
            }

    ctx.trajectory = predictions
    critical_labs = [k for k, v in predictions.items() if v["trajectory_status"] == "CRITICAL_WITHIN_6H"]
    if critical_labs:
        await ctx.log("TRAJECTORY_PREDICTOR", f"Projected critical in 6h: {', '.join(critical_labs)}", "warn")
    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "TRAJECTORY_PREDICTOR", "data": predictions})
    await ctx.set_agent_status(orch, "TRAJECTORY_PREDICTOR", "done")


async def agent_semantic_retriever(ctx: AgentContext) -> List[Dict]:
    orch = "EVIDENCE"
    await ctx.set_agent_status(orch, "SEMANTIC_RETRIEVER", "active")

    query_parts = [
        ctx.patient.admitDiag,
        *[f["message"] for f in ctx.lab_findings],
        *ctx.patient.medications,
        *(["acute kidney injury creatinine"] if ctx.outliers else []),
        f"sepsis lactate {ctx.patient.labs.get('lactate', [{}])[-1].get('v', '')} mmol/L",
        "SOFA organ failure",
    ]
    query = ". ".join(str(x) for x in query_parts if x)

    retrieved = []
    if data._guideline_embeddings and await ollama.is_online():
        await ctx.log("SEMANTIC_RETRIEVER", "Computing bge-m3 query embedding...", "info")
        try:
            q_emb = await ollama.embed(query)
            scored = []
            for i, g in enumerate(data.GUIDELINES):
                score = cosine_sim(q_emb, data._guideline_embeddings[i])
                scored.append({**g, "score": round(float(score), 4)})
            scored.sort(key=lambda x: x["score"], reverse=True)
            retrieved = scored[:cfg.TOP_K_GUIDELINES]
            await ctx.log("SEMANTIC_RETRIEVER", f"Semantic retrieval done. Top score: {retrieved[0]['score']}", "info")
        except Exception as e:
            await ctx.log("SEMANTIC_RETRIEVER", f"Embed failed: {e}, using keyword fallback", "warn")

    if not retrieved:
        lower = query.lower()
        retrieved = sorted([{**g, "score": keyword_score(lower, g["keywords"])} for g in data.GUIDELINES], key=lambda x: x["score"], reverse=True)[:cfg.TOP_K_GUIDELINES]
        await ctx.log("SEMANTIC_RETRIEVER", "Using keyword fallback retrieval", "warn")

    ctx.retrieved_guidelines = retrieved
    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "SEMANTIC_RETRIEVER", "data": [{"source": g["source"], "section": g["section"], "score": g.get("score", 0), "text": g["text"]} for g in retrieved]})
    await ctx.set_agent_status(orch, "SEMANTIC_RETRIEVER", "done")
    return retrieved


async def agent_rag_explainer(ctx: AgentContext, guidelines: List[Dict]):
    orch = "EVIDENCE"
    await ctx.set_agent_status(orch, "RAG_EXPLAINER", "active")

    guideline_ctx = "\n\n".join(f"[{g['source']} {g['section']}]: {g['text']}" for g in guidelines)
    findings_str = "; ".join(f["message"] for f in ctx.lab_findings) or "No major lab trends"
    outlier_str = "; ".join(f"⚠ {o['lab'].upper()} value {o['value']} (Z={o['z']}) inconsistent with 72h trend" for o in ctx.outliers) or "None"

    messages = [
        {"role": "system", "content": "You are a clinical evidence agent. Using retrieved guidelines, write 3-4 concise, cited clinical observations relevant to the patient findings. Use [Source §Section] citation format. Be direct and clinical."},
        {"role": "user", "content": f"Patient: {ctx.patient.name}, {ctx.patient.age}{ctx.patient.sex}. {ctx.patient.admitDiag}.\nLab findings: {findings_str}\nOutliers: {outlier_str}\n\nGuidelines:\n{guideline_ctx}\n\nWrite cited observations:"},
    ]

    explanation = ""
    try:
        if await ollama.is_online():
            explanation = await ollama.chat(cfg.RAG_EXPLAIN_MODEL, messages, max_tokens=500)
        else:
            raise RuntimeError("Ollama offline")
    except Exception:
        try:
            explanation = await nim.chat(cfg.FALLBACK_MODEL, messages, ctx.nim_key, max_tokens=500)
        except Exception as e:
            explanation = f"RAG explanation unavailable: {e}"

    ctx.rag_explanation = explanation
    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "RAG_EXPLAINER", "data": {"explanation": explanation}})
    await ctx.set_agent_status(orch, "RAG_EXPLAINER", "done")


async def agent_synthesis(ctx: AgentContext):
    orch = "SYNTHESIS"
    await ctx.set_agent_status(orch, "CHIEF_AGENT", "active")

    pt = ctx.patient
    sofa = ctx.sofa
    news2 = ctx.news2
    outlier_note = ""
    if ctx.outliers:
        outlier_note = "\n".join(
            f"⚠ OUTLIER HELD: {o['lab'].upper()} {o['value']} at {o['timepoint']} (Z={o['z']} vs 72h mean {o['mean']}±{o['std']}) — DIAGNOSIS HELD, REDRAW REQUIRED"
            for o in ctx.outliers
        )

    lab_traj = "\n".join(
        f"{k.upper()}: {' → '.join(str(x['v']) + ((' [OUTLIER]') if x.get('outlier') else '') for x in v)} {data.MIMIC_IV['lab_refs'].get(k, {}).get('unit', '')}"
        for k, v in pt.labs.items() if isinstance(v, list)
    )

    med_conflicts = "\n".join(f"• {c['med']}: {c['conflict']}" for c in ctx.med_conflicts if c["conflict"])
    guideline_ctx = "\n".join(f"[{g['source']} {g['section']}]: {g['text']}" for g in ctx.retrieved_guidelines)
    alert_summary = "\n".join(f"• [{a['level']}] {a['message']}" for a in ctx.alert_events)
    traj_summary = "\n".join(f"• {k.upper()}: pred 6h={v.get('pred_6h', '?')}, 12h={v.get('pred_12h', '?')} [{v.get('trajectory_status', '?')}]" for k, v in ctx.trajectory.items())

    mortality_pct = sofa["mortality_pct"] if sofa else "?"
    sofa_str = f"{sofa['total']}/24 (Resp:{sofa['resp']} Coag:{sofa['coag']} Liver:{sofa['liver']} Cardio:{sofa['cardio']} CNS:{sofa['cns']} Renal:{sofa['renal']}, MIMIC-IV predicted mortality {mortality_pct}%)" if sofa else "N/A"

    prompt = f"""You are the Chief Diagnostic Agent for HC01 — ICU Diagnostic Risk Assistant.
Synthesize all agent outputs below and produce a structured clinical assessment.

PATIENT: {pt.name}, {pt.age}{pt.sex}, Day {pt.daysInICU:.1f} ICU, {pt.weight}kg
ADMISSION: {pt.admitDiag}

VITALS: HR {pt.vitals.hr}bpm | BP {pt.vitals.bpSys}/{pt.vitals.bpDia}mmHg | MAP {pt.vitals.map}mmHg | RR {pt.vitals.rr}/min | SpO2 {pt.vitals.spo2}% | Temp {pt.vitals.temp}°C | GCS {pt.vitals.gcs}/15 | FiO2 {pt.vitals.fio2}

LAB TRAJECTORIES (72h):
{lab_traj}

SOFA: {sofa_str}
NEWS2: {news2['total']}/20 — {news2['level']} RISK

TEMPORAL LAB FLAGS:
{chr(10).join('• ' + f['message'] for f in ctx.lab_findings) or 'None'}

TRAJECTORY PREDICTIONS:
{traj_summary or 'Insufficient data'}

{outlier_note}

ACTIVE ALERTS ({ctx.alert_level}):
{alert_summary or 'No active alerts'}

MEDICATION SAFETY:
{med_conflicts or 'No conflicts identified'}

RAG EVIDENCE:
{guideline_ctx}

RAG AGENT SYNTHESIS:
{ctx.rag_explanation or 'Not available'}

PARSED NOTE FINDINGS:
{json.dumps(ctx.parsed_notes)[:600] if ctx.parsed_notes else 'N/A'}

DATASETS: MIMIC-IV v2.2 (73,141 ICU stays), eICU-CRD (200,859 encounters), PhysioNet Sepsis 2019

Produce EXACTLY this structure:

CLINICAL ASSESSMENT:
[3 sentences: current picture, trajectory, immediate risk]

DIFFERENTIAL DIAGNOSIS:
1. [Diagnosis] — [XX]% confidence — [evidence citing guideline or dataset]
2. [Diagnosis] — [XX]% confidence — [evidence]
3. [Diagnosis] — [XX]% confidence — [evidence]

KEY CONCERNS:
• [Concern] [Guideline citation in brackets]
• [Concern] [Guideline citation]
• [Concern] [Guideline citation]

RECOMMENDED ACTIONS:
• [Action] — [Urgency: IMMEDIATE/URGENT/ROUTINE]
• [Action] — [Urgency]
• [Action] — [Urgency]

SHIFT HANDOVER BRIEF (30 seconds):
[2-3 plain English sentences for incoming doctor]

---
Be precise. Cite guidelines and MIMIC-IV stats. Flag held diagnoses explicitly. Reference dataset context where relevant."""

    messages = [{"role": "user", "content": prompt}]
    full_text = ""

    async def on_chunk(delta: str, accumulated: str):
        nonlocal full_text
        full_text = accumulated
        await ctx.send({"type": "stream_chunk", "orchestrator": orch, "agent": "CHIEF_AGENT", "content": delta})

    try:
        full_text = await nim.chat(cfg.CHIEF_MODEL, messages, ctx.nim_key, max_tokens=1800, on_chunk=on_chunk)
    except Exception as e:
        await ctx.log("CHIEF_AGENT", f"Nemotron failed: {e}. Using fallback.", "warn")
        try:
            full_text = await nim.chat(cfg.FALLBACK_MODEL, messages, ctx.nim_key, max_tokens=1500, on_chunk=on_chunk)
        except Exception as e2:
            full_text = f"Chief Agent unavailable: {e2}"

    ctx.synthesis = full_text
    hm = re.search(r'SHIFT HANDOVER BRIEF[^:]*:\s*([\s\S]+?)(?:\n\n|\n---|\n#|$)', full_text, re.I)
    ctx.handover = hm.group(1).strip() if hm else ""

    await ctx.send({"type": "agent_result", "orchestrator": orch, "agent": "CHIEF_AGENT", "data": {"synthesis": full_text, "handover": ctx.handover}})
    await ctx.set_agent_status(orch, "CHIEF_AGENT", "done")


async def master_orchestrate(ctx: AgentContext):
    start = time.time()
    pt = ctx.patient
    await ctx.log("HC01", f"=== PIPELINE START: {pt.name} {pt.age}{pt.sex} ===", "info")
    await ctx.send({"type": "orchestrator_status", "orchestrator": "MASTER", "status": "active"})

    ctx.sofa = calc_sofa(pt)
    ctx.news2 = calc_news2(pt)
    await ctx.send({"type": "agent_result", "orchestrator": "SCORING", "agent": "SOFA_NEWS2", "data": {"sofa": ctx.sofa, "news2": ctx.news2}})

    await ctx.log("HC01", "Wave 1: Launching 5 agents in parallel...", "info")
    for orch in ["CLINICAL", "SAFETY", "TEMPORAL"]:
        await ctx.set_orch_status(orch, "active")

    await asyncio.gather(
        agent_note_parser(ctx),
        agent_outlier_detector(ctx),
        agent_med_safety(ctx),
        agent_alert_escalation(ctx),
        agent_trend_classifier(ctx),
        agent_trajectory_predictor(ctx),
    )

    for orch in ["CLINICAL", "SAFETY", "TEMPORAL"]:
        await ctx.set_orch_status(orch, "done")
    await ctx.log("HC01", "Wave 1 complete.", "info")

    await ctx.set_orch_status("EVIDENCE", "active")
    guidelines = await agent_semantic_retriever(ctx)
    await agent_rag_explainer(ctx, guidelines)
    await ctx.set_orch_status("EVIDENCE", "done")

    await ctx.set_orch_status("SYNTHESIS", "active")
    await agent_synthesis(ctx)
    await ctx.set_orch_status("SYNTHESIS", "done")

    elapsed = round((time.time() - start) * 1000)
    await ctx.send({"type": "pipeline_complete", "duration_ms": elapsed, "alert_level": ctx.alert_level, "sofa_total": ctx.sofa["total"], "news2_total": ctx.news2["total"]})
    await ctx.log("HC01", f"=== PIPELINE COMPLETE in {elapsed}ms ===", "info")


async def preembed_guidelines():
    if not await ollama.is_online():
        return
    models = await ollama.available_models()
    if not any("bge-m3" in m for m in models):
        return
    embeds = []
    for g in data.GUIDELINES:
        text = f"{g['source']} {g['section']}: {g['text']}"
        emb = await ollama.embed(text)
        embeds.append(emb)
    data._guideline_embeddings = embeds
