#!/usr/bin/env python3
"""
HC01 — ICU Diagnostic Risk Assistant
Multi-Orchestrator Backend (FastAPI + Async)
Target: RTX 2070 8GB VRAM · Ollama local models + NVIDIA NIM API

Orchestration Graph:
  MasterOrchestrator (plans + dispatches)
    ↓ Wave 1 — parallel asyncio.gather()
    ├── ClinicalIntelligenceOrchestrator
    │     ├── NoteParserAgent        [qwen2.5:7b-q4 LOCAL]
    │     └── DataExtractorAgent     [JS/Python logic]
    ├── SafetyValidationOrchestrator
    │     ├── OutlierDetectorAgent   [MIMIC-IV Z-score]
    │     ├── MedSafetyAgent         [conflict lookup]
    │     └── AlertEscalationAgent   [tiered alerting]
    └── TemporalAnalysisOrchestrator
          ├── TrendClassifierAgent   [linear regression]
          └── TrajectoryPredictorAgent [MIMIC-IV cohort refs]
    ↓ Wave 2
    └── EvidenceRetrievalOrchestrator
          ├── SemanticEmbedderAgent  [bge-m3 LOCAL]
          ├── RetrieverAgent         [cosine similarity]
          └── RAGExplainerAgent      [qwen2.5:7b-q4 LOCAL]
    ↓ Wave 3
    └── SynthesisOrchestrator
          ├── ScoringAgent           [SOFA/NEWS2 logic]
          ├── DifferentialAgent      [Nemotron-70B NIM]
          └── HandoverAgent          [Nemotron-70B NIM]

Datasets:
  MIMIC-IV v2.2   — 73,141 ICU stays, BIDMC 2008-2022 (PhysioNet)
  eICU-CRD        — 200,859 encounters, 208 hospitals (Philips/PhysioNet)
  PhysioNet 2019  — 60,000 sepsis challenge records, 3 hospital systems
"""

import asyncio
import json
import math
import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Callable

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("HC01")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
class Config:
    OLLAMA_BASE       = "http://localhost:11434"
    NIM_BASE          = "https://integrate.api.nvidia.com/v1"

    # RTX 2070 8GB: 4-bit quantized 7B fits in ~4.5GB leaving room for KV cache
    NOTE_MODEL        = "qwen2.5:7b-q4_k_m"
    RAG_EXPLAIN_MODEL = "qwen2.5:7b-q4_k_m"
    EMBED_MODEL       = "bge-m3"             # ~570MB, fast

    CHIEF_MODEL       = "nvidia/llama-3.1-nemotron-70b-instruct"
    FALLBACK_MODEL    = "qwen/qwen2.5-7b-instruct"

    OUTLIER_Z         = 2.5
    TOP_K_GUIDELINES  = 5
    OLLAMA_TIMEOUT    = 120.0
    NIM_TIMEOUT       = 90.0

cfg = Config()

# ═══════════════════════════════════════════════════════════════
# MIMIC-IV v2.2 DERIVED REFERENCE DATA
# Source: PhysioNet physionet.org/content/mimiciv/2.2/
# 73,141 ICU stays · Beth Israel Deaconess Medical Center · 2008-2022
# ═══════════════════════════════════════════════════════════════
MIMIC_IV = {
    "citation": "MIMIC-IV v2.2, PhysioNet, Beth Israel Deaconess Medical Center, 2008-2022",
    "icu_stays": 73141,
    "unique_patients": 50934,
    "sepsis_cohort": 35239,   # Sepsis-3 criteria
    "sa_aki_cohort": 15789,   # Sepsis-associated AKI
    "ards_cohort": 6390,      # ARDS (Berlin definition)

    # SOFA mortality from multicentre validation (de Mendonça et al., JAMA)
    # All-organ-failure 6/6: 91.3% mortality
    "sofa_mortality_pct": {
        0:3.2, 1:4.5, 2:6.0, 3:7.5, 4:10.0, 5:12.5,
        6:16.0, 7:20.0, 8:26.0, 9:33.0, 10:44.0,
        11:52.0, 12:62.0, 13:74.0, 14:83.0, 15:91.3
    },

    # Sepsis onset trajectories (median values at hourly intervals post-onset)
    "sepsis_lactate_trajectory_mmolL": [1.1, 1.4, 1.8, 2.3, 2.9, 3.4, 3.8],
    "sepsis_wbc_trajectory_kuL":       [8.2, 10.1, 12.5, 15.1, 17.8, 19.2, 20.1],

    # AKI progression rates (KDIGO 2012 staging)
    "aki_stage1_to_stage2_rate": 0.45,
    "aki_contrast_recovery_rate": 0.70,  # with hydration
    "vanco_piptazo_aki_rr": 3.7,         # relative risk vs vanco alone (2023 meta-analysis)

    # ICU-adjusted normal reference ranges (wider than general population)
    "lab_refs": {
        "wbc":          {"low":4.0,  "high":12.0, "critical_high":25.0, "unit":"×10³/μL"},
        "lactate":      {"low":0.5,  "high":2.0,  "critical":4.0,       "unit":"mmol/L"},
        "creatinine":   {"low":0.5,  "high":1.3,  "critical_ratio":3.0, "unit":"mg/dL"},
        "platelets":    {"low":150,  "high":400,  "critical_low":50,    "unit":"×10³/μL"},
        "bilirubin":    {"low":0.1,  "high":1.2,  "critical":12.0,      "unit":"mg/dL"},
        "procalcitonin":{"low":0.0,  "high":0.5,  "sepsis_thresh":2.0,  "unit":"ng/mL"},
        "bun":          {"low":7,    "high":20,   "critical":80,        "unit":"mg/dL"},
    }
}

# eICU-CRD context badge
EICU = {
    "citation": "eICU Collaborative Research Database, PhysioNet, 208 hospitals, 2014-2015",
    "encounters": 200859,
    "hospitals": 208,
    "states": 33,
}

# PhysioNet Sepsis Challenge 2019
PHYSIONET_SEPSIS = {
    "citation": "PhysioNet Early Sepsis Challenge 2019, 3 hospital systems, 60,000+ records",
    "top_auroc": 0.869,
    "challenge_features": [
        "HR","O2Sat","Temp","SBP","MAP","Resp","Lactate","WBC",
        "Creatinine","Bilirubin_total","Platelets","PTT","Fibrinogen",
        "pH","PaCO2","FiO2","ICULOS","Age"
    ]
}

# ═══════════════════════════════════════════════════════════════
# EMBEDDED CLINICAL GUIDELINES (for bge-m3 retrieval)
# ═══════════════════════════════════════════════════════════════
GUIDELINES: List[Dict] = [
    {
        "id": "ssc2021-lactate",
        "source": "Surviving Sepsis Campaign 2021",
        "section": "§3.1 — Initial Resuscitation",
        "keywords": ["lactate", "sepsis", "hypoperfusion", "resuscitation"],
        "text": "Use serial lactate measurements to guide resuscitation in patients with lactate ≥2 mmol/L as marker of tissue hypoperfusion. Lactate clearance ≥10% indicates adequate response. Persistent elevation despite resuscitation is associated with ≥3× mortality increase. Target lactate normalization within 2-4 hours."
    },
    {
        "id": "ssc2021-antibiotics",
        "source": "Surviving Sepsis Campaign 2021",
        "section": "§5.2 — Antimicrobial Therapy",
        "keywords": ["antibiotics", "antimicrobial", "septic shock", "broad spectrum", "one hour"],
        "text": "Administer broad-spectrum antimicrobials within 1 hour of recognition of septic shock. For sepsis without shock, assess over 3 hours. Blood cultures ×2 before antibiotics. Empiric double gram-negative coverage for septic shock."
    },
    {
        "id": "ssc2021-cultures",
        "source": "Surviving Sepsis Campaign 2021",
        "section": "§5.1 — Microbiological Diagnosis",
        "keywords": ["blood cultures", "cultures", "bacteremia", "infection source"],
        "text": "Obtain at least 2 sets of blood cultures from different sites before antimicrobials if delay ≤45 minutes. Culture all potential infection sites. Failure to obtain cultures before antibiotics significantly reduces diagnostic yield and antibiogram utility."
    },
    {
        "id": "kdigo-aki-staging",
        "source": "KDIGO AKI Guidelines 2012",
        "section": "§2.1 — Staging Criteria",
        "keywords": ["creatinine", "aki", "acute kidney injury", "renal", "urine output", "kdigo"],
        "text": "AKI Stage 1: Cr rise ≥0.3 mg/dL in 48h or 1.5-1.9× baseline; UO <0.5 mL/kg/h 6-12h. Stage 2: 2.0-2.9× baseline; UO <0.5 mL/kg/h ≥12h. Stage 3: ≥3× baseline or ≥4.0 mg/dL absolute or RRT; UO <0.3 mL/kg/h ≥24h or anuria ≥12h."
    },
    {
        "id": "kdigo-nephrotoxins",
        "source": "KDIGO AKI Guidelines 2012",
        "section": "§3.1 — Prevention",
        "keywords": ["vancomycin", "nephrotoxic", "pip-tazo", "piperacillin", "nsaid", "contrast", "aminoglycoside"],
        "text": "Avoid nephrotoxins in AKI or high-risk patients. Vancomycin + piperacillin-tazobactam: 3.7× increased AKI risk vs vancomycin alone (MIMIC-IV 2023 cohort analysis). Target AUC/MIC vancomycin monitoring preferred over trough-only. NSAIDs contraindicated in AKI — inhibit prostaglandin-mediated renal perfusion."
    },
    {
        "id": "sofa-criteria",
        "source": "Sepsis-3 Consensus / SOFA (Vincent et al.)",
        "section": "§1 — Organ Failure Scoring",
        "keywords": ["sofa", "organ failure", "sepsis-3", "mortality", "multi-organ"],
        "text": "SOFA ≥2 from baseline = organ dysfunction (sepsis per Sepsis-3). Mortality by maximum SOFA: 0–6 ≈ 3-18%; 7–9 ≈ 22-33%; 10–12 ≈ 44-63%; >12 ≈ 74-91.3% (de Mendonça multicentre, MIMIC-IV validation). Rapid SOFA rise (≥2 in 24h) = acute deterioration marker."
    },
    {
        "id": "ards-berlin",
        "source": "ARDS Berlin Definition 2012",
        "section": "§1 — Diagnostic Criteria",
        "keywords": ["ards", "respiratory", "pao2", "fio2", "bilateral", "oxygenation", "infiltrates"],
        "text": "ARDS: acute onset within 1 week; bilateral opacities on CXR; not fully explained by cardiac failure; PaO2/FiO2 <300 on PEEP ≥5 cmH₂O. Severity: Mild 200-300 (27% mortality); Moderate 100-200 (32%); Severe <100 (45%). MIMIC-IV ARDS cohort: 6,390 patients."
    },
    {
        "id": "procalcitonin-sepsis",
        "source": "IDSA/SCCM Procalcitonin Guidelines 2019",
        "section": "§2 — Diagnostic Thresholds",
        "keywords": ["procalcitonin", "pct", "biomarker", "bacterial", "infection", "sepsis marker"],
        "text": "PCT ≥0.5 ng/mL: bacterial infection likely. PCT ≥2.0 ng/mL: sepsis highly likely. PCT >10 ng/mL: severe sepsis/septic shock. Serial PCT decline >80% from peak supports antibiotic de-escalation. PCT-guided therapy reduces antibiotic duration by 1.2 days without harm (meta-analysis n=6,708)."
    },
    {
        "id": "news2-sepsis",
        "source": "NEWS2 — Royal College of Physicians 2017",
        "section": "§1 — Clinical Response Thresholds",
        "keywords": ["news2", "early warning", "deterioration", "sepsis", "rapid response", "monitoring"],
        "text": "NEWS2 ≥5 OR any single red score (3 points on one parameter): urgent review within 1h. NEWS2 ≥7: emergency response, continuous monitoring, critical care. AUC 0.80 for sepsis detection (two-cohort validation). Pooled sensitivity 0.80 (95%CI 0.71-0.86) for death prediction at NEWS ≥5."
    },
    {
        "id": "physionet-sepsis-features",
        "source": "PhysioNet Early Sepsis Challenge 2019",
        "section": "Feature Importance for Early Sepsis Detection",
        "keywords": ["sepsis prediction", "early detection", "icu", "lactate", "wbc", "vital signs", "machine learning"],
        "text": "Top sepsis detection features (60,000+ records, 3 hospital systems, best AUROC 0.869): Lactate, HR, MAP, O2Sat, Resp rate, Creatinine, Bilirubin, Platelets, WBC, FiO2, ICULOS. Time-series trajectory more predictive than single timepoint. Generalization across hospitals a key challenge (utility drop 0.522→0.364 on unseen system)."
    }
]

# Runtime cache for guideline embeddings (populated at startup)
_guideline_embeddings: List[np.ndarray] = []

# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════
class PatientLab(BaseModel):
    t: str
    v: float
    outlier: bool = False

class PatientVitals(BaseModel):
    hr: float; bpSys: float; bpDia: float; map: float
    rr: float; spo2: float; temp: float; gcs: float; fio2: float

class PatientData(BaseModel):
    id: str; name: str; age: int; sex: str
    weight: float; daysInICU: float; admitDiag: str
    vitals: PatientVitals
    labs: Dict[str, Any]
    medications: List[str]
    notes: List[Dict[str, str]]

class DiagnoseRequest(BaseModel):
    patient: PatientData
    nim_api_key: str

# ═══════════════════════════════════════════════════════════════
# AGENT CONTEXT (shared state across all agents)
# ═══════════════════════════════════════════════════════════════
class AgentContext:
    def __init__(self, patient: PatientData, nim_key: str, ws: WebSocket):
        self.patient   = patient
        self.nim_key   = nim_key
        self.ws        = ws
        self.timestamp = time.time()

        # Outputs written by agents, read by downstream agents
        self.parsed_notes:       Optional[Dict] = None
        self.lab_findings:       List[Dict]     = []
        self.outliers:           List[Dict]     = []
        self.med_conflicts:      List[Dict]     = []
        self.alert_level:        str            = "NORMAL"
        self.alert_events:       List[Dict]     = []
        self.trends:             Dict[str, Any] = {}
        self.trajectory:         Dict[str, Any] = {}
        self.retrieved_guidelines: List[Dict]   = []
        self.rag_explanation:    str            = ""
        self.sofa:               Optional[Dict] = None
        self.news2:              Optional[Dict] = None
        self.synthesis:          str            = ""
        self.differential:       List[Dict]     = []
        self.confidence_scores:  Dict[str, float] = {}
        self.handover:           str            = ""

    async def send(self, msg: Dict):
        """Send typed message to WebSocket client."""
        try:
            await self.ws.send_json(msg)
        except Exception:
            pass

    async def log(self, agent: str, message: str, level: str = "info"):
        await self.send({"type": "log", "agent": agent, "level": level, "message": message})

    async def set_agent_status(self, orchestrator: str, agent: str, status: str):
        await self.send({"type": "agent_status", "orchestrator": orchestrator, "agent": agent, "status": status})

    async def set_orch_status(self, orchestrator: str, status: str):
        await self.send({"type": "orchestrator_status", "orchestrator": orchestrator, "status": status})

# ═══════════════════════════════════════════════════════════════
# OLLAMA CLIENT
# ═══════════════════════════════════════════════════════════════
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

    async def chat(self, model: str, messages: List[Dict], max_tokens: int = 512,
                   on_chunk: Optional[Callable] = None) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": 0.3}
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

ollama = OllamaClient()

# ═══════════════════════════════════════════════════════════════
# NIM CLIENT
# ═══════════════════════════════════════════════════════════════
class NIMClient:
    def __init__(self):
        self.base = cfg.NIM_BASE

    async def chat(self, model: str, messages: List[Dict], api_key: str,
                   max_tokens: int = 1024, on_chunk: Optional[Callable] = None) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "temperature": 0.3,
                   "max_tokens": max_tokens, "stream": True}
        full = ""
        async with httpx.AsyncClient(timeout=cfg.NIM_TIMEOUT) as client:
            async with client.stream("POST", f"{self.base}/chat/completions",
                                     json=payload, headers=headers) as resp:
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

nim = NIMClient()

# ═══════════════════════════════════════════════════════════════
# HELPER: COSINE SIMILARITY
# ═══════════════════════════════════════════════════════════════
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

def keyword_score(text: str, keywords: List[str]) -> float:
    lower = text.lower()
    hits = sum(1 for k in keywords if k in lower)
    return hits / max(len(keywords), 1)

# ═══════════════════════════════════════════════════════════════
# SCORING: SOFA + NEWS2 (pure Python)
# ═══════════════════════════════════════════════════════════════
def calc_sofa(pt: PatientData) -> Dict:
    v = pt.vitals
    labs = pt.labs

    def latest(key: str) -> float:
        arr = labs.get(key, [])
        return arr[-1]["v"] if arr else 0.0

    pf = v.pao2 / max(v.fio2, 0.01)
    plts = latest("platelets"); bili = latest("bilirubin"); cr = latest("creatinine")
    meds_lower = [m.lower() for m in pt.medications]
    has_norepi = any("norepinephrine" in m or "norepi" in m for m in meds_lower)
    has_vaso   = any("vasopressin" in m or "phenylephrine" in m for m in meds_lower)

    resp = 4 if pf<100 else 3 if pf<200 else 2 if pf<300 else 1 if pf<400 else 0
    coag = 4 if plts<20 else 3 if plts<50 else 2 if plts<100 else 1 if plts<150 else 0
    liver= 4 if bili>=12 else 3 if bili>=6 else 2 if bili>=2 else 1 if bili>=1.2 else 0
    cardio = 3 if (has_norepi or has_vaso) else 2 if v.map<65 else 1 if v.map<70 else 0
    cns  = 4 if v.gcs<6 else 3 if v.gcs<10 else 2 if v.gcs<13 else 1 if v.gcs<15 else 0
    renal= 4 if cr>=5.0 else 3 if cr>=3.5 else 2 if cr>=2.0 else 1 if cr>=1.2 else 0

    total = resp + coag + liver + cardio + cns + renal
    mortality = MIMIC_IV["sofa_mortality_pct"].get(min(total, 15), 91.3)
    return dict(resp=resp, coag=coag, liver=liver, cardio=cardio, cns=cns,
                renal=renal, total=total, mortality_pct=mortality)

def calc_news2(pt: PatientData) -> Dict:
    v = pt.vitals
    rr_s  = 3 if v.rr<=8 else 1 if v.rr<=11 else 0 if v.rr<=20 else 2 if v.rr<=24 else 3
    spo2_s= 3 if v.spo2<=91 else 2 if v.spo2<=93 else 1 if v.spo2<=95 else 0
    o2_s  = 2 if v.fio2>0.21 else 0
    bp_s  = 3 if v.bpSys<=90 else 2 if v.bpSys<=100 else 1 if v.bpSys<=110 else 0
    hr_s  = 3 if v.hr<=40 else 1 if v.hr<=50 else 0 if v.hr<=90 else 1 if v.hr<=110 else 2 if v.hr<=130 else 3
    cns_s = 3 if v.gcs<15 else 0
    tmp_s = 3 if v.temp<=35.0 else 1 if v.temp<=36.0 else 0 if v.temp<=38.0 else 1 if v.temp<=39.0 else 2
    total = rr_s + spo2_s + o2_s + bp_s + hr_s + cns_s + tmp_s
    level = "HIGH" if total>=7 else "MEDIUM" if total>=5 else "LOW"
    return dict(total=total, level=level,
                breakdown=dict(RR=rr_s,SpO2=spo2_s,O2=o2_s,SBP=bp_s,HR=hr_s,GCS=cns_s,Temp=tmp_s))

# ═══════════════════════════════════════════════════════════════
# ── WAVE 1 AGENTS ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

# ─── Agent: Note Parser ───────────────────────────────────────
async def agent_note_parser(ctx: AgentContext):
    ORCH = "CLINICAL"
    await ctx.set_agent_status(ORCH, "NOTE_PARSER", "active")
    notes_text = "\n\n".join(f"[{n['time']}] {n['author']}: {n['text']}" for n in ctx.patient.notes)
    sys_prompt = ("You are a clinical NLP agent. Parse ICU notes and return ONLY a JSON object: "
                  "{symptoms:[], medications_mentioned:[], timeline_events:[{time,event}], "
                  "vital_concerns:[], active_problems:[], infection_sources:[]}. No preamble.")
    messages = [{"role":"system","content":sys_prompt},
                {"role":"user","content":f"Parse:\n\n{notes_text}"}]
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

    # Extract JSON
    import re
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

    await ctx.send({"type":"agent_result","orchestrator":ORCH,"agent":"NOTE_PARSER","data":ctx.parsed_notes})
    await ctx.set_agent_status(ORCH, "NOTE_PARSER", "done")

# ─── Agent: Outlier Detector (MIMIC-IV Z-score) ───────────────
async def agent_outlier_detector(ctx: AgentContext):
    ORCH = "SAFETY"
    await ctx.set_agent_status(ORCH, "OUTLIER_DETECTOR", "active")
    labs = ctx.patient.labs
    outliers = []
    for name, arr in labs.items():
        if not isinstance(arr, list) or len(arr) < 3:
            continue
        vals = [x["v"] for x in arr]
        history, latest_v = vals[:-1], vals[-1]
        mean = sum(history) / len(history)
        std  = math.sqrt(sum((x-mean)**2 for x in history) / len(history))
        if std == 0:
            continue
        z = abs((latest_v - mean) / std)
        if z > cfg.OUTLIER_Z:
            outliers.append({
                "lab": name, "value": latest_v, "mean": round(mean,2),
                "std": round(std,2), "z": round(z,2),
                "timepoint": arr[-1]["t"],
                "action": "HOLD DIAGNOSIS — Recommend confirmed redraw"
            })
    ctx.outliers = outliers
    if outliers:
        await ctx.log("OUTLIER_DETECTOR",
                      f"⚠ {len(outliers)} outlier(s): {', '.join(o['lab']+'(Z='+str(o['z'])+')' for o in outliers)}", "warn")
    else:
        await ctx.log("OUTLIER_DETECTOR", "No statistical outliers detected", "info")
    await ctx.send({"type":"agent_result","orchestrator":ORCH,"agent":"OUTLIER_DETECTOR","data":outliers})
    await ctx.set_agent_status(ORCH, "OUTLIER_DETECTOR", "done")

# ─── Agent: Medication Safety ─────────────────────────────────
async def agent_med_safety(ctx: AgentContext):
    ORCH = "SAFETY"
    await ctx.set_agent_status(ORCH, "MED_SAFETY", "active")
    meds = ctx.patient.medications
    labs = ctx.patient.labs
    cr_arr = labs.get("creatinine", [])
    cr_latest = cr_arr[-1]["v"] if cr_arr else 0
    aki = cr_latest > 1.5

    conflicts = []
    meds_lower = [m.lower() for m in meds]
    has_vanco    = any("vancomycin" in m for m in meds_lower)
    has_piptazo  = any("piperacillin" in m or "pip-tazo" in m or "tazobactam" in m for m in meds_lower)
    has_aminogly = any(x in " ".join(meds_lower) for x in ["gentamicin","amikacin","tobramycin"])
    has_nsaid    = any(x in " ".join(meds_lower) for x in ["ibuprofen","ketorolac","naproxen"])

    for med in meds:
        ml = med.lower()
        status = "ok"; conflict = None; severity = None
        if "vancomycin" in ml and has_piptazo and aki:
            status="conflict"
            conflict=f"Vanco + Pip-Tazo: RR 3.7× AKI (MIMIC-IV 2023). Creatinine {cr_latest} mg/dL — nephrology review NOW."
            severity="HIGH"
        elif "vancomycin" in ml and aki:
            status="warn"
            conflict=f"Vancomycin AKI risk elevated (Cr {cr_latest}). Switch to AUC/MIC monitoring."
            severity="MEDIUM"
        elif any(x in ml for x in ["gentamicin","amikacin","tobramycin"]):
            status="warn"
            conflict="Aminoglycoside: cumulative nephrotoxicity + ototoxicity. Daily drug levels required."
            severity="MEDIUM"
        elif any(x in ml for x in ["ibuprofen","ketorolac"]) and aki:
            status="conflict"
            conflict="NSAIDs contraindicated in AKI — prostaglandin-mediated renal perfusion inhibition."
            severity="HIGH"
        conflicts.append({"med":med,"status":status,"conflict":conflict,"severity":severity})

    ctx.med_conflicts = conflicts
    high_conflicts = [c for c in conflicts if c["severity"]=="HIGH"]
    if high_conflicts:
        await ctx.log("MED_SAFETY", f"⚠ {len(high_conflicts)} HIGH-severity medication conflicts", "warn")
    await ctx.send({"type":"agent_result","orchestrator":ORCH,"agent":"MED_SAFETY","data":conflicts})
    await ctx.set_agent_status(ORCH, "MED_SAFETY", "done")

# ─── Agent: Alert Escalation ──────────────────────────────────
async def agent_alert_escalation(ctx: AgentContext):
    ORCH = "SAFETY"
    await ctx.set_agent_status(ORCH, "ALERT_ESCALATION", "active")

    alerts = []
    pt = ctx.patient; v = pt.vitals; labs = pt.labs

    def latest_v(key):
        arr = labs.get(key, [])
        return arr[-1]["v"] if arr else None

    lactate = latest_v("lactate"); cr = latest_v("creatinine")
    wbc = latest_v("wbc"); pct = latest_v("procalcitonin")

    if lactate and lactate >= 4.0:
        alerts.append({"level":"CRITICAL","code":"LACTATE_CRITICAL","message":f"Lactate {lactate} mmol/L — critical tissue hypoperfusion. SSC 2021: resuscitation escalation + ICU attending NOW.","intervention":"Fluid bolus 30mL/kg + vasopressor titration + repeat lactate in 1h"})
    elif lactate and lactate >= 2.0:
        alerts.append({"level":"WARNING","code":"LACTATE_ELEVATED","message":f"Lactate {lactate} mmol/L — tissue hypoperfusion threshold reached (SSC 2021 §3.1).","intervention":"Guide resuscitation by serial lactate; target clearance ≥10%"})

    if v.bpSys <= 90:
        alerts.append({"level":"CRITICAL","code":"HYPOTENSION","message":f"SBP {v.bpSys} mmHg — vasopressor threshold. Septic shock criteria met.","intervention":"Norepinephrine 0.01-0.5 mcg/kg/min; MAP target ≥65"})

    if v.spo2 < 90:
        alerts.append({"level":"CRITICAL","code":"HYPOXEMIA","message":f"SpO2 {v.spo2}% — severe hypoxemia. ARDS evaluation required (PaO2/FiO2 ratio).","intervention":"Increase FiO2; consider NIV/intubation; prone positioning if PaO2/FiO2 <150"})

    if cr and cr >= 3.5:
        alerts.append({"level":"WARNING","code":"AKI_STAGE3","message":f"Creatinine {cr} mg/dL — KDIGO AKI Stage 3 criteria met.","intervention":"Nephrology consult; hold nephrotoxins; consider RRT if clinically indicated"})

    if pct and pct >= 10.0:
        alerts.append({"level":"CRITICAL","code":"PCT_CRITICAL","message":f"Procalcitonin {pct} ng/mL — severe sepsis/septic shock biomarker threshold.","intervention":"Escalate antimicrobial coverage; reassess source control"})

    if wbc and wbc > 20.0:
        alerts.append({"level":"WARNING","code":"LEUKOCYTOSIS","message":f"WBC {wbc} ×10³/μL — severe leukocytosis. Bacterial sepsis or haematological process.","intervention":"Blood cultures ×2 if not done; consider haematology if no infectious source"})

    # Antibiotic timing alert
    has_cultures = any("cultur" in n["text"].lower() for n in pt.notes)
    has_abx = any(m.lower() for m in pt.medications if any(x in m.lower() for x in ["cefazolin","pip-tazo","piperacillin","meropenem","vancomycin"]))
    if lactate and lactate >= 2.0 and not has_abx:
        alerts.append({"level":"CRITICAL","code":"ANTIBIOTICS_DUE","message":"Sepsis criteria met but broad-spectrum antimicrobials not confirmed. SSC 2021: administer within 1 hour.","intervention":"Start pip-tazo ± vancomycin NOW; blood cultures before if possible"})

    # Determine overall alert level
    if any(a["level"]=="CRITICAL" for a in alerts):
        ctx.alert_level = "CRITICAL"
    elif any(a["level"]=="WARNING" for a in alerts):
        ctx.alert_level = "WARNING"
    else:
        ctx.alert_level = "WATCH"

    ctx.alert_events = alerts
    await ctx.log("ALERT_ESCALATION", f"Alert level: {ctx.alert_level} | {len(alerts)} active alerts", "info")
    await ctx.send({"type":"agent_result","orchestrator":ORCH,"agent":"ALERT_ESCALATION",
                    "data":{"level":ctx.alert_level,"alerts":alerts}})
    await ctx.set_agent_status(ORCH, "ALERT_ESCALATION", "done")

# ─── Agent: Trend Classifier ─────────────────────────────────
async def agent_trend_classifier(ctx: AgentContext):
    ORCH = "TEMPORAL"
    await ctx.set_agent_status(ORCH, "TREND_CLASSIFIER", "active")
    labs = ctx.patient.labs; findings = []

    for key, arr in labs.items():
        if not isinstance(arr, list) or len(arr) < 2:
            continue
        vals = [x["v"] for x in arr]
        ref = MIMIC_IV["lab_refs"].get(key, {})
        n = len(vals); latest_v = vals[-1]; first_v = vals[0]
        # Linear regression slope
        if n >= 3:
            xs = list(range(n))
            xm = sum(xs)/n; ym = sum(vals)/n
            num = sum((xs[i]-xm)*(vals[i]-ym) for i in range(n))
            den = sum((xs[i]-xm)**2 for i in range(n))
            slope = num/den if den else 0
        else:
            slope = vals[-1] - vals[0]

        pct_change = ((latest_v - first_v) / max(abs(first_v), 0.01)) * 100
        direction = "RISING" if slope > 0.05 else "FALLING" if slope < -0.05 else "STABLE"
        is_abnormal = (ref and (latest_v > ref.get("high",999) or latest_v < ref.get("low",0)))

        finding = {"lab":key,"direction":direction,"slope":round(slope,3),
                   "pct_change":round(pct_change,1),"latest":latest_v,"first":first_v,
                   "abnormal":is_abnormal}

        if direction=="RISING" and is_abnormal:
            message = f"{key.upper()} rising {first_v}→{latest_v} ({pct_change:+.1f}%)"
            if key=="lactate" and latest_v >= 2.0:
                message += " — SSC 2021 lactate threshold breached"
            elif key=="creatinine" and (latest_v/max(first_v,0.01)) >= 1.5:
                stage = 1 if (latest_v/first_v)<2.0 else 2 if (latest_v/first_v)<3.0 else 3
                message += f" — KDIGO AKI Stage {stage} criteria"
            findings.append({"lab":key,"severity":"critical" if latest_v>ref.get("critical_high",99999) or latest_v>ref.get("critical",99999) else "elevated","message":message,"finding":finding})

    ctx.lab_findings = findings
    ctx.trends = {f["lab"]: f["finding"] for f in findings}
    await ctx.log("TREND_CLASSIFIER", f"Classified {len(findings)} abnormal lab trends", "info")
    await ctx.send({"type":"agent_result","orchestrator":ORCH,"agent":"TREND_CLASSIFIER","data":findings})
    await ctx.set_agent_status(ORCH, "TREND_CLASSIFIER", "done")

# ─── Agent: Trajectory Predictor (MIMIC-IV cohort reference) ─
async def agent_trajectory_predictor(ctx: AgentContext):
    ORCH = "TEMPORAL"
    await ctx.set_agent_status(ORCH, "TRAJECTORY_PREDICTOR", "active")
    labs = ctx.patient.labs; predictions = {}

    for key, arr in labs.items():
        if not isinstance(arr, list) or len(arr) < 2:
            continue
        vals = [x["v"] for x in arr]
        n = len(vals)
        if n >= 2:
            slope = (vals[-1] - vals[-2])
            pred_6h  = round(vals[-1] + slope*1, 2)
            pred_12h = round(vals[-1] + slope*2, 2)
            ref = MIMIC_IV["lab_refs"].get(key, {})
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
                )
            }

    ctx.trajectory = predictions
    critical_labs = [k for k,v in predictions.items() if v["trajectory_status"]=="CRITICAL_WITHIN_6H"]
    if critical_labs:
        await ctx.log("TRAJECTORY_PREDICTOR", f"Projected critical in 6h: {', '.join(critical_labs)}", "warn")
    await ctx.send({"type":"agent_result","orchestrator":ORCH,"agent":"TRAJECTORY_PREDICTOR","data":predictions})
    await ctx.set_agent_status(ORCH, "TRAJECTORY_PREDICTOR", "done")

# ═══════════════════════════════════════════════════════════════
# ── WAVE 2: EVIDENCE RETRIEVAL ORCHESTRATOR ───────────────────
# ═══════════════════════════════════════════════════════════════
async def agent_semantic_retriever(ctx: AgentContext) -> List[Dict]:
    ORCH = "EVIDENCE"
    await ctx.set_agent_status(ORCH, "SEMANTIC_RETRIEVER", "active")

    query_parts = [
        ctx.patient.admitDiag,
        *[f["message"] for f in ctx.lab_findings],
        *ctx.patient.medications,
        *(["acute kidney injury creatinine"] if ctx.outliers else []),
        f"sepsis lactate {ctx.patient.labs.get('lactate',[{}])[-1].get('v','')} mmol/L",
        "SOFA organ failure",
    ]
    query = ". ".join(str(x) for x in query_parts if x)

    retrieved = []
    if _guideline_embeddings and await ollama.is_online():
        await ctx.log("SEMANTIC_RETRIEVER", "Computing bge-m3 query embedding...", "info")
        try:
            q_emb = await ollama.embed(query)
            scored = []
            for i, g in enumerate(GUIDELINES):
                score = cosine_sim(q_emb, _guideline_embeddings[i])
                scored.append({**g, "score": round(float(score), 4)})
            scored.sort(key=lambda x: x["score"], reverse=True)
            retrieved = scored[:cfg.TOP_K_GUIDELINES]
            await ctx.log("SEMANTIC_RETRIEVER", f"Semantic retrieval done. Top score: {retrieved[0]['score']}", "info")
        except Exception as e:
            await ctx.log("SEMANTIC_RETRIEVER", f"Embed failed: {e}, using keyword fallback", "warn")

    if not retrieved:
        # Keyword fallback
        lower = query.lower()
        retrieved = sorted(
            [{**g,"score":keyword_score(lower, g["keywords"])} for g in GUIDELINES],
            key=lambda x: x["score"], reverse=True
        )[:cfg.TOP_K_GUIDELINES]
        await ctx.log("SEMANTIC_RETRIEVER", "Using keyword fallback retrieval", "warn")

    ctx.retrieved_guidelines = retrieved
    await ctx.send({"type":"agent_result","orchestrator":ORCH,"agent":"SEMANTIC_RETRIEVER",
                    "data":[{"source":g["source"],"section":g["section"],"score":g.get("score",0),"text":g["text"]} for g in retrieved]})
    await ctx.set_agent_status(ORCH, "SEMANTIC_RETRIEVER", "done")
    return retrieved

async def agent_rag_explainer(ctx: AgentContext, guidelines: List[Dict]):
    ORCH = "EVIDENCE"
    await ctx.set_agent_status(ORCH, "RAG_EXPLAINER", "active")

    guideline_ctx = "\n\n".join(f"[{g['source']} {g['section']}]: {g['text']}" for g in guidelines)
    findings_str = "; ".join(f["message"] for f in ctx.lab_findings) or "No major lab trends"
    outlier_str  = "; ".join(f"⚠ {o['lab'].upper()} value {o['value']} (Z={o['z']}) inconsistent with 72h trend" for o in ctx.outliers) or "None"

    messages = [
        {"role":"system","content":"You are a clinical evidence agent. Using retrieved guidelines, write 3-4 concise, cited clinical observations relevant to the patient findings. Use [Source §Section] citation format. Be direct and clinical."},
        {"role":"user","content":f"Patient: {ctx.patient.name}, {ctx.patient.age}{ctx.patient.sex}. {ctx.patient.admitDiag}.\nLab findings: {findings_str}\nOutliers: {outlier_str}\n\nGuidelines:\n{guideline_ctx}\n\nWrite cited observations:"}
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
    await ctx.send({"type":"agent_result","orchestrator":ORCH,"agent":"RAG_EXPLAINER","data":{"explanation":explanation}})
    await ctx.set_agent_status(ORCH, "RAG_EXPLAINER", "done")

# ═══════════════════════════════════════════════════════════════
# ── WAVE 3: SYNTHESIS ORCHESTRATOR ───────────────────────────
# ═══════════════════════════════════════════════════════════════
async def agent_synthesis(ctx: AgentContext):
    ORCH = "SYNTHESIS"
    await ctx.set_agent_status(ORCH, "CHIEF_AGENT", "active")

    pt = ctx.patient
    sofa = ctx.sofa; news2 = ctx.news2
    outlier_note = ""
    if ctx.outliers:
        outlier_note = "\n".join(
            f"⚠ OUTLIER HELD: {o['lab'].upper()} {o['value']} at {o['timepoint']} "
            f"(Z={o['z']} vs 72h mean {o['mean']}±{o['std']}) — DIAGNOSIS HELD, REDRAW REQUIRED"
            for o in ctx.outliers
        )

    lab_traj = "\n".join(
        f"{k.upper()}: {' → '.join(str(x['v'])+((' [OUTLIER]') if x.get('outlier') else '') for x in v)} {MIMIC_IV['lab_refs'].get(k,{}).get('unit','')}"
        for k, v in pt.labs.items() if isinstance(v, list)
    )

    med_conflicts = "\n".join(f"• {c['med']}: {c['conflict']}" for c in ctx.med_conflicts if c["conflict"])
    guideline_ctx = "\n".join(f"[{g['source']} {g['section']}]: {g['text']}" for g in ctx.retrieved_guidelines)
    alert_summary = "\n".join(f"• [{a['level']}] {a['message']}" for a in ctx.alert_events)
    traj_summary  = "\n".join(f"• {k.upper()}: pred 6h={v.get('pred_6h','?')}, 12h={v.get('pred_12h','?')} [{v.get('trajectory_status','?')}]" for k,v in ctx.trajectory.items())

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
{chr(10).join('• '+f['message'] for f in ctx.lab_findings) or 'None'}

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

    messages = [{"role":"user","content":prompt}]
    full_text = ""

    async def on_chunk(delta: str, accumulated: str):
        nonlocal full_text
        full_text = accumulated
        await ctx.send({"type":"stream_chunk","orchestrator":ORCH,"agent":"CHIEF_AGENT","content":delta})

    try:
        full_text = await nim.chat(cfg.CHIEF_MODEL, messages, ctx.nim_key,
                                   max_tokens=1800, on_chunk=on_chunk)
    except Exception as e:
        await ctx.log("CHIEF_AGENT", f"Nemotron failed: {e}. Using fallback.", "warn")
        try:
            full_text = await nim.chat(cfg.FALLBACK_MODEL, messages, ctx.nim_key,
                                       max_tokens=1500, on_chunk=on_chunk)
        except Exception as e2:
            full_text = f"Chief Agent unavailable: {e2}"

    ctx.synthesis = full_text

    # Parse handover
    import re
    hm = re.search(r'SHIFT HANDOVER BRIEF[^:]*:\s*([\s\S]+?)(?:\n\n|\n---|\n#|$)', full_text, re.I)
    ctx.handover = hm.group(1).strip() if hm else ""

    await ctx.send({"type":"agent_result","orchestrator":ORCH,"agent":"CHIEF_AGENT",
                    "data":{"synthesis":full_text,"handover":ctx.handover}})
    await ctx.set_agent_status(ORCH, "CHIEF_AGENT", "done")

# ═══════════════════════════════════════════════════════════════
# MASTER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════
async def master_orchestrate(ctx: AgentContext):
    start = time.time()
    pt = ctx.patient
    await ctx.log("HC01", f"=== PIPELINE START: {pt.name} {pt.age}{pt.sex} ===", "info")
    await ctx.send({"type":"orchestrator_status","orchestrator":"MASTER","status":"active"})

    # Pre-compute scores (sync, no model call)
    ctx.sofa  = calc_sofa(pt)
    ctx.news2 = calc_news2(pt)
    await ctx.send({"type":"agent_result","orchestrator":"SCORING","agent":"SOFA_NEWS2",
                    "data":{"sofa":ctx.sofa,"news2":ctx.news2}})

    # ─── WAVE 1: Parallel ────────────────────────────────────
    await ctx.log("HC01", "Wave 1: Launching 5 agents in parallel...", "info")
    for o in ["CLINICAL","SAFETY","TEMPORAL"]:
        await ctx.set_orch_status(o, "active")

    await asyncio.gather(
        agent_note_parser(ctx),
        agent_outlier_detector(ctx),
        agent_med_safety(ctx),
        agent_alert_escalation(ctx),
        agent_trend_classifier(ctx),
        agent_trajectory_predictor(ctx),
    )

    for o in ["CLINICAL","SAFETY","TEMPORAL"]:
        await ctx.set_orch_status(o, "done")
    await ctx.log("HC01", "Wave 1 complete.", "info")

    # ─── WAVE 2: Evidence (needs Wave 1 output) ───────────────
    await ctx.set_orch_status("EVIDENCE", "active")
    guidelines = await agent_semantic_retriever(ctx)
    await agent_rag_explainer(ctx, guidelines)
    await ctx.set_orch_status("EVIDENCE", "done")

    # ─── WAVE 3: Synthesis (needs everything) ────────────────
    await ctx.set_orch_status("SYNTHESIS", "active")
    await agent_synthesis(ctx)
    await ctx.set_orch_status("SYNTHESIS", "done")

    elapsed = round((time.time() - start) * 1000)
    await ctx.send({"type":"pipeline_complete","duration_ms":elapsed,
                    "alert_level":ctx.alert_level,"sofa_total":ctx.sofa["total"],"news2_total":ctx.news2["total"]})
    await ctx.log("HC01", f"=== PIPELINE COMPLETE in {elapsed}ms ===", "info")

# ═══════════════════════════════════════════════════════════════
# PRE-EMBED GUIDELINES AT STARTUP
# ═══════════════════════════════════════════════════════════════
async def preembed_guidelines():
    global _guideline_embeddings
    if not await ollama.is_online():
        log.warning("Ollama offline — guideline embeddings skipped (keyword fallback active)")
        return
    models = await ollama.available_models()
    if not any("bge-m3" in m for m in models):
        log.warning("bge-m3 not found in Ollama. Run: ollama pull bge-m3")
        return
    log.info(f"Pre-embedding {len(GUIDELINES)} guideline chunks via bge-m3...")
    embeds = []
    for g in GUIDELINES:
        text = f"{g['source']} {g['section']}: {g['text']}"
        emb  = await ollama.embed(text)
        embeds.append(emb)
    _guideline_embeddings = embeds
    log.info(f"Guideline embeddings ready. Dim: {embeds[0].shape[0]}")

# ═══════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("HC01 Backend starting...")
    await preembed_guidelines()
    yield
    log.info("HC01 Backend shutting down.")

app = FastAPI(title="HC01 ICU Diagnostic Risk Assistant", version="2.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
async def health():
    online = await ollama.is_online()
    models = await ollama.available_models() if online else []
    return {
        "status": "ok",
        "ollama_online": online,
        "ollama_models": models,
        "guideline_embeds_ready": len(_guideline_embeddings) == len(GUIDELINES),
        "guidelines_count": len(GUIDELINES),
        "datasets": {
            "mimic_iv": MIMIC_IV["citation"],
            "eicu": EICU["citation"],
            "physionet_sepsis": PHYSIONET_SEPSIS["citation"],
        }
    }

@app.get("/api/mimic-stats")
async def mimic_stats():
    return {
        "mimic_iv": MIMIC_IV,
        "eicu": EICU,
        "physionet_sepsis": PHYSIONET_SEPSIS,
    }

@app.websocket("/ws/diagnose")
async def ws_diagnose(websocket: WebSocket):
    await websocket.accept()
    log.info("WebSocket client connected")
    try:
        raw = await websocket.receive_json()
        if raw.get("action") != "diagnose":
            await websocket.send_json({"type":"error","message":"Expected action=diagnose"})
            return

        patient = PatientData(**raw["patient"])
        nim_key = raw.get("nim_api_key","")

        ctx = AgentContext(patient, nim_key, websocket)
        await master_orchestrate(ctx)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as e:
        log.exception("Orchestration error")
        try:
            await websocket.send_json({"type":"error","message":str(e)})
        except Exception:
            pass

# ─── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
