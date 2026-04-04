# HC01 — Data Model & System Flow

---

## 1. Data Sources (Inputs)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES                               │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  MIMIC-III CSVs  │  │   FHIR Bundle    │  │  FHIR Live EHR   │  │
│  │  (150 patients)  │  │  (120 synthetic) │  │  (optional REST) │  │
│  │                  │  │                  │  │                  │  │
│  │ NOTEEVENTS.csv   │  │ hc01_synthetic_  │  │ OAuth2 token     │  │
│  │ LABEVENTS.csv    │  │ fhir_bundle.json │  │ Patient/Obs/Med  │  │
│  │ D_LABITEMS.csv   │  │                  │  │ Encounter/Cond   │  │
│  │ ICUSTAYS.csv     │  │ 120 × Patient    │  │                  │  │
│  │ PATIENTS.csv     │  │ 120 × Encounter  │  │                  │  │
│  │ PRESCRIPTIONS    │  │ 120 × Condition  │  │                  │  │
│  └────────┬─────────┘  │ 3528 × Obs       │  └────────┬─────────┘  │
│           │            │ 375 × MedReq     │           │            │
│           │            │ 361 × DocRef     │           │            │
│           │            └────────┬─────────┘           │            │
└───────────┼─────────────────────┼─────────────────────┼────────────┘
            │                     │                      │
            ▼                     ▼                      ▼
     data_loader.py          fhir_local.py            ehr.py
     build_patient_data()    FHIRLocalLoader()        FHIRClient()
            │                     │                      │
            └──────────┬──────────┘                      │
                       ▼                                  │
                  PatientData ◄─────────────────────────-─┘
```

---

## 2. Core Data Models

### `PatientData` — the universal patient object passed to all agents

```
PatientData
├── id            str         "fhir-hc01-patient-001" | subject_id
├── name          str         "Chen Wei"
├── age           int         0–130  (validated)
├── sex           str         "M" | "F" | "U"  (normalised to 1 char)
├── weight        float       kg, 1–500  (validated)
├── daysInICU     float       4.8
├── admitDiag     str         "Sepsis (disorder)"
│
├── vitals        PatientVitals
│   ├── hr        float       bpm  [10–300]
│   ├── bpSys     float       mmHg
│   ├── bpDia     float       mmHg
│   ├── map       float       mmHg
│   ├── rr        float       breaths/min
│   ├── spo2      float       % [50–100]
│   ├── temp      float       °C  [30–45]
│   ├── gcs       float       [3–15]
│   ├── fio2      float       fraction (0–1), auto-normalised from %
│   └── pao2      float       mmHg  (default 0.0)
│
├── labs          Dict[str, List[PatientLab]]
│   │             key = lab name  e.g. "wbc", "creatinine", "lactate"
│   └── each →   PatientLab
│                 ├── t        str    ISO datetime "2026-04-03T08:00:00"
│                 ├── v        float  value
│                 └── outlier  bool   flagged by OUTLIER_DETECTOR
│
├── medications   List[str]   ["norepinephrine 0.1 mcg/kg/min", ...]
│
└── notes         List[Dict]
                  ├── time    str    "2026-04-03T06:00"
                  ├── author  str    "Nurse"
                  └── text    str    free-text clinical note
```

### `PatientLab`

```
PatientLab
├── t        str    ISO timestamp
├── v        float  numeric value
└── outlier  bool   = True if Z-score > 2.5 (cfg.OUTLIER_Z)
```

### `PatientVitals`  (validated on ingestion)

```
PatientVitals
├── hr      float   heart rate bpm
├── bpSys   float   systolic BP
├── bpDia   float   diastolic BP
├── map     float   mean arterial pressure
├── rr      float   respiratory rate
├── spo2    float   oxygen saturation %
├── temp    float   temperature °C
├── gcs     float   Glasgow Coma Scale
├── fio2    float   fraction inspired O2  (auto-converts 21→0.21)
└── pao2    float   partial pressure O2 mmHg
```

---

## 3. API Endpoints & Request / Response

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            REST + WebSocket APIs                            │
├──────────────────────┬────────────────────────────┬────────────────────────┤
│  Endpoint            │  Request Body              │  Response              │
├──────────────────────┼────────────────────────────┼────────────────────────┤
│ GET /api/patients    │  —                         │ List[PatientSummary]   │
│                      │                            │  subject_id, hadm_id   │
│                      │                            │  age, sex, careunit    │
│                      │                            │  los_days, admit_time  │
├──────────────────────┼────────────────────────────┼────────────────────────┤
│ GET /api/fhir-local  │  —                         │ List[{fhir_id, name,   │
│      /patients       │                            │  gender, dob,          │
│                      │                            │  obs_count, med_count}]│
├──────────────────────┼────────────────────────────┼────────────────────────┤
│ GET /api/fhir-local  │  —  (path param: fhir_id)  │ {fhir_id, patient:     │
│      /patient/{id}   │                            │   PatientData,         │
│                      │                            │   source: "fhir_local"}│
├──────────────────────┼────────────────────────────┼────────────────────────┤
│ POST /api/fhir-local │  {}  (path param: fhir_id) │ DiagnoseHTTPResponse   │
│      /diagnose/{id}  │  nim_api_key?: string      │  (see §4)              │
├──────────────────────┼────────────────────────────┼────────────────────────┤
│ POST /api/diagnose   │  DiagnoseHTTPRequest       │ DiagnoseHTTPResponse   │
│  (blocking REST)     │  ├── patient: PatientData  │  (see §4)              │
│                      │  ├── nim_api_key?: string  │                        │
│                      │  └── include_council: bool │                        │
├──────────────────────┼────────────────────────────┼────────────────────────┤
│ WS /ws/diagnose      │  DiagnoseRequest (JSON)    │ Streaming SSE-style    │
│  (streaming)         │  ├── action: "diagnose"    │  agent_result msgs     │
│                      │  ├── patient: PatientData  │  (see §5)              │
│                      │  └── nim_api_key?: string  │                        │
├──────────────────────┼────────────────────────────┼────────────────────────┤
│ GET /api/priority    │  —                         │ List[{fhir_id, name,   │
│      -queue          │                            │  sofa, alert_level,    │
│                      │                            │  rank, primary_diag}]  │
├──────────────────────┼────────────────────────────┼────────────────────────┤
│ POST /api/assistant  │  AssistantQueryRequest     │ AssistantQueryResponse │
│      /query          │  ├── query: string         │  ├── intent: string    │
│                      │  ├── context_patient?:str  │  ├── answer: string    │
│                      │  └── current_diagnostics?: │  ├── action?: string   │
│                      │      DiagContext (see §6)  │  ├── action_data?: {}  │
│                      │                            │  └── tts_text: string  │
├──────────────────────┼────────────────────────────┼────────────────────────┤
│ POST /api/voice      │  SynthesizeRequest         │ {available: bool,      │
│      /synthesize     │  ├── text: string          │  audio_b64: string,    │
│                      │  ├── voice?: "female"|"male│  mime_type: string}    │
│                      │  └── speed?: float (1.0)   │                        │
├──────────────────────┼────────────────────────────┼────────────────────────┤
│ POST /api/voice      │  TranscribeRequest         │ {text?: string,        │
│      /transcribe     │  └── audio_b64: string     │  available: bool}      │
└──────────────────────┴────────────────────────────┴────────────────────────┘
```

---

## 4. `DiagnoseHTTPResponse` — full analysis output

```
DiagnoseHTTPResponse
├── synthesis          str      full AI clinical narrative
├── handover           str      shift handover brief (2–4 sentences)
├── alert_level        str      "CRITICAL" | "WARNING" | "NORMAL"
├── sofa_total         int      SOFA total score 0–24
├── news2_total        int      NEWS2 total score
│
├── sofa               Dict
│   ├── total          int
│   ├── resp           int      PaO2/FiO2 ratio component
│   ├── coag           int      platelets component
│   ├── liver          int      bilirubin component
│   ├── cardio         int      MAP / vasopressors component
│   ├── cns            int      GCS component
│   ├── renal          int      creatinine component
│   ├── mortality_pct  float    estimated mortality %
│   └── pf_ratio       float    PaO2/FiO2
│
├── news2              Dict
│   ├── total          int
│   ├── level          str      "LOW" | "MEDIUM" | "HIGH"
│   └── breakdown      Dict     {RR, SpO2, O2, SBP, HR, GCS, Temp}
│
├── lab_findings       List[Dict]
│   └── each           {lab, value, unit, status, trend, reference}
│
├── alerts             List[Dict]
│   └── each           {type, severity, message, timestamp}
│
├── med_conflicts      List[Dict]
│   └── each           {drug, status, conflict?, severity?, note?}
│
├── outliers           List[Dict]
│   └── each           {lab, value, z, severity, note}
│
├── trajectory         Dict     {predictions: {lab: {slope, critical_in_h}}}
├── disease_timeline   List[Dict]
│   └── each           {timestamp, category, parameter, value, status}
├── timeline_str       str      human-readable timeline text
│
├── parsed_notes       Dict?    {diagnoses, medications, vitals, allergies, plan}
├── family_communication Dict?  {summary, questions_answered, next_steps}
│
├── hold_reasons       List[str]  outlier-triggered diagnosis holds
├── council_consult    str?       LLM council second-opinion (if enabled)
└── council_status     str?       "available" | "timeout" | "disabled"
```

---

## 5. WebSocket Streaming Messages (`/ws/diagnose`)

The frontend opens a WebSocket and receives a stream of typed messages:

```
CLIENT → SERVER (once)
{
  "action":      "diagnose",
  "patient":     PatientData,       ← full object
  "nim_api_key": ""                  ← optional override
}

SERVER → CLIENT (stream)
┌─────────────────────┬────────────────────────────────────────────────────┐
│ type                │ payload fields                                     │
├─────────────────────┼────────────────────────────────────────────────────┤
│ agent_status        │ orchestrator, agent, status ("active"|"done"|"err")│
│ orchestrator_status │ orchestrator, status                               │
│ log                 │ agent, level ("info"|"warn"|"error"), message      │
│ stream_chunk        │ orchestrator, agent, content  ← CHIEF_AGENT live  │
├─────────────────────┼────────────────────────────────────────────────────┤
│ agent_result        │ agent, data (varies by agent — see below)          │
│   NOTE_PARSER       │ {diagnoses, medications, vitals, allergies, plan}  │
│   OUTLIER_DETECTOR  │ List[{lab, value, z, severity}]                    │
│   MED_SAFETY        │ List[{drug, status, conflict, severity, note}]     │
│   ALERT_ESCALATION  │ {level, alerts: List[{type, severity, message}]}   │
│   SOFA_NEWS2        │ {sofa: SOFADict, news2: NEWS2Dict}                 │
│   TREND_CLASSIFIER  │ List[{lab, trend, slope, change_pct}]              │
│   TRAJECTORY_PRED.  │ {predictions: {lab: {slope, critical_in_hours}}}   │
│   TEMPORAL_LAB_MAP  │ {timeline: List[{timestamp, category, param, val}]}│
│   SEMANTIC_RETR.    │ List[{title, content, score, source}]              │
│   RAG_EXPLAINER     │ {explanation, guidelines_used}                     │
│   FAMILY_COMM.      │ {summary, questions_answered, next_steps}          │
│   CHIEF_AGENT       │ {synthesis, handover, alert_level,                 │
│                     │  sofa_total, news2_total, hold_reasons}            │
├─────────────────────┼────────────────────────────────────────────────────┤
│ pipeline_complete   │ sofa_total, news2_total, alert_level, duration_s   │
│ error               │ message                                            │
└─────────────────────┴────────────────────────────────────────────────────┘
```

---

## 6. Ward Assistant — `DiagContext` (live diagnostics passthrough)

When a patient is actively being viewed and analysis has run, the frontend
passes live results back to the assistant so it can answer patient-specific
questions:

```
AssistantQueryRequest
├── query               str     "what's the SOFA score?"
├── context_patient     str?    "hc01-patient-001"
└── current_diagnostics Dict?   (DiagContext)
    ├── sofa            int | SOFADict   (number or full breakdown)
    ├── news2           int | NEWS2Dict
    ├── alertLevel      str     "CRITICAL" | "WARNING" | "NORMAL"
    ├── handover        str     first 500 chars of handover brief
    ├── outliers        List    up to 10 outlier dicts
    ├── medConflicts    List    up to 5 conflict dicts
    └── alerts          List    up to 5 alert dicts
```

```
AssistantQueryResponse
├── intent      str     "answer_question" | "load_patient"
│                       "show_priority"  | "run_diagnostics"
├── answer      str     clinical answer (≤100 words)
├── action      str?    "load_patient" | "show_priority" |
│                       "run_diagnostics" | null
├── action_data Dict?   {fhir_id: str}  (for load_patient)
└── tts_text    str     1-sentence spoken version of answer
```

---

## 7. Agent Orchestration Pipeline

```
PatientData
     │
     ▼
AgentContext (shared state object)
     │
     ├──── PHASE 1 · PARALLEL ──────────────────────────────────────────────┐
     │     ├── NOTE_PARSER          parse free-text notes → diagnoses/meds  │
     │     ├── OUTLIER_DETECTOR     Z-score lab spike detection              │
     │     ├── TREND_CLASSIFIER     lab slope / direction                   │
     │     └── TEMPORAL_LAB_MAPPER  chronological event timeline            │
     │                                                                       │
     ├──── PHASE 2 · SAFETY (depends on Phase 1) ───────────────────────────┤
     │     ├── MED_SAFETY           drug–drug + drug–lab interactions        │
     │     └── ALERT_ESCALATION     rule-based + LLM alert triage            │
     │                                                                       │
     ├──── PHASE 3 · PARALLEL ──────────────────────────────────────────────┤
     │     ├── TRAJECTORY_PREDICTOR 6–24h lab trajectory forecasting        │
     │     ├── SEMANTIC_RETRIEVER   RAG: top-K clinical guidelines           │
     │     ├── RAG_EXPLAINER        NIM explanation of guideline relevance   │
     │     └── FAMILY_COMMUNICATOR  lay-language family update draft         │
     │                                                                       │
     ├──── PHASE 4 · SCORING ───────────────────────────────────────────────┤
     │     └── SOFA_NEWS2           calculate SOFA + NEWS2 scores            │
     │                                                                       │
     └──── PHASE 5 · SYNTHESIS ─────────────────────────────────────────────┘
           └── CHIEF_AGENT          Nemotron 120B (or fallback)
                                    Full clinical narrative + handover brief
                                    Streams token-by-token to frontend
```

---

## 8. Voice Pipeline

```
SPEECH INPUT
     │
     ▼ audio bytes (WAV / WebM)
┌────────────────────────────────┐
│         transcribe_audio()     │
│  1. Riva gRPC (Whisper Lg v3)  │ ← grpc.nvcf.nvidia.com:443
│     NIM_STT_API_KEY            │   function-id: b702f636-...
│  2. faster-whisper (local)     │ ← fallback, base model
└────────────┬───────────────────┘
             │ text transcript
             ▼
┌────────────────────────────────┐
│         voice_query()          │
│  LLM: Qwen 2.5 7B (NIM)        │
│  Context: PatientData + history│
│  Max 400 tokens, temp 0.3      │
└────────────┬───────────────────┘
             │ response text
             ▼
┌────────────────────────────────┐
│       synthesize_speech()      │
│  1. edge-tts (Microsoft Neural)│ ← en-US-AriaNeural / GuyNeural
│     Returns: MP3 (magic 0xFF)  │   free, no key needed
│  2. pyttsx3 (local, last resort│ ← WAV, robotic — avoid
└────────────┬───────────────────┘
             │ audio bytes (MP3)
             ▼
        browser Audio()
```

---

## 9. LLM Routing

```
Task                Model                  Key               Fallback
─────────────────────────────────────────────────────────────────────
CHIEF_AGENT         Nemotron 120B          NIM_API_KEY_CHIEF  Qwen 2.5 7B
Ward Assistant      Llama 3 8B             NIM_API_KEY_FAST   Qwen 2.5 7B
Voice Query         Qwen 2.5 7B            NIM_API_KEY_FALLB  Ollama local
Embeddings (RAG)    nv-embedqa-e5-v5       NIM_API_KEY_CHIEF  nomic-embed
Clinical Notes      qwen3:4b               —                  (Ollama local)
STT                 Whisper Large v3       NIM_STT_API_KEY    faster-whisper
TTS                 MS Neural (edge-tts)   —  (free)          pyttsx3
```

---

## 10. Auto-Speak Trigger (new)

When `CHIEF_AGENT` result arrives in the frontend:

```
CHIEF_AGENT result received
        │
        ├── state.lastDiag.alertLevel  = data.alert_level
        ├── state.lastDiag.handover    = data.handover
        ├── state.lastDiag.synthesis   = data.synthesis
        │
        └── setTimeout(autoSpeakReport, 600ms)
                │
                │  builds spoken text (≤380 chars):
                │  "Alert status: CRITICAL.
                │   SOFA score 11.
                │   [first 2 sentences of handover]
                │   [top alert message]"
                │
                ▼
            POST /api/voice/synthesize
                │
                ▼
            edge-tts → MP3 → browser Audio().play()
```
