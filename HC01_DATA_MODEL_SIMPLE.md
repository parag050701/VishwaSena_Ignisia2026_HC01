# HC01 — System Data Flow (Technical Overview)

---

## How Data Enters the System

```
MIMIC-III CSVs          FHIR Bundle (JSON)        Live FHIR EHR
(150 real patients)     (120 synthetic patients)  (optional, OAuth2)
        │                       │                        │
        └───────────────────────┴────────────────────────┘
                                │
                                ▼
                          PatientData
                     (single unified object)
```

---

## PatientData — What Gets Passed Around

```
PatientData
├── id, name, age, sex, weight
├── daysInICU, admitDiag
│
├── vitals     HR, BP, SpO2, Temp, RR, GCS, FiO2, PaO2
├── labs       { lab_name → [ { time, value } ] }
├── medications  [ "norepinephrine 0.1 mcg/kg/min", ... ]
└── notes      [ { time, author, text } ]
```

> All fields are validated on ingestion (ranges checked, FiO2 auto-normalised from % to fraction).

---

## Analysis Pipeline

```
PatientData
     │
     ├─── PHASE 1 (parallel) ──────────────────────────────────────────
     │    NOTE_PARSER          extract diagnoses/meds from free text
     │    OUTLIER_DETECTOR     flag abnormal lab spikes (Z-score > 2.5)
     │    TREND_CLASSIFIER     lab direction (improving / worsening)
     │    TEMPORAL_LAB_MAPPER  build chronological event timeline
     │
     ├─── PHASE 2 · Safety ────────────────────────────────────────────
     │    MED_SAFETY           drug–drug + drug–lab interactions
     │    ALERT_ESCALATION     triage → NORMAL / WARNING / CRITICAL
     │
     ├─── PHASE 3 (parallel) ──────────────────────────────────────────
     │    TRAJECTORY_PREDICTOR forecast labs 6–24 h ahead
     │    SEMANTIC_RETRIEVER   fetch relevant clinical guidelines (RAG)
     │    FAMILY_COMMUNICATOR  generate lay-language family update
     │
     ├─── PHASE 4 · Scoring ───────────────────────────────────────────
     │    SOFA_NEWS2           calculate SOFA (0–24) + NEWS2 scores
     │
     └─── PHASE 5 · Synthesis ─────────────────────────────────────────
          CHIEF_AGENT          full clinical report + handover brief
                               streamed token-by-token to UI
```

---

## What Comes Out

```
Output
├── alert_level        CRITICAL / WARNING / NORMAL
├── sofa               score 0–24  +  per-organ breakdown
├── news2              score  +  risk level
├── handover           2–4 sentence shift brief
├── synthesis          full AI clinical narrative
├── alerts             [ { severity, message } ]
├── outliers           [ { lab, value, Z-score } ]
├── med_conflicts      [ { drug, issue, severity } ]
├── trajectory         { lab → predicted_critical_in_hours }
└── guidelines         top-5 relevant clinical guidelines
```

---

## APIs at a Glance

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/fhir-local/patients` | GET | List 120 patients |
| `/api/fhir-local/patient/{id}` | GET | Full patient record |
| `/api/fhir-local/diagnose/{id}` | POST | Run analysis (blocking) |
| `/ws/diagnose` | WebSocket | Run analysis (streaming) |
| `/api/priority-queue` | GET | All patients ranked by SOFA |
| `/api/assistant/query` | POST | Ward chatbot query |
| `/api/voice/synthesize` | POST | Text → speech (MP3) |
| `/api/voice/transcribe` | POST | Speech → text |

---

## Voice

```
Mic → audio bytes
        │
        ▼
   STT:  NVIDIA Riva (Whisper Large v3)  →  fallback: faster-whisper
        │
        ▼  transcript
   LLM:  Llama 3 8B / Qwen 2.5 7B (NVIDIA NIM)
        │
        ▼  response text
   TTS:  edge-tts (Microsoft Neural)  →  fallback: pyttsx3
        │
        ▼
   Browser plays MP3 automatically
```

Auto-speak fires when analysis completes:
`"Alert status CRITICAL. SOFA score 11. [key finding]. [urgent action]."`

---

## LLM Stack

| Task | Model | Provider |
|---|---|---|
| Clinical synthesis | Nemotron 120B | NVIDIA NIM |
| Ward assistant | Llama 3 8B | NVIDIA NIM |
| Voice queries | Qwen 2.5 7B | NVIDIA NIM |
| Embeddings / RAG | nv-embedqa-e5-v5 | NVIDIA NIM |
| Local fallback | qwen3:4b | Ollama (on-device) |
| STT | Whisper Large v3 | NVIDIA Riva gRPC |
| TTS | Neural (Aria/Guy) | Microsoft edge-tts |
