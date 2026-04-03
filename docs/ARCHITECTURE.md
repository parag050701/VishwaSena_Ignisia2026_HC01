# HC01 Modular Architecture

## Overview

This project is an ICU diagnostic assistant that runs a FastAPI backend and a browser frontend.
The backend accepts a patient payload over WebSocket, runs a staged clinical reasoning pipeline, and streams structured updates back to the browser.

## File Layout

- `app/config.py` holds runtime configuration such as model names, timeouts, and retrieval settings.
- `app/data.py` holds all embedded clinical reference data, including MIMIC-IV context, eICU context, PhysioNet context, and guideline chunks.
- `app/models.py` defines the Pydantic request models and the shared agent context.
- `app/clients.py` wraps Ollama and NVIDIA NIM API access.
- `app/scoring.py` contains pure scoring helpers for SOFA, NEWS2, cosine similarity, and keyword retrieval.
- `app/agents.py` contains the orchestrated clinical agents plus the master pipeline.
- `app/main.py` exposes the FastAPI app and HTTP/WebSocket routes.
- `server.py` is now a compatibility wrapper so existing startup commands still work.

## How It Works

1. The frontend sends a `diagnose` message over `/ws/diagnose` with the patient payload and optional NIM key.
2. `app/main.py` builds a `PatientData` model and an `AgentContext` object.
3. `master_orchestrate()` in `app/agents.py` runs the pipeline in three waves.
4. Wave 1 runs note parsing, outlier detection, medication safety, alert escalation, trend classification, and trajectory prediction in parallel.
5. Wave 2 retrieves relevant guidelines and generates a short evidence summary.
6. Wave 3 calls the chief synthesis model to produce the final clinical assessment, differential diagnosis, concerns, actions, and handover brief.
7. Every step streams status and results back to the browser.

## Functionalities

- ICU note parsing and clinical NLP extraction.
- Lab outlier detection using a statistical Z-score threshold.
- Medication safety checks for AKI and nephrotoxin combinations.
- Alert escalation for lactate, hypotension, hypoxemia, procalcitonin, leukocytosis, and antibiotic timing.
- Temporal trend classification and short-horizon trajectory prediction.
- Guideline retrieval via embeddings or keyword fallback.
- Final synthesis into a structured doctor-facing report.
