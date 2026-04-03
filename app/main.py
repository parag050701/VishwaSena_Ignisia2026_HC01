import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import data
from .agents import master_orchestrate, preembed_guidelines
from .clients import ollama
from .models import AgentContext, PatientData

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("HC01")


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
        "guideline_embeds_ready": len(data._guideline_embeddings) == len(data.GUIDELINES),
        "guidelines_count": len(data.GUIDELINES),
        "datasets": {
            "mimic_iv": data.MIMIC_IV["citation"],
            "eicu": data.EICU["citation"],
            "physionet_sepsis": data.PHYSIONET_SEPSIS["citation"],
        },
    }


@app.get("/api/mimic-stats")
async def mimic_stats():
    return {
        "mimic_iv": data.MIMIC_IV,
        "eicu": data.EICU,
        "physionet_sepsis": data.PHYSIONET_SEPSIS,
    }


@app.websocket("/ws/diagnose")
async def ws_diagnose(websocket: WebSocket):
    await websocket.accept()
    log.info("WebSocket client connected")
    try:
        raw = await websocket.receive_json()
        if raw.get("action") != "diagnose":
            await websocket.send_json({"type": "error", "message": "Expected action=diagnose"})
            return

        patient = PatientData(**raw["patient"])
        nim_key = raw.get("nim_api_key", "")

        ctx = AgentContext(patient, nim_key, websocket)
        await master_orchestrate(ctx)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as e:
        log.exception("Orchestration error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
