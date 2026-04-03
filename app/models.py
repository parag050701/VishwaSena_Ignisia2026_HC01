from typing import Any, Dict, List, Optional

from fastapi import WebSocket
from pydantic import BaseModel


class PatientLab(BaseModel):
    t: str
    v: float
    outlier: bool = False


class PatientVitals(BaseModel):
    hr: float
    bpSys: float
    bpDia: float
    map: float
    rr: float
    spo2: float
    temp: float
    gcs: float
    fio2: float
    pao2: float = 0.0


class PatientData(BaseModel):
    id: str
    name: str
    age: int
    sex: str
    weight: float
    daysInICU: float
    admitDiag: str
    vitals: PatientVitals
    labs: Dict[str, Any]
    medications: List[str]
    notes: List[Dict[str, str]]


class DiagnoseRequest(BaseModel):
    patient: PatientData
    nim_api_key: str


class AgentContext:
    def __init__(self, patient: PatientData, nim_key: str, ws: WebSocket):
        import time

        self.patient = patient
        self.nim_key = nim_key
        self.ws = ws
        self.timestamp = time.time()

        self.parsed_notes: Optional[Dict] = None
        self.lab_findings: List[Dict] = []
        self.outliers: List[Dict] = []
        self.med_conflicts: List[Dict] = []
        self.alert_level: str = "NORMAL"
        self.alert_events: List[Dict] = []
        self.trends: Dict[str, Any] = {}
        self.trajectory: Dict[str, Any] = {}
        self.retrieved_guidelines: List[Dict] = []
        self.rag_explanation: str = ""
        self.sofa: Optional[Dict] = None
        self.news2: Optional[Dict] = None
        self.synthesis: str = ""
        self.differential: List[Dict] = []
        self.confidence_scores: Dict[str, float] = {}
        self.handover: str = ""

    async def send(self, msg: Dict):
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
