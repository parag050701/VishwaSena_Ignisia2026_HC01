from typing import Any, Dict, List, Optional

from fastapi import WebSocket
from pydantic import BaseModel, field_validator


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

    @field_validator("hr")
    @classmethod
    def _hr(cls, v: float) -> float:
        if not (10 <= v <= 300):
            raise ValueError(f"HR {v} out of physiological range [10, 300]")
        return v

    @field_validator("spo2")
    @classmethod
    def _spo2(cls, v: float) -> float:
        if not (50 <= v <= 100):
            raise ValueError(f"SpO2 {v} out of range [50, 100]")
        return v

    @field_validator("gcs")
    @classmethod
    def _gcs(cls, v: float) -> float:
        if not (3 <= v <= 15):
            raise ValueError(f"GCS {v} out of range [3, 15]")
        return v

    @field_validator("fio2")
    @classmethod
    def _fio2(cls, v: float) -> float:
        # Accept both 0.21-1.0 and 21-100 (percent) form; normalise to fraction
        if v > 1.0:
            v = v / 100.0
        if not (0.0 < v <= 1.0):
            raise ValueError(f"FiO2 {v} out of range (0, 1]")
        return round(v, 4)

    @field_validator("temp")
    @classmethod
    def _temp(cls, v: float) -> float:
        if not (30.0 <= v <= 45.0):
            raise ValueError(f"Temp {v} out of range [30, 45]")
        return v


class PatientData(BaseModel):
    id: str
    name: str
    age: int
    sex: str
    weight: float
    daysInICU: float
    admitDiag: str
    vitals: PatientVitals
    labs: Dict[str, Any]        # {lab_name: [{t, v, outlier?}]}
    medications: List[str]
    notes: List[Dict[str, str]] # [{time, author, text}]

    @field_validator("age")
    @classmethod
    def _age(cls, v: int) -> int:
        if not (0 <= v <= 130):
            raise ValueError(f"Age {v} out of range [0, 130]")
        return v

    @field_validator("weight")
    @classmethod
    def _weight(cls, v: float) -> float:
        if not (1.0 <= v <= 500.0):
            raise ValueError(f"Weight {v} out of range [1, 500]")
        return v

    @field_validator("sex")
    @classmethod
    def _sex(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in ("M", "F", "U", "MALE", "FEMALE", "UNKNOWN"):
            raise ValueError(f"Sex must be M/F/U, got {v!r}")
        return v[0]  # normalise to single char


class DiagnoseRequest(BaseModel):
    patient: PatientData
    nim_api_key: str = ""


class PatientSummary(BaseModel):
    """Lightweight patient record returned by /api/patients."""
    subject_id: int
    hadm_id: int
    icustay_id: int
    age: int
    sex: str
    first_careunit: str
    los_days: float
    admit_time: str


class AgentContext:
    def __init__(self, patient: PatientData, nim_key: str, ws: Optional[WebSocket]):
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
        self.disease_timeline: List[Dict] = []   # chronological events from TEMPORAL_LAB_MAPPER
        self.timeline_str: str = ""              # human-readable timeline
        self.retrieved_guidelines: List[Dict] = []
        self.rag_explanation: str = ""
        self.sofa: Optional[Dict] = None
        self.news2: Optional[Dict] = None
        self.synthesis: str = ""
        self.handover: str = ""

    async def send(self, msg: Dict) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send_json(msg)
        except Exception:
            pass

    async def log(self, agent: str, message: str, level: str = "info") -> None:
        await self.send({"type": "log", "agent": agent, "level": level, "message": message})

    async def set_agent_status(self, orchestrator: str, agent: str, status: str) -> None:
        await self.send({"type": "agent_status", "orchestrator": orchestrator, "agent": agent, "status": status})

    async def set_orch_status(self, orchestrator: str, status: str) -> None:
        await self.send({"type": "orchestrator_status", "orchestrator": orchestrator, "status": status})
