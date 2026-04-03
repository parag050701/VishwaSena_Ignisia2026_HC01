"""
Local FHIR bundle reader for HC01.
Loads hc01_synthetic_fhir_bundle.json into memory and serves it
with the same interface as FHIRClient.to_patient_data().
No HTTP needed — works offline for demos.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ehr import _FHIRMapper, _LOINC_TO_LAB, _LOINC_TO_VITAL
from .models import PatientData

log = logging.getLogger("HC01.fhir_local")

_BUNDLE_PATH = Path(__file__).resolve().parent.parent / "data" / "fhir" / "hc01_synthetic_fhir_bundle.json"

# ─── In-memory index ──────────────────────────────────────────────────────────

class FHIRLocalStore:
    """
    In-memory index of a FHIR transaction bundle.
    Indexed by resourceType + id for O(1) lookups.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._by_type: Dict[str, Dict[str, Dict]] = {}   # type -> id -> resource
        self._obs_by_patient: Dict[str, List[Dict]] = {} # patient_id -> [obs]
        self._med_by_patient: Dict[str, List[Dict]] = {}
        self._cond_by_patient: Dict[str, List[Dict]] = {}
        self._enc_by_patient: Dict[str, List[Dict]] = {}
        self._doc_by_patient: Dict[str, List[Dict]] = {}

    def load(self) -> None:
        if self._loaded:
            return
        if not _BUNDLE_PATH.exists():
            log.warning("Local FHIR bundle not found: %s", _BUNDLE_PATH)
            self._loaded = True
            return

        log.info("Loading local FHIR bundle from %s…", _BUNDLE_PATH)
        with _BUNDLE_PATH.open() as fh:
            bundle = json.load(fh)

        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            rt  = res.get("resourceType", "")
            rid = res.get("id", "")
            self._by_type.setdefault(rt, {})[rid] = res

            # Index by patient reference
            subj_ref = (
                res.get("subject", {}).get("reference", "") or
                res.get("patient", {}).get("reference", "")
            )
            pid = subj_ref.replace("Patient/", "") if subj_ref else None
            if pid:
                if rt == "Observation":
                    self._obs_by_patient.setdefault(pid, []).append(res)
                elif rt in ("MedicationRequest", "MedicationStatement"):
                    self._med_by_patient.setdefault(pid, []).append(res)
                elif rt == "Condition":
                    self._cond_by_patient.setdefault(pid, []).append(res)
                elif rt == "Encounter":
                    self._enc_by_patient.setdefault(pid, []).append(res)
                elif rt == "DocumentReference":
                    self._doc_by_patient.setdefault(pid, []).append(res)

        counts = {k: len(v) for k, v in self._by_type.items()}
        log.info("FHIR bundle loaded: %s", counts)
        self._loaded = True

    # ── Accessors ─────────────────────────────────────────────────────────

    def get_patient(self, fhir_id: str) -> Optional[Dict]:
        self.load()
        return self._by_type.get("Patient", {}).get(fhir_id)

    def list_patients(self) -> List[Dict]:
        self.load()
        return list(self._by_type.get("Patient", {}).values())

    def get_observations(self, patient_id: str) -> List[Dict]:
        self.load()
        return self._obs_by_patient.get(patient_id, [])

    def get_medications(self, patient_id: str) -> List[Dict]:
        self.load()
        return self._med_by_patient.get(patient_id, [])

    def get_conditions(self, patient_id: str) -> List[Dict]:
        self.load()
        return self._cond_by_patient.get(patient_id, [])

    def get_encounters(self, patient_id: str) -> List[Dict]:
        self.load()
        return self._enc_by_patient.get(patient_id, [])

    def get_documents(self, patient_id: str) -> List[Dict]:
        self.load()
        return self._doc_by_patient.get(patient_id, [])

    def to_fhir_record(self, fhir_id: str) -> Optional[Dict]:
        """Build the same record dict that FHIRClient.get_full_record() returns."""
        pt = self.get_patient(fhir_id)
        if pt is None:
            return None
        return {
            "patient":      pt,
            "observations": self.get_observations(fhir_id),
            "medications":  self.get_medications(fhir_id),
            "conditions":   self.get_conditions(fhir_id),
            "encounters":   self.get_encounters(fhir_id),
            "documents":    self.get_documents(fhir_id),
        }

    def to_patient_data(self, fhir_id: str) -> Optional[PatientData]:
        record = self.to_fhir_record(fhir_id)
        if record is None:
            return None
        return _FHIRMapper(record).build()

    def list_patient_summaries(self) -> List[Dict]:
        """Lightweight list for /api/fhir-local/patients."""
        self.load()
        result = []
        for pt in self.list_patients():
            pid  = pt.get("id", "")
            name_entry = (pt.get("name") or [{}])[0]
            given  = " ".join(name_entry.get("given", []))
            family = name_entry.get("family", "")
            result.append({
                "fhir_id":  pid,
                "name":     f"{given} {family}".strip(),
                "gender":   pt.get("gender", "unknown"),
                "dob":      pt.get("birthDate", ""),
                "obs_count": len(self.get_observations(pid)),
                "med_count": len(self.get_medications(pid)),
            })
        return result


# ─── Singleton ────────────────────────────────────────────────────────────────
_store: Optional[FHIRLocalStore] = None

def get_local_store() -> FHIRLocalStore:
    global _store
    if _store is None:
        _store = FHIRLocalStore()
        _store.load()
    return _store
