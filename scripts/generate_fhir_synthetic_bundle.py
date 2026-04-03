#!/usr/bin/env python3
"""Generate FHIR R4 synthetic resources from HC01 synthetic ICU dataset.

Input:
- data/hc01_synthetic_icu_dataset.json

Outputs:
- data/fhir/hc01_synthetic_fhir_bundle.json  (transaction bundle)
- data/fhir/hc01_synthetic_fhir_ndjson/      (per-resource NDJSON files)
- data/fhir/README.md                        (import instructions)
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_JSON = ROOT / "data" / "hc01_synthetic_icu_dataset.json"
FHIR_DIR = ROOT / "data" / "fhir"
BUNDLE_PATH = FHIR_DIR / "hc01_synthetic_fhir_bundle.json"
NDJSON_DIR = FHIR_DIR / "hc01_synthetic_fhir_ndjson"

FHIR_PROFILE = {
    "source": "HC01 synthetic generator",
    "schema": "FHIR R4",
    "version": "1.0.0",
}

LAB_LOINC = {
    "creatinine": ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", "mg/dL"),
    "lactate": ("32693-4", "Lactate [Moles/volume] in Blood", "mmol/L"),
    "wbc": ("26464-8", "Leukocytes [#/volume] in Blood", "10*3/uL"),
    "platelets": ("777-3", "Platelets [#/volume] in Blood", "10*3/uL"),
    "bilirubin": ("1975-2", "Bilirubin.total [Mass/volume] in Serum or Plasma", "mg/dL"),
    "bun": ("3094-0", "Urea nitrogen [Mass/volume] in Serum or Plasma", "mg/dL"),
    "procalcitonin": ("33959-8", "Procalcitonin [Mass/volume] in Serum or Plasma", "ng/mL"),
    "pao2": ("2703-7", "Oxygen [Partial pressure] in Arterial blood", "mm[Hg]"),
}

VITAL_LOINC = {
    "hr": ("8867-4", "Heart rate", "/min"),
    "bpSys": ("8480-6", "Systolic blood pressure", "mm[Hg]"),
    "bpDia": ("8462-4", "Diastolic blood pressure", "mm[Hg]"),
    "map": ("8478-0", "Mean blood pressure", "mm[Hg]"),
    "rr": ("9279-1", "Respiratory rate", "/min"),
    "spo2": ("59408-5", "Oxygen saturation in Arterial blood by Pulse oximetry", "%"),
    "temp": ("8310-5", "Body temperature", "Cel"),
    "gcs": ("9269-2", "Glasgow coma score total", "{score}"),
    "fio2": ("19994-3", "Oxygen/Inspired gas setting [Volume Fraction] Ventilator", "%"),
    "pao2": ("2703-7", "Oxygen [Partial pressure] in Arterial blood", "mm[Hg]"),
}


@dataclass
class FHIRResourceSet:
    patients: List[Dict[str, Any]]
    encounters: List[Dict[str, Any]]
    conditions: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    medication_requests: List[Dict[str, Any]]
    document_references: List[Dict[str, Any]]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_time_label(label: str, base_dt: datetime) -> datetime:
    s = (label or "").strip().lower()
    if s in {"day 1", "initial"}:
        return base_dt - timedelta(days=2)
    if s in {"day 2", "mid-shift"}:
        return base_dt - timedelta(days=1)
    if s in {"today", "latest"}:
        return base_dt
    if s.endswith("+"):
        core = s[:-1]
        try:
            hh, mm = core.split(":")
            return base_dt.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        except Exception:
            return base_dt
    if ":" in s:
        try:
            hh, mm = s.split(":")
            return base_dt.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        except Exception:
            return base_dt
    return base_dt


def _birthdate_from_age(age: int, ref: datetime) -> str:
    year = ref.year - max(0, min(130, int(age)))
    return f"{year:04d}-01-01"


def _canonical_casetype(case_type: str) -> Tuple[str, str]:
    mapping = {
        "sepsis": ("91302008", "Sepsis (disorder)"),
        "ards": ("67782005", "Acute respiratory distress syndrome (disorder)"),
        "aki": ("14669001", "Acute kidney injury (disorder)"),
        "cardiac": ("703273000", "Cardiogenic shock (disorder)"),
        "neuro": ("443883004", "Encephalopathy (disorder)"),
        "stable": ("698247007", "Stable clinical condition (finding)"),
    }
    return mapping.get(case_type, ("55607006", "Problem"))


def _build_patient(patient: Dict[str, Any], idx: int, ref_dt: datetime) -> Dict[str, Any]:
    pid = f"hc01-patient-{idx:03d}"
    raw_name = patient.get("name", f"Synthetic {idx}")
    parts = raw_name.replace("Mr.", "").replace("Ms.", "").strip().split(" ")
    given = parts[:-1] or [parts[0] if parts else "Synthetic"]
    family = parts[-1] if parts else str(idx)
    sex = (patient.get("sex") or "U").lower()
    gender = "male" if sex.startswith("m") else "female" if sex.startswith("f") else "unknown"

    return {
        "resourceType": "Patient",
        "id": pid,
        "identifier": [
            {"system": "urn:hc01:synthetic-subject-id", "value": patient.get("subject_id", f"HC01-P{idx:03d}")},
            {"system": "urn:hc01:synthetic-patient-id", "value": patient.get("id", f"P{idx:03d}")},
        ],
        "active": True,
        "name": [{"use": "official", "family": family, "given": given}],
        "gender": gender,
        "birthDate": _birthdate_from_age(int(patient.get("age", 65)), ref_dt),
    }


def _build_encounter(patient_res: Dict[str, Any], patient: Dict[str, Any], idx: int, ref_dt: datetime) -> Dict[str, Any]:
    eid = f"hc01-encounter-{idx:03d}"
    los = float(patient.get("daysInICU", 2.0))
    start = ref_dt - timedelta(days=max(0.25, los))
    reason_text = patient.get("admitDiag", "ICU Admission")
    return {
        "resourceType": "Encounter",
        "id": eid,
        "status": "in-progress",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "IMP", "display": "inpatient encounter"},
        "subject": {"reference": f"Patient/{patient_res['id']}"},
        "period": {"start": _iso(start), "end": _iso(ref_dt)},
        "reasonCode": [{"text": reason_text}],
        "serviceType": {"text": "Critical Care Medicine"},
    }


def _build_condition(patient_res: Dict[str, Any], encounter_res: Dict[str, Any], patient: Dict[str, Any], idx: int) -> Dict[str, Any]:
    cid = f"hc01-condition-{idx:03d}"
    code, display = _canonical_casetype(str(patient.get("caseType", "")))
    return {
        "resourceType": "Condition",
        "id": cid,
        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
        "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "encounter-diagnosis"}]}],
        "code": {
            "coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}],
            "text": patient.get("admitDiag", display),
        },
        "subject": {"reference": f"Patient/{patient_res['id']}"},
        "encounter": {"reference": f"Encounter/{encounter_res['id']}"},
    }


def _obs_base(obs_id: str, patient_ref: str, encounter_ref: str, loinc: str, display: str, dt: datetime, category: str) -> Dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": category}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}], "text": display},
        "subject": {"reference": patient_ref},
        "encounter": {"reference": encounter_ref},
        "effectiveDateTime": _iso(dt),
    }


def _build_vital_observations(patient_res: Dict[str, Any], encounter_res: Dict[str, Any], patient: Dict[str, Any], idx: int, ref_dt: datetime) -> List[Dict[str, Any]]:
    obs = []
    v = patient.get("vitals", {})
    for key, (loinc, display, unit) in VITAL_LOINC.items():
        if key not in v:
            continue
        value = float(v[key])
        if key == "fio2" and value <= 1.0:
            value = value * 100.0
        oid = f"hc01-obs-vital-{idx:03d}-{key}"
        base = _obs_base(oid, f"Patient/{patient_res['id']}", f"Encounter/{encounter_res['id']}", loinc, display, ref_dt, "vital-signs")
        base["valueQuantity"] = {"value": round(value, 3), "unit": unit, "system": "http://unitsofmeasure.org", "code": unit}
        obs.append(base)
    return obs


def _build_lab_observations(patient_res: Dict[str, Any], encounter_res: Dict[str, Any], patient: Dict[str, Any], idx: int, ref_dt: datetime) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    labs = patient.get("labs", {})
    for key, series in labs.items():
        if key not in LAB_LOINC:
            continue
        loinc, display, unit = LAB_LOINC[key]
        if isinstance(series, (int, float)):
            dt = ref_dt
            oid = f"hc01-obs-lab-{idx:03d}-{key}-0"
            base = _obs_base(oid, f"Patient/{patient_res['id']}", f"Encounter/{encounter_res['id']}", loinc, display, dt, "laboratory")
            base["valueQuantity"] = {"value": round(float(series), 3), "unit": unit, "system": "http://unitsofmeasure.org", "code": unit}
            observations.append(base)
            continue

        if not isinstance(series, list):
            continue

        for j, point in enumerate(series):
            if not isinstance(point, dict) or "v" not in point:
                continue
            dt = _parse_time_label(str(point.get("t", "")), ref_dt)
            oid = f"hc01-obs-lab-{idx:03d}-{key}-{j}"
            base = _obs_base(oid, f"Patient/{patient_res['id']}", f"Encounter/{encounter_res['id']}", loinc, display, dt, "laboratory")
            base["valueQuantity"] = {
                "value": round(float(point["v"]), 3),
                "unit": unit,
                "system": "http://unitsofmeasure.org",
                "code": unit,
            }
            if point.get("outlier"):
                base["note"] = [{"text": "Flagged as synthetic outlier in source dataset."}]
            observations.append(base)
    return observations


def _build_medication_requests(patient_res: Dict[str, Any], encounter_res: Dict[str, Any], patient: Dict[str, Any], idx: int) -> List[Dict[str, Any]]:
    out = []
    meds = patient.get("medications", []) or []
    for j, med in enumerate(meds):
        mid = f"hc01-medreq-{idx:03d}-{j:02d}"
        out.append(
            {
                "resourceType": "MedicationRequest",
                "id": mid,
                "status": "active",
                "intent": "order",
                "subject": {"reference": f"Patient/{patient_res['id']}"},
                "encounter": {"reference": f"Encounter/{encounter_res['id']}"},
                "medicationCodeableConcept": {"text": med.split(" — ")[0][:120]},
                "dosageInstruction": [{"text": med[:200]}],
            }
        )
    return out


def _build_document_references(patient_res: Dict[str, Any], encounter_res: Dict[str, Any], patient: Dict[str, Any], idx: int, ref_dt: datetime) -> List[Dict[str, Any]]:
    out = []
    notes = patient.get("notes", []) or []
    for j, note in enumerate(notes):
        did = f"hc01-doc-{idx:03d}-{j:02d}"
        dt = _parse_time_label(str(note.get("time", "")), ref_dt)
        text = str(note.get("text", ""))
        payload = base64.b64encode(text.encode("utf-8")).decode("utf-8")
        out.append(
            {
                "resourceType": "DocumentReference",
                "id": did,
                "status": "current",
                "type": {"text": "Clinical Note"},
                "subject": {"reference": f"Patient/{patient_res['id']}"},
                "date": _iso(dt),
                "description": str(note.get("author", "Clinical Note"))[:120],
                "context": {
                    "encounter": [{"reference": f"Encounter/{encounter_res['id']}"}],
                },
                "content": [
                    {
                        "attachment": {
                            "contentType": "text/plain",
                            "data": payload,
                            "title": f"Synthetic Note {j + 1}",
                        }
                    }
                ],
            }
        )
    return out


def build_resources(dataset: Dict[str, Any]) -> FHIRResourceSet:
    patients_out: List[Dict[str, Any]] = []
    encounters_out: List[Dict[str, Any]] = []
    conditions_out: List[Dict[str, Any]] = []
    observations_out: List[Dict[str, Any]] = []
    medreq_out: List[Dict[str, Any]] = []
    docs_out: List[Dict[str, Any]] = []

    now = datetime.now(timezone.utc)

    for i, patient in enumerate(dataset.get("patients", []), start=1):
        ref_dt = now - timedelta(hours=i)

        p = _build_patient(patient, i, ref_dt)
        e = _build_encounter(p, patient, i, ref_dt)
        c = _build_condition(p, e, patient, i)
        obs_v = _build_vital_observations(p, e, patient, i, ref_dt)
        obs_l = _build_lab_observations(p, e, patient, i, ref_dt)
        meds = _build_medication_requests(p, e, patient, i)
        docs = _build_document_references(p, e, patient, i, ref_dt)

        patients_out.append(p)
        encounters_out.append(e)
        conditions_out.append(c)
        observations_out.extend(obs_v)
        observations_out.extend(obs_l)
        medreq_out.extend(meds)
        docs_out.extend(docs)

    return FHIRResourceSet(
        patients=patients_out,
        encounters=encounters_out,
        conditions=conditions_out,
        observations=observations_out,
        medication_requests=medreq_out,
        document_references=docs_out,
    )


def to_transaction_bundle(resources: FHIRResourceSet) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []

    def add(resource: Dict[str, Any]) -> None:
        rid = resource["id"]
        rtype = resource["resourceType"]
        entries.append(
            {
                "fullUrl": f"urn:uuid:{rtype}-{rid}",
                "resource": resource,
                "request": {
                    "method": "PUT",
                    "url": f"{rtype}/{rid}",
                },
            }
        )

    for collection in [
        resources.patients,
        resources.encounters,
        resources.conditions,
        resources.observations,
        resources.medication_requests,
        resources.document_references,
    ]:
        for r in collection:
            add(r)

    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "meta": {"tag": [{"system": "urn:hc01", "code": "synthetic"}]},
        "identifier": {"system": "urn:hc01:fhir-export", "value": "hc01-synthetic-bundle-v1"},
        "timestamp": _iso(datetime.now(timezone.utc)),
        "entry": entries,
    }


def write_ndjson(resources: FHIRResourceSet) -> None:
    NDJSON_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        "Patient.ndjson": resources.patients,
        "Encounter.ndjson": resources.encounters,
        "Condition.ndjson": resources.conditions,
        "Observation.ndjson": resources.observations,
        "MedicationRequest.ndjson": resources.medication_requests,
        "DocumentReference.ndjson": resources.document_references,
    }
    for filename, rows in mapping.items():
        out = NDJSON_DIR / filename
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(bundle: Dict[str, Any], resources: FHIRResourceSet) -> None:
    counts = {
        "Patient": len(resources.patients),
        "Encounter": len(resources.encounters),
        "Condition": len(resources.conditions),
        "Observation": len(resources.observations),
        "MedicationRequest": len(resources.medication_requests),
        "DocumentReference": len(resources.document_references),
        "BundleEntry": len(bundle.get("entry", [])),
    }

    readme = f"""# HC01 FHIR Synthetic Export

This folder contains FHIR R4-compliant synthetic data generated from `data/hc01_synthetic_icu_dataset.json`.

## Files

- `hc01_synthetic_fhir_bundle.json`: single FHIR `Bundle` (`type=transaction`) for one-shot import.
- `hc01_synthetic_fhir_ndjson/`: per-resource NDJSON files.

## Resource counts

- Patient: {counts['Patient']}
- Encounter: {counts['Encounter']}
- Condition: {counts['Condition']}
- Observation: {counts['Observation']}
- MedicationRequest: {counts['MedicationRequest']}
- DocumentReference: {counts['DocumentReference']}
- Total bundle entries: {counts['BundleEntry']}

## Import examples

### 1) HAPI FHIR (transaction bundle)

```bash
curl -X POST \
  -H "Content-Type: application/fhir+json" \
  --data @data/fhir/hc01_synthetic_fhir_bundle.json \
  https://hapi.fhir.org/baseR4
```

### 2) Local FHIR server

```bash
curl -X POST \
  -H "Content-Type: application/fhir+json" \
  --data @data/fhir/hc01_synthetic_fhir_bundle.json \
  http://localhost:8080/fhir
```

## Notes

- All data is synthetic and deidentified.
- Observations use LOINC coding for vitals and labs used by `app/ehr.py` mapping.
- DocumentReference note content is stored in base64 text/plain attachments.
"""
    (FHIR_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    FHIR_DIR.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    resources = build_resources(dataset)
    bundle = to_transaction_bundle(resources)

    BUNDLE_PATH.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_ndjson(resources)
    write_readme(bundle, resources)

    print("FHIR synthetic export generated:")
    print(f"- {BUNDLE_PATH}")
    print(f"- {NDJSON_DIR}")
    print(f"- total entries: {len(bundle.get('entry', []))}")


if __name__ == "__main__":
    main()
