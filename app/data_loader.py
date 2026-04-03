"""MIMIC-III CSV data loader + synthetic dataset loader."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import cfg
from .models import PatientData, PatientVitals

log = logging.getLogger("HC01.data_loader")

# ─── Known MIMIC item-ID → standardised lab key mapping ────────────────────
# Covers MIMIC-III itemids; extend as needed.
_ITEMID_TO_KEY: Dict[int, str] = {
    50813: "lactate",
    50912: "creatinine",
    51301: "wbc",
    51265: "platelets",
    50885: "bilirubin",   # Bilirubin, Total
    51006: "bun",
    50960: "magnesium",
    50971: "potassium",
    50983: "sodium",
    51221: "hematocrit",
    51222: "hemoglobin",
    50931: "glucose",
    50821: "pao2",
    50820: "ph",
    50818: "paco2",
    51144: "bands",
    50889: "crp",         # C-Reactive Protein
    51244: "lymphocytes",
    51248: "mch",
    51279: "rbc",
}

# Reverse map: label substring → standardised key (used when itemid not in map)
_LABEL_TO_KEY: Dict[str, str] = {
    "lactate": "lactate",
    "creatinine": "creatinine",
    "white blood cell": "wbc",
    "wbc": "wbc",
    "platelet": "platelets",
    "bilirubin": "bilirubin",
    "urea nitrogen": "bun",
    "bun": "bun",
    "hemoglobin": "hemoglobin",
    "hematocrit": "hematocrit",
    "glucose": "glucose",
    "potassium": "potassium",
    "sodium": "sodium",
    "pao2": "pao2",
    "procalcitonin": "procalcitonin",
}


class MIMICDataLoader:
    """Load and process MIMIC-III CSV records."""

    def __init__(self) -> None:
        self._cache: Dict[str, pd.DataFrame] = {}
        self._label_map: Optional[Dict[int, str]] = None  # itemid -> lab key

    # ── CSV helpers ──────────────────────────────────────────────────────

    def _load_csv(self, key: str) -> Optional[pd.DataFrame]:
        if key in self._cache:
            return self._cache[key]
        path = cfg.MIMIC_CSVS.get(key)
        if not path or not os.path.exists(path):
            log.warning("CSV not found: %s → %s", key, path)
            return None
        df = pd.read_csv(path, low_memory=False, sep="|")
        df.columns = [c.lower() for c in df.columns]  # Normalise to lowercase
        self._cache[key] = df
        log.info("Loaded %s (%d rows)", key, len(df))
        return df

    def _build_label_map(self) -> Dict[int, str]:
        """Build itemid → standardised lab key from D_LABITEMS."""
        if self._label_map is not None:
            return self._label_map
        d_labs = self._load_csv("d_labitems")
        result: Dict[int, str] = dict(_ITEMID_TO_KEY)  # start with hardcoded
        if d_labs is not None:
            for _, row in d_labs.iterrows():
                iid = int(row["itemid"])
                if iid in result:
                    continue
                label = str(row.get("label", "")).lower()
                for substr, key in _LABEL_TO_KEY.items():
                    if substr in label:
                        result[iid] = key
                        break
        self._label_map = result
        return result

    # ── Patient listing ──────────────────────────────────────────────────

    def list_patients(self) -> List[Dict]:
        """Return all ICU stays with basic demographics."""
        icu = self._load_csv("icustays")
        pts = self._load_csv("patients")
        if icu is None or pts is None:
            return []

        merged = icu.merge(pts, on="subject_id", how="left")

        results = []
        for _, row in merged.iterrows():
            age = self._calc_age(row.get("dob"), row.get("intime"))
            results.append({
                "subject_id":     int(row["subject_id"]),
                "hadm_id":        int(row["hadm_id"]),
                "icustay_id":     int(row["icustay_id"]),
                "age":            age,
                "sex":            str(row.get("gender", "U"))[0].upper(),
                "first_careunit": str(row.get("first_careunit", "ICU")),
                "los_days":       round(float(row.get("los", 0)), 2),
                "admit_time":     str(row.get("intime", "")),
            })
        return results

    # ── Admission ────────────────────────────────────────────────────────

    def get_admission(self, subject_id: int, hadm_id: int) -> Optional[Dict]:
        icu = self._load_csv("icustays")
        pts = self._load_csv("patients")
        if icu is None or pts is None:
            return None

        stay = icu[(icu["subject_id"] == subject_id) & (icu["hadm_id"] == hadm_id)]
        if stay.empty:
            return None
        stay = stay.iloc[0]

        pat = pts[pts["subject_id"] == subject_id]
        if pat.empty:
            return None
        pat = pat.iloc[0]

        age = self._calc_age(pat.get("dob"), stay.get("intime"))
        return {
            "subject_id":     subject_id,
            "hadm_id":        hadm_id,
            "icustay_id":     int(stay.get("icustay_id", 0)),
            "age":            age,
            "sex":            str(pat.get("gender", "U"))[0].upper(),
            "first_careunit": str(stay.get("first_careunit", "ICU")),
            "los_days":       round(float(stay.get("los", 2.0)), 2),
            "intime":         str(stay.get("intime", "")),
            "outtime":        str(stay.get("outtime", "")),
        }

    @staticmethod
    def _calc_age(dob_raw, intime_raw) -> int:
        """Calculate age in years; defaults to 65 if data unavailable."""
        try:
            dob = pd.Timestamp(dob_raw)
            intime = pd.Timestamp(intime_raw)
            if pd.isna(dob) or pd.isna(intime):
                return 65
            age = (intime - dob).days // 365
            # MIMIC-III shifts DOBs >89 years old by 300 years
            if age > 150:
                age = 91
            return max(0, min(age, 130))
        except Exception:
            return 65

    # ── Notes ────────────────────────────────────────────────────────────

    def get_notes(self, subject_id: int, hadm_id: int) -> List[Dict]:
        notes = self._load_csv("noteevents")
        if notes is None:
            return []
        subset = notes[
            (notes["subject_id"] == subject_id) &
            (notes["hadm_id"] == hadm_id)
        ]
        results = []
        for _, row in subset.iterrows():
            text = str(row.get("text", ""))
            if not text or text == "nan":
                continue
            results.append({
                "time":   str(row.get("charttime", row.get("chartdate", ""))),
                "author": str(row.get("category", "Clinical Note")),
                "text":   text[:800],
            })
        return results[:20]  # Cap at 20 notes

    # ── Labs ─────────────────────────────────────────────────────────────

    def get_labs(self, subject_id: int, hadm_id: int) -> Dict[str, List[Dict]]:
        """Return labs as {key: [{t, v}]} sorted by time."""
        labs = self._load_csv("labevents")
        if labs is None:
            return {}

        subset = labs[
            (labs["subject_id"] == subject_id) &
            (labs["hadm_id"] == hadm_id)
        ].copy()

        if subset.empty:
            return {}

        lmap = self._build_label_map()
        result: Dict[str, List[Dict]] = {}

        for _, row in subset.iterrows():
            iid = int(row.get("itemid", 0))
            key = lmap.get(iid)
            if key is None:
                continue
            val = row.get("valuenum")
            if pd.isna(val):
                continue
            val = float(val)
            if val < 0 or val > 1_000_000:  # Sanity filter
                continue
            t = str(row.get("charttime", ""))
            result.setdefault(key, []).append({"t": t, "v": round(val, 3)})

        # Sort each series chronologically and keep last 8 readings
        for key in result:
            result[key].sort(key=lambda x: x["t"])
            result[key] = result[key][-8:]

        return result

    # ── Medications ──────────────────────────────────────────────────────

    def get_medications(self, subject_id: int, hadm_id: int) -> List[str]:
        presc = self._load_csv("prescriptions")
        if presc is None:
            return []
        subset = presc[
            (presc["subject_id"] == subject_id) &
            (presc["hadm_id"] == hadm_id)
        ]
        meds = []
        seen = set()
        for _, row in subset.iterrows():
            drug = str(row.get("drug_name_generic") or row.get("drug") or "").strip()
            if drug and drug != "nan" and drug.lower() not in seen:
                seen.add(drug.lower())
                dose  = str(row.get("dose_val_rx", "")).strip()
                unit  = str(row.get("dose_unit_rx", "")).strip()
                route = str(row.get("route", "")).strip()
                parts = [drug]
                if dose and dose != "nan":
                    parts.append(dose)
                if unit and unit != "nan":
                    parts.append(unit)
                if route and route not in ("nan", ""):
                    parts.append(f"({route})")
                meds.append(" ".join(parts))
        return meds[:30]  # Cap at 30 medications

    # ── Vitals synthesis ─────────────────────────────────────────────────

    def synthesize_vitals(self, subject_id: int, hadm_id: int, labs: Dict) -> PatientVitals:
        """
        Synthesise realistic ICU vitals.
        Seeded from subject/hadm IDs for reproducibility.
        Uses available lab context for physiological consistency.
        """
        seed = int(hashlib.md5(f"{subject_id}:{hadm_id}".encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)

        def latest_lab(key: str) -> Optional[float]:
            arr = labs.get(key, [])
            return arr[-1]["v"] if arr else None

        lactate = latest_lab("lactate") or 0.0
        creatinine = latest_lab("creatinine") or 1.0
        wbc = latest_lab("wbc") or 8.0

        # Severity-adjusted distributions
        sepsis_score = min(1.0, lactate / 4.0 + max(0, wbc - 12) / 20.0)

        hr    = float(np.clip(rng.normal(85 + 20 * sepsis_score, 12), 45, 155))
        sbp   = float(np.clip(rng.normal(120 - 20 * sepsis_score, 15), 75, 200))
        dbp   = float(np.clip(rng.normal(70 - 10 * sepsis_score, 10), 40, 115))
        map_  = round((sbp + 2 * dbp) / 3, 1)
        rr    = float(np.clip(rng.normal(16 + 6 * sepsis_score, 3), 8, 40))
        spo2  = float(np.clip(rng.normal(96 - 3 * sepsis_score, 1.5), 82, 100))
        temp  = float(np.clip(rng.normal(37.0 + 0.8 * sepsis_score, 0.5), 35.0, 41.0))
        gcs   = float(np.clip(rng.normal(13 - 2 * sepsis_score, 1.5), 3, 15))
        fio2  = float(np.clip(rng.normal(0.21 + 0.3 * sepsis_score, 0.05), 0.21, 1.0))

        # pao2 — estimated from SpO2 via simplified reverse pulse-ox
        pao2 = 0.0
        if spo2 >= 97:
            pao2 = float(np.clip(rng.normal(130, 15), 100, 200))
        elif spo2 >= 92:
            pao2 = float(np.clip(rng.normal(75, 10), 55, 100))
        else:
            pao2 = float(np.clip(rng.normal(55, 8), 40, 70))

        return PatientVitals(
            hr=round(hr, 1),
            bpSys=round(sbp, 1),
            bpDia=round(dbp, 1),
            map=round(map_, 1),
            rr=round(rr, 1),
            spo2=round(spo2, 1),
            temp=round(temp, 1),
            gcs=round(gcs),
            fio2=round(fio2, 3),
            pao2=round(pao2, 1),
        )

    # ── Lab enrichment ───────────────────────────────────────────────────

    def _enrich_labs(
        self,
        seed: int,
        base_labs: Dict[str, List[Dict]],
        n_timepoints: int = 6,
    ) -> Dict[str, List[Dict]]:
        """
        If a lab has only 1 real reading, synthesise a plausible 72-hour trajectory
        around that anchor value so agents can detect trends.
        Labs with 2+ real readings are left unchanged.
        """
        rng = np.random.default_rng(seed)
        result: Dict[str, List[Dict]] = {}

        # Reference: lab_key -> (typical_mean, typical_std, direction_bias)
        # direction_bias: +1 = tends to rise in sick patients, -1 = tends to fall
        _lab_dynamics: Dict[str, tuple] = {
            "lactate":    (1.8, 0.6,  +1),
            "creatinine": (1.2, 0.4,  +1),
            "wbc":        (12,  4,    +1),
            "platelets":  (220, 80,   -1),
            "bilirubin":  (1.0, 0.5,  +1),
            "bun":        (20,  10,   +1),
            "hemoglobin": (10,  2,    -1),
            "hematocrit": (32,  6,    -1),
            "sodium":     (138, 4,    0),
            "potassium":  (4.2, 0.6,  0),
            "glucose":    (130, 30,   +1),
            "pao2":       (90,  20,   -1),
        }

        for key, arr in base_labs.items():
            if len(arr) >= 3:
                result[key] = arr
                continue

            anchor = arr[-1]["v"] if arr else _lab_dynamics.get(key, (1.0, 0.2, 0))[0]
            anchor = float(anchor)
            _, std, bias = _lab_dynamics.get(key, (anchor, max(anchor * 0.1, 0.2), 0))

            # Build backward trajectory (older → newer readings)
            vals = []
            v = anchor
            for _ in range(n_timepoints - 1, -1, -1):
                noise = float(rng.normal(0, std * 0.15))
                drift = bias * std * 0.08
                vals.append(round(max(0.0, v - drift - noise), 3))
                v = v - drift - noise

            vals.reverse()
            # Replace last with real anchor
            vals[-1] = round(anchor, 3)

            base_time = arr[-1]["t"] if arr else "2024-01-01 06:00:00"
            result[key] = [
                {"t": base_time.replace(" ", "T") if "T" not in str(base_time) else str(base_time),
                 "v": round(float(v_), 3)}
                for v_ in vals
            ]

        return result

    # ── Full patient builder ─────────────────────────────────────────────

    def build_patient_data(self, subject_id: int, hadm_id: int) -> Optional[PatientData]:
        """
        Build a complete PatientData object from MIMIC CSVs.
        Returns None if the patient/admission is not found.
        """
        admission = self.get_admission(subject_id, hadm_id)
        if admission is None:
            log.warning("Patient %d / %d not found in CSVs", subject_id, hadm_id)
            return None

        notes    = self.get_notes(subject_id, hadm_id)
        raw_labs = self.get_labs(subject_id, hadm_id)
        meds     = self.get_medications(subject_id, hadm_id)

        # Enrich sparse lab series to 6-timepoint 72h trajectories
        seed = int(hashlib.md5(f"enrich:{subject_id}:{hadm_id}".encode()).hexdigest(), 16) % (2**31)
        labs = self._enrich_labs(seed, raw_labs)

        vitals   = self.synthesize_vitals(subject_id, hadm_id, labs)

        # Derive admission diagnosis from notes
        admit_diag = "ICU Admission"
        if notes:
            first_text = notes[0]["text"]
            # Grab first meaningful clause
            for phrase in first_text.split("."):
                phrase = phrase.strip()
                if len(phrase) > 10:
                    admit_diag = phrase[:120]
                    break

        # Estimate weight from height/demographics (MIMIC doesn't store weight in CSVs)
        rng = np.random.default_rng(subject_id % 10000)
        weight = float(np.clip(rng.normal(75, 12), 40, 160))

        return PatientData(
            id=f"{subject_id}-{hadm_id}",
            name=f"Patient {subject_id}",
            age=admission["age"],
            sex=admission["sex"],
            weight=round(weight, 1),
            daysInICU=admission["los_days"],
            admitDiag=admit_diag,
            vitals=vitals,
            labs=labs,
            medications=meds,
            notes=notes if notes else [{
                "time": admission["intime"],
                "author": "System",
                "text": f"Admitted to {admission['first_careunit']}. No notes available.",
            }],
        )


# ─── Module-level singleton ──────────────────────────────────────────────────
_loader: Optional[MIMICDataLoader] = None


def get_loader() -> MIMICDataLoader:
    global _loader
    if _loader is None:
        _loader = MIMICDataLoader()
    return _loader


# ─── Synthetic dataset loader ─────────────────────────────────────────────────

_SYNTHETIC_JSON = Path(__file__).resolve().parent.parent / "data" / "hc01_synthetic_icu_dataset.json"
_SYNTHETIC_CACHE: Optional[List[Dict]] = None


def _load_synthetic_raw() -> List[Dict]:
    global _SYNTHETIC_CACHE
    if _SYNTHETIC_CACHE is not None:
        return _SYNTHETIC_CACHE
    if not _SYNTHETIC_JSON.exists():
        log.warning("Synthetic dataset not found at %s", _SYNTHETIC_JSON)
        return []
    with _SYNTHETIC_JSON.open() as fh:
        data = json.load(fh)
    _SYNTHETIC_CACHE = data.get("patients", []) if isinstance(data, dict) else data
    log.info("Loaded %d synthetic ICU cases", len(_SYNTHETIC_CACHE))
    return _SYNTHETIC_CACHE


def list_synthetic_patients(
    case_type: Optional[str] = None,
    trajectory: Optional[str] = None,
) -> List[Dict]:
    """
    Return lightweight summary records for the synthetic dataset.
    Optionally filter by caseType ('sepsis','ards','aki','cardiac','neuro','stable')
    or trajectory ('worsening','improving','stable').
    """
    raw = _load_synthetic_raw()
    results = []
    for p in raw:
        if case_type and p.get("caseType", "").lower() != case_type.lower():
            continue
        if trajectory and p.get("trajectory", "").lower() != trajectory.lower():
            continue
        scores = p.get("latestScores", {})
        results.append({
            "id":          p["id"],
            "subject_id":  p.get("subject_id", p["id"]),
            "name":        p.get("name", "Unknown"),
            "age":         p.get("age", 0),
            "sex":         p.get("sex", "U"),
            "admitDiag":   p.get("admitDiag", ""),
            "caseType":    p.get("caseType", ""),
            "trajectory":  p.get("trajectory", ""),
            "daysInICU":   p.get("daysInICU", 0),
            "sofa":        scores.get("sofa", 0),
            "news2":       scores.get("news2", 0),
            "labels":      p.get("labels", []),
            "mortality_risk": p.get("outcomes", {}).get("mortality_risk", "unknown"),
        })
    return results


def get_synthetic_patient(patient_id: str) -> Optional[PatientData]:
    """
    Return a PatientData object for a synthetic patient by id (e.g. 'P001').
    The JSON schema is PatientData-compatible so we can pass it directly.
    """
    raw = _load_synthetic_raw()
    record = next((p for p in raw if p["id"] == patient_id), None)
    if record is None:
        return None
    # Strip non-PatientData fields
    allowed = {"id", "name", "age", "sex", "weight", "daysInICU", "admitDiag",
               "vitals", "labs", "medications", "notes"}
    clean = {k: v for k, v in record.items() if k in allowed}
    # subject_id is required by PatientData but stored separately in synthetic data
    if "id" not in clean:
        clean["id"] = patient_id
    try:
        return PatientData(**clean)
    except Exception as exc:
        log.error("Failed to parse synthetic patient %s: %s", patient_id, exc)
        return None


# ─── Demo patients ────────────────────────────────────────────────────────────
# Pre-built rich clinical scenarios that showcase all pipeline features.

def get_demo_patients() -> List[PatientData]:
    """Return 3 demo ICU patients with realistic multi-timepoint data."""
    from .models import PatientVitals

    return [
        # ── Demo 1: Septic Shock + Vancomycin AKI conflict ─────────────────
        PatientData(
            id="demo-sepsis-001",
            name="Demo Patient A",
            age=67,
            sex="M",
            weight=82.0,
            daysInICU=1.4,
            admitDiag="Septic shock — presumed gram-negative source (UTI vs pneumonia)",
            vitals=PatientVitals(
                hr=118, bpSys=84, bpDia=48, map=60.0,
                rr=26, spo2=91, temp=38.9, gcs=12, fio2=0.50, pao2=58.0,
            ),
            labs={
                "lactate":    [{"t": "T-72h", "v": 1.1}, {"t": "T-48h", "v": 1.8},
                               {"t": "T-24h", "v": 2.6}, {"t": "T-12h", "v": 3.5},
                               {"t": "T-6h",  "v": 4.2}, {"t": "T-0h",  "v": 5.1}],
                "creatinine": [{"t": "T-72h", "v": 0.9}, {"t": "T-48h", "v": 1.2},
                               {"t": "T-24h", "v": 1.8}, {"t": "T-12h", "v": 2.6},
                               {"t": "T-6h",  "v": 3.2}, {"t": "T-0h",  "v": 3.9}],
                "wbc":        [{"t": "T-72h", "v": 8.2}, {"t": "T-48h", "v": 11.4},
                               {"t": "T-24h", "v": 15.2}, {"t": "T-12h", "v": 18.7},
                               {"t": "T-6h",  "v": 21.3}, {"t": "T-0h",  "v": 23.1}],
                "platelets":  [{"t": "T-72h", "v": 210}, {"t": "T-48h", "v": 185},
                               {"t": "T-24h", "v": 140}, {"t": "T-12h", "v": 98},
                               {"t": "T-6h",  "v": 64},  {"t": "T-0h",  "v": 48}],
                "procalcitonin": [{"t": "T-48h", "v": 2.1}, {"t": "T-24h", "v": 8.4},
                                  {"t": "T-0h",  "v": 14.2}],
            },
            medications=[
                "Vancomycin 1250 mg (IV)",
                "Piperacillin-Tazobactam 4.5g q6h (IV)",
                "Norepinephrine 0.15 mcg/kg/min (IV)",
                "Normal Saline 500 mL bolus (IV)",
                "Pantoprazole 40 mg (IV)",
            ],
            notes=[{
                "time": "T-6h",
                "author": "ICU Attending",
                "text": (
                    "67M admitted with hypotension, fever, and confusion. Temp 38.9C, HR 118, "
                    "BP 84/48. Urine output 15 mL/hr. UA positive for bacteria and WBCs. "
                    "CXR: bilateral infiltrates. Started on vancomycin + pip-tazo. "
                    "Norepinephrine titrating. Creatinine rising — AKI concern. "
                    "Lactate climbing despite 2L IVF. Source unclear — UTI vs pneumonia."
                ),
            }],
        ),

        # ── Demo 2: ARDS + Trajectory toward critical ──────────────────────
        PatientData(
            id="demo-ards-002",
            name="Demo Patient B",
            age=54,
            sex="F",
            weight=68.0,
            daysInICU=3.2,
            admitDiag="ARDS secondary to COVID-19 pneumonia — moderate severity",
            vitals=PatientVitals(
                hr=102, bpSys=108, bpDia=62, map=77.3,
                rr=30, spo2=88, temp=37.6, gcs=13, fio2=0.80, pao2=62.0,
            ),
            labs={
                "pao2":       [{"t": "T-72h", "v": 195}, {"t": "T-48h", "v": 140},
                               {"t": "T-24h", "v": 105}, {"t": "T-12h", "v": 82},
                               {"t": "T-6h",  "v": 68},  {"t": "T-0h",  "v": 62}],
                "lactate":    [{"t": "T-48h", "v": 1.4}, {"t": "T-24h", "v": 1.9},
                               {"t": "T-0h",  "v": 2.3}],
                "creatinine": [{"t": "T-72h", "v": 0.8}, {"t": "T-48h", "v": 1.0},
                               {"t": "T-24h", "v": 1.3}, {"t": "T-0h",  "v": 1.6}],
                "wbc":        [{"t": "T-72h", "v": 14.1}, {"t": "T-48h", "v": 16.2},
                               {"t": "T-24h", "v": 17.8}, {"t": "T-0h",  "v": 19.5}],
                "bilirubin":  [{"t": "T-48h", "v": 0.8}, {"t": "T-24h", "v": 1.3},
                               {"t": "T-0h",  "v": 2.1}],
            },
            medications=[
                "Dexamethasone 6 mg (IV)",
                "Remdesivir 200 mg (IV)",
                "Enoxaparin 40 mg (SC)",
                "Propofol infusion (IV)",
                "Fentanyl infusion (IV)",
            ],
            notes=[{
                "time": "T-8h",
                "author": "Pulmonology",
                "text": (
                    "54F with known COVID-19, Day 3 ICU. Worsening hypoxemia despite prone "
                    "positioning. FiO2 now 80%, SpO2 88%. P/F ratio 77.5 — severe ARDS. "
                    "Bilateral infiltrates on CXR. Currently intubated, ARDSNet protocol: "
                    "TV 408 mL (6 mL/kg IBW). Plateau pressure 28 cmH2O. "
                    "Bilirubin trending up. Renal function borderline. Dexamethasone day 3."
                ),
            }],
        ),

        # ── Demo 3: Multi-organ dysfunction — AKI + outlier labs ───────────
        PatientData(
            id="demo-mods-003",
            name="Demo Patient C",
            age=78,
            sex="M",
            weight=71.0,
            daysInICU=5.8,
            admitDiag="Post-operative sepsis with multi-organ dysfunction (abdominal surgery)",
            vitals=PatientVitals(
                hr=94, bpSys=96, bpDia=55, map=68.7,
                rr=22, spo2=94, temp=38.2, gcs=11, fio2=0.40, pao2=72.0,
            ),
            labs={
                "creatinine": [{"t": "T-96h", "v": 1.1}, {"t": "T-72h", "v": 1.4},
                               {"t": "T-48h", "v": 2.2}, {"t": "T-24h", "v": 3.1},
                               {"t": "T-12h", "v": 3.8}, {"t": "T-0h",  "v": 4.6}],
                "bilirubin":  [{"t": "T-72h", "v": 1.8}, {"t": "T-48h", "v": 3.2},
                               {"t": "T-24h", "v": 5.4}, {"t": "T-12h", "v": 7.1},
                               {"t": "T-0h",  "v": 8.9}],
                "platelets":  [{"t": "T-72h", "v": 180}, {"t": "T-48h", "v": 130},
                               {"t": "T-24h", "v": 85},  {"t": "T-12h", "v": 58},
                               {"t": "T-0h",  "v": 38}],
                "lactate":    [{"t": "T-48h", "v": 2.8}, {"t": "T-24h", "v": 2.2},
                               {"t": "T-12h", "v": 1.9}, {"t": "T-0h",  "v": 1.6}],
                "wbc":        [{"t": "T-72h", "v": 18.2}, {"t": "T-48h", "v": 22.1},
                               # Outlier: single spike that is statistically abnormal
                               {"t": "T-24h", "v": 38.5}, {"t": "T-12h", "v": 23.4},
                               {"t": "T-0h",  "v": 24.1}],
                "bun":        [{"t": "T-48h", "v": 28},  {"t": "T-24h", "v": 45},
                               {"t": "T-0h",  "v": 72}],
            },
            medications=[
                "Meropenem 1g q8h (IV)",
                "Vancomycin 1000 mg q12h (IV)",
                "Furosemide 40 mg (IV)",
                "Norepinephrine 0.08 mcg/kg/min (IV)",
                "Insulin infusion (IV)",
                "Metoprolol 5 mg (IV)",
            ],
            notes=[{
                "time": "T-4h",
                "author": "ICU Fellow",
                "text": (
                    "78M Day 5 post abdominal surgery. Persistent sepsis with progressive "
                    "multi-organ dysfunction. Creatinine rising to 4.6 — KDIGO Stage 3 AKI, "
                    "nephrology consulted for possible RRT. Bilirubin climbing — hepatic "
                    "dysfunction likely ischaemic. Platelet count dropping — DIC vs sepsis. "
                    "WBC spike yesterday (38.5) not reproducible — lab error suspected, "
                    "redraw sent. Lactate trending down on norepinephrine. "
                    "Vancomycin levels high — holding next dose."
                ),
            }],
        ),
    ]
