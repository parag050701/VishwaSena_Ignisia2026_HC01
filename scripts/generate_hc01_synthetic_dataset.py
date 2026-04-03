#!/usr/bin/env python3
"""Generate a synthetic ICU dataset for HC01.

The output is aligned with the app/backend patient schema:
- id, name, age, sex, weight, daysInICU, admitDiag
- vitals, labs, medications, notes
- labels, outcomes, latest scores for convenience

This script writes:
- data/hc01_synthetic_icu_dataset.json
- data/hc01_synthetic_icu_cases.csv
"""

from __future__ import annotations

import csv
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JSON_PATH = DATA_DIR / "hc01_synthetic_icu_dataset.json"
CSV_PATH = DATA_DIR / "hc01_synthetic_icu_cases.csv"
SEED = 20260403
TARGET_COUNT = 120

# Make repo modules importable when running the script directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from app.models import PatientData  # type: ignore
    from app.scoring import calc_news2, calc_sofa  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without app deps
    PatientData = None
    calc_news2 = None
    calc_sofa = None


FIRST_NAMES = [
    "Amina", "Maya", "Noah", "Elena", "Kofi", "Sara", "Omar", "Lina",
    "Iris", "Mateo", "Zuri", "Arjun", "Nadia", "Leo", "Priya", "Kenji",
    "Hana", "Imani", "Jonah", "Fatima", "Avery", "Sofia", "Daniel", "Mei",
]
LAST_NAMES = [
    "Diallo", "Rodriguez", "Okafor", "Patel", "Singh", "Nakamura", "Martinez",
    "Chen", "Osei", "Sato", "Kim", "Brown", "Nguyen", "Singh", "Garcia",
    "Mensah", "Ali", "Silva", "Hassan", "Williams", "Rossi", "Walker", "Bello",
    "Hernandez",
]

NOTE_AUTHORS = [
    "Dr. A. Patel (Surgical ICU)",
    "Dr. R. Martinez (Evening)",
    "Dr. S. Sharma (Attending)",
    "Nurse Day Shift",
    "Nurse Night Shift",
    "Resident Team",
    "ICU Fellow",
    "Nephrology Consult",
    "Respiratory Therapy",
]

BASE_SEEDS: List[Dict[str, Any]] = [
    {
        "id": "P001",
        "subject_id": "HC01-P001",
        "name": "Mr. Chen Wei",
        "age": 67,
        "sex": "M",
        "weight": 82,
        "daysInICU": 4.8,
        "lastLabHoursAgo": 7,
        "admitDiag": "Suspected Sepsis / Source Under Investigation",
        "caseType": "sepsis",
        "trajectory": "worsening",
        "labels": ["sepsis", "septic_shock_risk", "aki", "oliguria"],
        "vitals": {"hr": 118, "bpSys": 90, "bpDia": 58, "map": 69, "rr": 24, "spo2": 93, "temp": 38.9, "gcs": 13, "fio2": 0.28, "pao2": 72},
        "labs": {
            "wbc": [{"t": "08:00", "v": 4.2}, {"t": "14:00", "v": 8.1}, {"t": "20:00", "v": 14.3}, {"t": "06:00+", "v": 18.7}],
            "lactate": [{"t": "08:00", "v": 0.9}, {"t": "14:00", "v": 1.4}, {"t": "20:00", "v": 2.1}, {"t": "06:00+", "v": 2.8}],
            "creatinine": [{"t": "08:00", "v": 0.8}, {"t": "14:00", "v": 1.1}, {"t": "20:00", "v": 1.4}, {"t": "06:00+", "v": 4.2, "outlier": True}],
            "platelets": [{"t": "08:00", "v": 210}, {"t": "06:00+", "v": 145}],
            "bilirubin": [{"t": "08:00", "v": 0.8}, {"t": "06:00+", "v": 1.2}],
            "procalcitonin": [{"t": "08:00", "v": 0.4}, {"t": "14:00", "v": 2.8}, {"t": "06:00+", "v": 11.2}],
            "bun": [{"t": "08:00", "v": 18}, {"t": "14:00", "v": 26}, {"t": "06:00+", "v": 42}],
        },
        "refs": {
            "wbc": {"low": 4.5, "high": 11.0, "unit": "x10^3/uL"},
            "lactate": {"low": 0.5, "high": 2.0, "unit": "mmol/L"},
            "creatinine": {"low": 0.6, "high": 1.2, "unit": "mg/dL"},
            "platelets": {"low": 150, "high": 400, "unit": "x10^3/uL"},
            "bilirubin": {"low": 0.1, "high": 1.2, "unit": "mg/dL"},
            "procalcitonin": {"low": 0.0, "high": 0.5, "unit": "ng/mL"},
            "bun": {"low": 7, "high": 20, "unit": "mg/dL"},
        },
        "medications": [
            "Piperacillin-Tazobactam 4.5g IV q6h",
            "Vancomycin 1.5g IV q12h",
            "Norepinephrine 0.05 mcg/kg/min",
            "Normal Saline 125 mL/hr",
        ],
        "notes": [
            {
                "time": "08:30",
                "author": "Dr. S. Sharma (Attending)",
                "text": "Patient with chills, tachycardia, fever, and borderline blood pressure. Empiric broad-spectrum antibiotics started. Blood cultures drawn.",
            },
            {
                "time": "14:00",
                "author": "Nurse P. Okafor (Day Shift)",
                "text": "Patient increasingly confused. GCS down to 13. BP trending lower. Fluids increased and attending notified.",
            },
            {
                "time": "20:00",
                "author": "Dr. R. Martinez (Evening)",
                "text": "Persistent fever and rising WBC. Lactate trending up. Oxygen requirement increased. ICU review requested.",
            },
            {
                "time": "06:00+",
                "author": "Night Nurse L. Chen",
                "text": "Norepinephrine started overnight. Urine output low. Creatinine returned markedly elevated versus prior. Repeat labs ordered.",
            },
        ],
        "outcomes": {"vasopressor_started": True, "intubated": False, "dialysis": False, "mortality_risk": "high"},
    },
    {
        "id": "P002",
        "subject_id": "HC01-P002",
        "name": "Ms. Maria Rodriguez",
        "age": 54,
        "sex": "F",
        "weight": 68,
        "daysInICU": 2.1,
        "lastLabHoursAgo": 4,
        "admitDiag": "Post-op Day 2 - Bowel Resection - ARDS Developing",
        "caseType": "ards",
        "trajectory": "worsening",
        "labels": ["ards", "respiratory_failure", "post_op_inflammation", "early_aki"],
        "vitals": {"hr": 102, "bpSys": 108, "bpDia": 65, "map": 79, "rr": 28, "spo2": 88, "temp": 37.6, "gcs": 14, "fio2": 0.60, "pao2": 62},
        "labs": {
            "wbc": [{"t": "Day 1", "v": 9.8}, {"t": "Day 2", "v": 12.1}, {"t": "Today", "v": 15.4}],
            "lactate": [{"t": "Day 1", "v": 1.2}, {"t": "Day 2", "v": 1.6}, {"t": "Today", "v": 2.3}],
            "creatinine": [{"t": "Day 1", "v": 0.9}, {"t": "Day 2", "v": 1.3}, {"t": "Today", "v": 1.8}],
            "platelets": [{"t": "Day 1", "v": 180}, {"t": "Today", "v": 112}],
            "bilirubin": [{"t": "Day 1", "v": 1.4}, {"t": "Today", "v": 2.8}],
            "pao2": 62,
        },
        "refs": {
            "wbc": {"low": 4.5, "high": 11.0, "unit": "x10^3/uL"},
            "lactate": {"low": 0.5, "high": 2.0, "unit": "mmol/L"},
            "creatinine": {"low": 0.5, "high": 1.1, "unit": "mg/dL"},
            "platelets": {"low": 150, "high": 400, "unit": "x10^3/uL"},
            "bilirubin": {"low": 0.1, "high": 1.2, "unit": "mg/dL"},
        },
        "medications": ["Meropenem 1g IV q8h", "Furosemide 40mg IV q12h", "Propofol infusion", "Fentanyl PCA", "Pantoprazole 40mg IV OD"],
        "notes": [
            {"time": "Post-op Day 1", "author": "Dr. A. Patel (Surgical ICU)", "text": "Post bowel resection, hemostasis adequate. Patient extubated in OR. Some abdominal distension noted."},
            {"time": "Post-op Day 2 AM", "author": "Dr. A. Patel", "text": "Oxygen requirement increasing overnight. Bilateral crackles present. CXR shows bilateral infiltrates - ARDS pattern suspected."},
            {"time": "Today 06:00", "author": "Nurse ICU B. Osei", "text": "Re-intubated at 02:30. FiO2 60 percent, PEEP 8. Urine output decreasing. Bilirubin up. Patient sedated and not following commands."},
        ],
        "outcomes": {"vasopressor_started": False, "intubated": True, "dialysis": False, "mortality_risk": "moderate_high"},
    },
    {
        "id": "P003",
        "subject_id": "HC01-P003",
        "name": "Mr. Kofi Okafor",
        "age": 72,
        "sex": "M",
        "weight": 80,
        "daysInICU": 3.5,
        "lastLabHoursAgo": 2,
        "admitDiag": "AKI Stage 2 - Monitoring - Post-Contrast",
        "caseType": "aki",
        "trajectory": "improving",
        "labels": ["aki", "improving", "ward_transfer_candidate"],
        "vitals": {"hr": 78, "bpSys": 132, "bpDia": 82, "map": 99, "rr": 16, "spo2": 97, "temp": 37.1, "gcs": 15, "fio2": 0.21, "pao2": 95},
        "labs": {
            "wbc": [{"t": "Day 1", "v": 8.2}, {"t": "Day 2", "v": 7.9}, {"t": "Today", "v": 7.4}],
            "lactate": [{"t": "Day 1", "v": 1.3}, {"t": "Day 2", "v": 1.1}, {"t": "Today", "v": 0.9}],
            "creatinine": [{"t": "Day 1", "v": 2.8}, {"t": "Day 2", "v": 2.4}, {"t": "Today", "v": 2.1}],
            "platelets": [{"t": "Day 1", "v": 195}, {"t": "Today", "v": 210}],
            "bilirubin": [{"t": "Day 1", "v": 0.9}, {"t": "Today", "v": 1.0}],
            "pao2": 95,
        },
        "refs": {
            "wbc": {"low": 4.5, "high": 11.0, "unit": "x10^3/uL"},
            "lactate": {"low": 0.5, "high": 2.0, "unit": "mmol/L"},
            "creatinine": {"low": 0.7, "high": 1.3, "unit": "mg/dL"},
            "platelets": {"low": 150, "high": 400, "unit": "x10^3/uL"},
            "bilirubin": {"low": 0.1, "high": 1.2, "unit": "mg/dL"},
        },
        "medications": ["IV Fluids NS 75 mL/hr", "Amlodipine 5mg PO OD", "Aspirin 100mg PO OD"],
        "notes": [
            {"time": "Day 1", "author": "Dr. B. Singh (Nephrology)", "text": "AKI after contrast CT. Creatinine peaked at 2.8. Baseline around 1.0. Holding nephrotoxins. Hydration started."},
            {"time": "Day 2", "author": "Dr. B. Singh", "text": "Creatinine trending down. Urine output recovering. Good response to fluids. Continue monitoring."},
            {"time": "Today", "author": "Nurse Day Shift", "text": "Patient alert, mobilising, and tolerating oral intake. O2 not required. Potential discharge to ward in 24-48 hours if trend continues."},
        ],
        "outcomes": {"vasopressor_started": False, "intubated": False, "dialysis": False, "mortality_risk": "low"},
    },
]

ARCHETYPES = {
    "sepsis": {
        "admit_diags": [
            "Septic Shock - Source Control Pending",
            "Sepsis with Suspected Pneumonia",
            "Urosepsis with AKI Risk",
            "Bacteremia - Hemodynamic Monitoring",
        ],
        "label_pool": ["sepsis", "shock_risk", "organ_dysfunction", "cultures_sent"],
        "trajectory_pool": ["worsening", "worsening", "worsening", "improving"],
        "age_range": (46, 86),
        "weight_range": (52, 108),
        "last_lab_range": (2, 9),
        "vitals": {
            "hr": (104, 138), "bpSys": (82, 108), "bpDia": (44, 68), "map": (55, 78),
            "rr": (20, 32), "spo2": (86, 96), "temp": (38.0, 40.2), "gcs": (11, 15), "fio2": (0.24, 0.50), "pao2": (58, 84),
        },
        "lab_bases": {
            "wbc": (11.0, 22.0), "lactate": (1.6, 4.8), "creatinine": (0.9, 4.5),
            "platelets": (70, 220), "bilirubin": (0.7, 2.6), "procalcitonin": (1.0, 18.0), "bun": (22, 62),
        },
        "meds": [
            ["Piperacillin-Tazobactam 4.5g IV q6h", "Vancomycin 1.5g IV q12h", "Norepinephrine 0.05 mcg/kg/min", "Normal Saline 125 mL/hr"],
            ["Meropenem 1g IV q8h", "Vasopressin 0.03 units/min", "Lactated Ringers 150 mL/hr"],
            ["Cefepime 2g IV q8h", "Norepinephrine infusion", "Albumin 25% 100 mL"],
        ],
        "note_templates": [
            "Fever, tachycardia, and worsening hypotension. Broad-spectrum antibiotics started. Blood cultures obtained.",
            "Mentation is worse. Perfusion remains borderline despite fluids. Repeat lactate and cultures pending.",
            "Pressor support required. Urine output falling. Repeat labs show rising inflammatory markers.",
        ],
        "outcomes": {"vasopressor_started": True, "intubated": False, "dialysis": False},
    },
    "ards": {
        "admit_diags": [
            "ARDS After Abdominal Surgery",
            "Pneumonia With Acute Hypoxemic Respiratory Failure",
            "Severe Hypoxemia - Ventilator Escalation",
            "Post-op Pulmonary Complication - ARDS",
        ],
        "label_pool": ["ards", "respiratory_failure", "hypoxemia", "ventilator_support"],
        "trajectory_pool": ["worsening", "worsening", "stable", "improving"],
        "age_range": (34, 81),
        "weight_range": (48, 118),
        "last_lab_range": (1, 6),
        "vitals": {
            "hr": (88, 126), "bpSys": (94, 132), "bpDia": (54, 82), "map": (66, 94),
            "rr": (24, 38), "spo2": (82, 94), "temp": (36.2, 38.4), "gcs": (9, 15), "fio2": (0.35, 0.80), "pao2": (52, 86),
        },
        "lab_bases": {
            "wbc": (9.0, 18.0), "lactate": (1.0, 2.8), "creatinine": (0.8, 2.4),
            "platelets": (90, 220), "bilirubin": (0.8, 3.5), "procalcitonin": (0.2, 3.8), "bun": (14, 40),
        },
        "meds": [
            ["Meropenem 1g IV q8h", "Propofol infusion", "Fentanyl infusion", "Pantoprazole 40mg IV OD"],
            ["Ceftriaxone 2g IV q24h", "Dexmedetomidine infusion", "Furosemide 40mg IV q12h"],
            ["Norepinephrine 0.03 mcg/kg/min", "Low tidal volume ventilation", "Vancomycin 1g IV q12h"],
        ],
        "note_templates": [
            "Oxygen requirement increased overnight. CXR shows bilateral infiltrates. ARDS suspected.",
            "Patient required escalation to ventilatory support. SpO2 remains low despite higher FiO2.",
            "Lung protective ventilation started. Bilateral crackles and worsening oxygenation documented.",
        ],
        "outcomes": {"vasopressor_started": False, "intubated": True, "dialysis": False},
    },
    "aki": {
        "admit_diags": [
            "AKI Stage 2 - Post Contrast",
            "AKI on CKD - Fluid Responsive",
            "Renal Injury After Sepsis Recovery",
            "Oliguria With Rising Creatinine",
        ],
        "label_pool": ["aki", "renal_dysfunction", "oliguria", "nephrotoxin_review"],
        "trajectory_pool": ["improving", "worsening", "stable", "improving"],
        "age_range": (48, 88),
        "weight_range": (50, 110),
        "last_lab_range": (1, 5),
        "vitals": {
            "hr": (62, 96), "bpSys": (106, 148), "bpDia": (62, 90), "map": (74, 106),
            "rr": (12, 22), "spo2": (92, 99), "temp": (36.1, 37.8), "gcs": (13, 15), "fio2": (0.21, 0.35), "pao2": (78, 108),
        },
        "lab_bases": {
            "wbc": (6.0, 12.0), "lactate": (0.8, 2.1), "creatinine": (1.6, 5.6),
            "platelets": (140, 290), "bilirubin": (0.5, 1.6), "procalcitonin": (0.1, 1.4), "bun": (24, 74),
        },
        "meds": [
            ["IV Fluids NS 75 mL/hr", "Avoid nephrotoxins", "Amlodipine 5mg PO OD"],
            ["Balanced crystalloids", "Strict I&O", "Aspirin 100mg PO OD"],
            ["Furosemide 40mg IV q12h", "Renal dose adjustment review", "Pantoprazole 40mg IV OD"],
        ],
        "note_templates": [
            "Creatinine remains elevated but is trending down. Urine output improving with hydration.",
            "Nephrology recommends nephrotoxin avoidance and repeat renal labs. Patient hemodynamically stable.",
            "Oliguria has improved. Renal function still abnormal but trajectory is favorable.",
        ],
        "outcomes": {"vasopressor_started": False, "intubated": False, "dialysis": False},
    },
    "cardiac": {
        "admit_diags": [
            "Cardiogenic Shock Surveillance",
            "Acute Decompensated Heart Failure",
            "Post-MI Hemodynamic Monitoring",
            "Volume Overload With Pressor Titration",
        ],
        "label_pool": ["cardiogenic", "heart_failure", "perfusion_risk", "pressor_titration"],
        "trajectory_pool": ["worsening", "stable", "improving", "worsening"],
        "age_range": (42, 90),
        "weight_range": (58, 120),
        "last_lab_range": (1, 6),
        "vitals": {
            "hr": (88, 132), "bpSys": (78, 118), "bpDia": (48, 76), "map": (58, 84),
            "rr": (16, 30), "spo2": (88, 97), "temp": (36.0, 38.0), "gcs": (12, 15), "fio2": (0.21, 0.45), "pao2": (64, 96),
        },
        "lab_bases": {
            "wbc": (7.0, 14.0), "lactate": (1.2, 3.2), "creatinine": (0.9, 2.8),
            "platelets": (110, 240), "bilirubin": (0.6, 2.0), "procalcitonin": (0.1, 2.0), "bun": (18, 44),
        },
        "meds": [
            ["Dobutamine infusion", "Furosemide 40mg IV q12h", "Aspirin 100mg PO OD"],
            ["Norepinephrine 0.03 mcg/kg/min", "Diuretics titration", "Metoprolol 25mg PO BID"],
            ["Nitroglycerin infusion", "Heparin infusion", "Pantoprazole 40mg IV OD"],
        ],
        "note_templates": [
            "Hypotension and tachycardia improved with hemodynamic support. Echo suggests reduced forward flow.",
            "Fluid balance negative after diuresis. Oxygenation improved slightly but perfusion remains borderline.",
            "Monitoring for cardiogenic shock continues. Pressor/inotrope requirements being titrated.",
        ],
        "outcomes": {"vasopressor_started": True, "intubated": False, "dialysis": False},
    },
    "neuro": {
        "admit_diags": [
            "Metabolic Encephalopathy - ICU Monitoring",
            "Post-Stroke Neuro Watch",
            "Seizure Evaluation With ICU Observation",
            "Altered Mental Status - Rule Out Infection",
        ],
        "label_pool": ["neuro", "encephalopathy", "airway_watch", "delirium"],
        "trajectory_pool": ["stable", "improving", "worsening", "stable"],
        "age_range": (29, 87),
        "weight_range": (49, 116),
        "last_lab_range": (1, 8),
        "vitals": {
            "hr": (64, 108), "bpSys": (108, 156), "bpDia": (64, 96), "map": (78, 112),
            "rr": (10, 24), "spo2": (94, 100), "temp": (36.0, 38.2), "gcs": (7, 15), "fio2": (0.21, 0.30), "pao2": (78, 110),
        },
        "lab_bases": {
            "wbc": (5.5, 13.0), "lactate": (0.8, 2.1), "creatinine": (0.6, 1.8),
            "platelets": (130, 300), "bilirubin": (0.4, 1.4), "procalcitonin": (0.1, 1.1), "bun": (11, 32),
        },
        "meds": [
            ["Levetiracetam 1g IV BID", "Thiamine 100mg IV daily", "Dextrose 5% infusion"],
            ["Low-dose sedation hold", "Aspirin 100mg PO OD", "Omeprazole 20mg PO OD"],
            ["Hypertonic saline PRN", "Head-of-bed elevation", "Airway observation"],
        ],
        "note_templates": [
            "Neurologic checks stable. GCS unchanged. Airway remains protected and hemodynamics stable.",
            "Intermittent confusion but no focal deficit. Continue close neuro observations and delirium prevention.",
            "Mentation slowly improving. Able to follow simple commands and participate in exam.",
        ],
        "outcomes": {"vasopressor_started": False, "intubated": False, "dialysis": False},
    },
    "stable": {
        "admit_diags": [
            "ICU Observation - Improving",
            "Post-Procedure Monitoring",
            "Stepdown Candidate",
            "Stable Medical Admission",
        ],
        "label_pool": ["stable", "improving", "ward_transfer_candidate", "low_risk"],
        "trajectory_pool": ["improving", "stable", "improving", "stable"],
        "age_range": (24, 89),
        "weight_range": (48, 118),
        "last_lab_range": (1, 8),
        "vitals": {
            "hr": (58, 92), "bpSys": (110, 146), "bpDia": (64, 90), "map": (78, 104),
            "rr": (10, 20), "spo2": (95, 100), "temp": (36.0, 37.8), "gcs": (14, 15), "fio2": (0.21, 0.28), "pao2": (88, 110),
        },
        "lab_bases": {
            "wbc": (4.8, 9.8), "lactate": (0.8, 1.8), "creatinine": (0.6, 1.4),
            "platelets": (160, 320), "bilirubin": (0.3, 1.2), "procalcitonin": (0.05, 0.9), "bun": (9, 26),
        },
        "meds": [
            ["Maintenance IV fluids", "VTE prophylaxis", "Acetaminophen PRN"],
            ["Oral antihypertensive review", "Pantoprazole 40mg PO OD", "Mobility plan"],
            ["Aspirin 100mg PO OD", "Insulin sliding scale", "Discharge medication reconciliation"],
        ],
        "note_templates": [
            "Patient comfortable, mobilising, and tolerating oral intake. No acute concerns overnight.",
            "Labs improving and vitals stable. Continue routine monitoring with possible ward transfer.",
            "Pain controlled, oxygen not required, and renal/liver function acceptable for stepdown planning.",
        ],
        "outcomes": {"vasopressor_started": False, "intubated": False, "dialysis": False},
    },
}

TIMEPOINTS = ["Day 1", "Day 2", "Today"]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round_sig(value: float) -> float:
    if value >= 10:
        return round(value, 1)
    return round(value, 2)


def make_name(idx: int) -> str:
    if idx <= len(BASE_SEEDS):
        return BASE_SEEDS[idx - 1]["name"]
    first = FIRST_NAMES[(idx - 1) % len(FIRST_NAMES)]
    last = LAST_NAMES[((idx - 1) // len(FIRST_NAMES)) % len(LAST_NAMES)]
    prefix = "Mr." if idx % 2 else "Ms."
    return f"{prefix} {first} {last}"


def make_series(rng: random.Random, start: float, end: float, points: int = 3, jitter: float = 0.08) -> List[float]:
    if points < 2:
        return [round_sig(start)]
    series = []
    for i in range(points):
        frac = i / (points - 1)
        value = start + (end - start) * frac
        wiggle = value * jitter * (rng.random() * 2 - 1)
        series.append(round_sig(max(0.0, value + wiggle)))
    return series


def lab_points(rng: random.Random, start: float, end: float, points: int, final_outlier: bool = False) -> List[Dict[str, Any]]:
    values = make_series(rng, start, end, points=points, jitter=0.05)
    series = [{"t": TIMEPOINTS[min(i, len(TIMEPOINTS) - 1)], "v": v} for i, v in enumerate(values)]
    if final_outlier:
        series[-1]["outlier"] = True
    return series


def vitals_for_case(rng: random.Random, archetype: str) -> Dict[str, Any]:
    spec = ARCHETYPES[archetype]["vitals"]
    return {
        "hr": round(rng.uniform(*spec["hr"])),
        "bpSys": round(rng.uniform(*spec["bpSys"])),
        "bpDia": round(rng.uniform(*spec["bpDia"])),
        "map": round(rng.uniform(*spec["map"])),
        "rr": round(rng.uniform(*spec["rr"])),
        "spo2": round(rng.uniform(*spec["spo2"])),
        "temp": round(rng.uniform(*spec["temp"]), 1),
        "gcs": round(rng.uniform(*spec["gcs"])),
        "fio2": round(rng.uniform(*spec["fio2"]), 2),
        "pao2": round(rng.uniform(*spec["pao2"])),
    }


def notes_for_case(
    archetype: str,
    name: str,
    diagnosis: str,
    rng: random.Random,
    worsened: bool,
) -> List[Dict[str, str]]:
    templates = ARCHETYPES[archetype]["note_templates"]
    authors = rng.sample(NOTE_AUTHORS, k=3)
    phase_text = "worsening" if worsened else "improving"
    return [
        {
            "time": "Initial",
            "author": authors[0],
            "text": f"{templates[0]} Case trajectory currently {phase_text}. {diagnosis}.",
        },
        {
            "time": "Mid-shift",
            "author": authors[1],
            "text": f"{templates[1]} Name in record: {name}.",
        },
        {
            "time": "Latest",
            "author": authors[2],
            "text": f"{templates[2]} Continue ICU monitoring and reassess at next lab cycle.",
        },
    ]


def build_case(idx: int, archetype: str, rng: random.Random) -> Dict[str, Any]:
    spec = ARCHETYPES[archetype]
    diagnosis = rng.choice(spec["admit_diags"])
    trajectory = rng.choice(spec["trajectory_pool"])
    age = rng.randint(*spec["age_range"])
    sex = rng.choice(["M", "F"])
    weight = round(rng.uniform(*spec["weight_range"]), 1)
    icu_day = round(rng.uniform(1.0, 9.5), 1)
    last_lab_hours = rng.randint(*spec["last_lab_range"])
    name = make_name(idx)

    vitals = vitals_for_case(rng, archetype)

    lab_bases = spec["lab_bases"]
    worsening = trajectory == "worsening"
    improving = trajectory == "improving"

    if archetype == "sepsis":
        wbc = lab_points(rng, rng.uniform(4.0, 10.0), rng.uniform(14.0, 23.0), 4)
        lactate = lab_points(rng, rng.uniform(0.8, 1.4), rng.uniform(2.2, 4.8), 4)
        creat = lab_points(rng, rng.uniform(0.7, 1.1), rng.uniform(1.8, 5.2), 4, final_outlier=rng.random() < 0.18)
        platelets = lab_points(rng, rng.uniform(230, 170), rng.uniform(150, 55), 2)
        bilirubin = lab_points(rng, rng.uniform(0.5, 1.0), rng.uniform(1.1, 3.0), 2)
        pct = lab_points(rng, rng.uniform(0.2, 0.8), rng.uniform(3.0, 15.0), 3)
        bun = lab_points(rng, rng.uniform(16, 24), rng.uniform(28, 64), 3)
    elif archetype == "ards":
        wbc = lab_points(rng, rng.uniform(8.0, 11.0), rng.uniform(12.0, 18.0), 3)
        lactate = lab_points(rng, rng.uniform(1.0, 1.6), rng.uniform(1.8, 3.0), 3)
        creat = lab_points(rng, rng.uniform(0.7, 1.0), rng.uniform(1.2, 2.4), 3)
        platelets = lab_points(rng, rng.uniform(210, 170), rng.uniform(140, 90), 3)
        bilirubin = lab_points(rng, rng.uniform(0.8, 1.6), rng.uniform(1.4, 3.8), 2)
        pct = lab_points(rng, rng.uniform(0.1, 0.5), rng.uniform(0.8, 4.0), 2)
        bun = lab_points(rng, rng.uniform(12, 18), rng.uniform(20, 40), 3)
    elif archetype == "aki":
        wbc = lab_points(rng, rng.uniform(5.0, 8.0), rng.uniform(4.8, 9.0), 3)
        lactate = lab_points(rng, rng.uniform(0.7, 1.4), rng.uniform(0.7, 1.8), 3)
        creat = lab_points(rng, rng.uniform(3.2, 5.6), rng.uniform(1.2, 2.6), 3, final_outlier=rng.random() < 0.10)
        platelets = lab_points(rng, rng.uniform(160, 230), rng.uniform(150, 280), 2)
        bilirubin = lab_points(rng, rng.uniform(0.4, 0.9), rng.uniform(0.6, 1.3), 2)
        pct = lab_points(rng, rng.uniform(0.05, 0.3), rng.uniform(0.1, 0.9), 2)
        bun = lab_points(rng, rng.uniform(34, 74), rng.uniform(18, 32), 3)
    elif archetype == "cardiac":
        wbc = lab_points(rng, rng.uniform(7.0, 10.0), rng.uniform(8.0, 14.0), 3)
        lactate = lab_points(rng, rng.uniform(1.2, 2.0), rng.uniform(1.8, 3.8), 3)
        creat = lab_points(rng, rng.uniform(0.8, 1.2), rng.uniform(1.2, 2.8), 3)
        platelets = lab_points(rng, rng.uniform(180, 240), rng.uniform(130, 200), 2)
        bilirubin = lab_points(rng, rng.uniform(0.5, 1.0), rng.uniform(0.9, 2.2), 2)
        pct = lab_points(rng, rng.uniform(0.1, 0.4), rng.uniform(0.6, 2.4), 2)
        bun = lab_points(rng, rng.uniform(16, 24), rng.uniform(24, 46), 3)
    elif archetype == "neuro":
        wbc = lab_points(rng, rng.uniform(5.2, 8.2), rng.uniform(5.0, 9.8), 3)
        lactate = lab_points(rng, rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.8), 3)
        creat = lab_points(rng, rng.uniform(0.6, 1.2), rng.uniform(0.6, 1.6), 3)
        platelets = lab_points(rng, rng.uniform(160, 240), rng.uniform(160, 250), 2)
        bilirubin = lab_points(rng, rng.uniform(0.4, 0.8), rng.uniform(0.4, 1.1), 2)
        pct = lab_points(rng, rng.uniform(0.05, 0.2), rng.uniform(0.05, 0.8), 2)
        bun = lab_points(rng, rng.uniform(10, 18), rng.uniform(10, 26), 2)
    else:
        wbc = lab_points(rng, rng.uniform(4.8, 7.8), rng.uniform(4.5, 9.5), 3)
        lactate = lab_points(rng, rng.uniform(0.8, 1.3), rng.uniform(0.7, 1.6), 3)
        creat = lab_points(rng, rng.uniform(0.6, 1.2), rng.uniform(0.6, 1.3), 3)
        platelets = lab_points(rng, rng.uniform(170, 260), rng.uniform(180, 290), 2)
        bilirubin = lab_points(rng, rng.uniform(0.4, 0.9), rng.uniform(0.3, 1.0), 2)
        pct = lab_points(rng, rng.uniform(0.03, 0.15), rng.uniform(0.03, 0.5), 2)
        bun = lab_points(rng, rng.uniform(10, 18), rng.uniform(9, 22), 2)

    patient = {
        "id": f"P{idx:03d}",
        "subject_id": f"HC01-P{idx:03d}",
        "name": name,
        "age": age,
        "sex": sex,
        "weight": weight,
        "daysInICU": icu_day,
        "lastLabHoursAgo": last_lab_hours,
        "admitDiag": diagnosis,
        "caseType": archetype,
        "trajectory": trajectory,
        "labels": sorted(set(spec["label_pool"] + (["improving"] if improving else ["worsening"] if worsening else []))),
        "vitals": vitals,
        "labs": {
            "wbc": wbc,
            "lactate": lactate,
            "creatinine": creat,
            "platelets": platelets,
            "bilirubin": bilirubin,
            "procalcitonin": pct,
            "bun": bun,
            "pao2": vitals["pao2"],
        },
        "refs": {
            "wbc": {"low": 4.5, "high": 11.0, "unit": "x10^3/uL"},
            "lactate": {"low": 0.5, "high": 2.0, "unit": "mmol/L"},
            "creatinine": {"low": 0.6, "high": 1.2, "unit": "mg/dL"},
            "platelets": {"low": 150, "high": 400, "unit": "x10^3/uL"},
            "bilirubin": {"low": 0.1, "high": 1.2, "unit": "mg/dL"},
            "procalcitonin": {"low": 0.0, "high": 0.5, "unit": "ng/mL"},
            "bun": {"low": 7, "high": 20, "unit": "mg/dL"},
        },
        "medications": rng.choice(spec["meds"]),
        "notes": notes_for_case(archetype, name, diagnosis, rng, worsened=worsening),
        "outcomes": {
            **spec["outcomes"],
            "mortality_risk": "high" if archetype == "sepsis" and trajectory == "worsening" else "moderate_high" if archetype in {"ards", "cardiac"} else "moderate" if archetype == "aki" else "low",
        },
    }

    # Add convenience fields used by the UI and summary views.
    if archetype == "sepsis":
        patient["labels"].extend(["culture_pending", "pressor_watch"])
    elif archetype == "ards":
        patient["labels"].extend(["ventilation", "hypoxemia"])
    elif archetype == "aki":
        patient["labels"].extend(["renal_recovery", "fluid_management"])
    elif archetype == "cardiac":
        patient["labels"].extend(["hemodynamic_support", "inotrope_watch"])
    elif archetype == "neuro":
        patient["labels"].extend(["delirium_watch", "airway_observation"])
    else:
        patient["labels"].extend(["discharge_planning", "routine_monitoring"])

    patient["notes"] = patient["notes"][:3]
    patient["labels"] = sorted(set(patient["labels"]))

    return patient


def validate_patient(patient: Dict[str, Any]) -> Dict[str, Any]:
    if PatientData is None:
        return patient
    validated = PatientData(**deepcopy(patient))
    return validated.model_dump()


def latest_value(series: Sequence[Dict[str, Any]]) -> float:
    return float(series[-1]["v"])


def compute_sofa_like(patient: Dict[str, Any]) -> int:
    v = patient["vitals"]
    labs = patient["labs"]

    pao2 = float(v.get("pao2", 0.0))
    fio2 = max(float(v["fio2"]), 0.21)
    pf = pao2 / fio2 if pao2 > 0 else None
    resp = 0 if pf is None else 4 if pf < 100 else 3 if pf < 200 else 2 if pf < 300 else 1 if pf < 400 else 0

    plts = latest_value(labs["platelets"])
    bili = latest_value(labs["bilirubin"])
    cr = latest_value(labs["creatinine"])
    coag = 4 if plts < 20 else 3 if plts < 50 else 2 if plts < 100 else 1 if plts < 150 else 0
    liver = 4 if bili >= 12 else 3 if bili >= 6 else 2 if bili >= 2 else 1 if bili >= 1.2 else 0
    meds_lower = [m.lower() for m in patient["medications"]]
    has_pressor = any("norepinephrine" in m or "vasopressin" in m or "dobutamine" in m for m in meds_lower)
    cardio = 3 if has_pressor else 2 if v["map"] < 65 else 1 if v["map"] < 70 else 0
    cns = 4 if v["gcs"] < 6 else 3 if v["gcs"] < 10 else 2 if v["gcs"] < 13 else 1 if v["gcs"] < 15 else 0
    renal = 4 if cr >= 5.0 else 3 if cr >= 3.5 else 2 if cr >= 2.0 else 1 if cr >= 1.2 else 0
    return int(resp + coag + liver + cardio + cns + renal)


def compute_news2_like(patient: Dict[str, Any]) -> int:
    v = patient["vitals"]
    rr_s = 3 if v["rr"] <= 8 else 1 if v["rr"] <= 11 else 0 if v["rr"] <= 20 else 2 if v["rr"] <= 24 else 3
    spo2_s = 3 if v["spo2"] <= 91 else 2 if v["spo2"] <= 93 else 1 if v["spo2"] <= 95 else 0
    o2_s = 2 if v["fio2"] > 0.21 else 0
    bp_s = 3 if v["bpSys"] <= 90 else 2 if v["bpSys"] <= 100 else 1 if v["bpSys"] <= 110 else 0 if v["bpSys"] <= 219 else 3
    hr_s = 3 if v["hr"] <= 40 else 1 if v["hr"] <= 50 else 0 if v["hr"] <= 90 else 1 if v["hr"] <= 110 else 2 if v["hr"] <= 130 else 3
    cns_s = 3 if v["gcs"] < 15 else 0
    tmp_s = 3 if v["temp"] <= 35.0 else 1 if v["temp"] <= 36.0 else 0 if v["temp"] <= 38.0 else 1 if v["temp"] <= 39.0 else 2
    return int(rr_s + spo2_s + o2_s + bp_s + hr_s + cns_s + tmp_s)


def mortality_band(risk: str) -> str:
    return risk


def build_dataset() -> List[Dict[str, Any]]:
    rng = random.Random(SEED)
    patients: List[Dict[str, Any]] = [validate_patient(p) for p in deepcopy(BASE_SEEDS)]

    archetype_order = [
        ("sepsis", 19),
        ("ards", 19),
        ("aki", 19),
        ("cardiac", 20),
        ("neuro", 20),
        ("stable", 20),
    ]

    idx = len(patients) + 1
    for archetype, count in archetype_order:
        for _ in range(count):
            patient = build_case(idx, archetype, rng)
            patient = validate_patient(patient)
            patients.append(patient)
            idx += 1

    # Assign derived summary fields.
    for patient in patients:
        patient["latestScores"] = {
            "sofa": compute_sofa_like(patient),
            "news2": compute_news2_like(patient),
        }
        patient["outcomes"]["scoreBand"] = mortality_band(patient["outcomes"]["mortality_risk"])

    return patients


def write_json(patients: List[Dict[str, Any]]) -> None:
    payload = {
        "metadata": {
            "dataset_name": "HC01 Synthetic ICU Dataset",
            "version": "2.0.0",
            "purpose": "Synthetic ICU cases for UI testing, agent orchestration, risk scoring, and retrieval demos",
            "schema": "PatientData-compatible (id, name, age, sex, weight, daysInICU, admitDiag, vitals, labs, medications, notes)",
            "schema_inspiration": ["MIMIC-IV core/hosp/icu/note modules", "eICU patient/vitals/lab/medication/note tables"],
            "record_count": len(patients),
            "generation_note": "All records are synthetic and deidentified. Values are ICU-like, clinically plausible, and designed for agent testing rather than research use.",
        },
        "patients": patients,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_csv(patients: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "id", "subject_id", "name", "age", "sex", "weight", "daysInICU", "lastLabHoursAgo",
        "admitDiag", "caseType", "trajectory", "latest_sofa", "latest_news2", "latest_lactate",
        "latest_creatinine", "latest_wbc", "latest_pao2", "vasopressor_started", "intubated",
        "dialysis", "mortality_risk",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for patient in patients:
            labs = patient["labs"]
            writer.writerow(
                {
                    "id": patient["id"],
                    "subject_id": patient["subject_id"],
                    "name": patient["name"],
                    "age": patient["age"],
                    "sex": patient["sex"],
                    "weight": patient["weight"],
                    "daysInICU": patient["daysInICU"],
                    "lastLabHoursAgo": patient["lastLabHoursAgo"],
                    "admitDiag": patient["admitDiag"],
                    "caseType": patient["caseType"],
                    "trajectory": patient["trajectory"],
                    "latest_sofa": patient["latestScores"]["sofa"],
                    "latest_news2": patient["latestScores"]["news2"],
                    "latest_lactate": latest_value(labs["lactate"]),
                    "latest_creatinine": latest_value(labs["creatinine"]),
                    "latest_wbc": latest_value(labs["wbc"]),
                    "latest_pao2": patient["vitals"]["pao2"],
                    "vasopressor_started": str(patient["outcomes"]["vasopressor_started"]).lower(),
                    "intubated": str(patient["outcomes"]["intubated"]).lower(),
                    "dialysis": str(patient["outcomes"]["dialysis"]).lower(),
                    "mortality_risk": patient["outcomes"]["mortality_risk"],
                }
            )


def write_readme(patients: List[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = {}
    for patient in patients:
        counts[patient["caseType"]] = counts.get(patient["caseType"], 0) + 1
    readme = f"""# HC01 Synthetic ICU Data

This folder contains a 120-record synthetic ICU dataset for UI testing, prompt development, and backend integration demos.

## Files

- `hc01_synthetic_icu_dataset.json`: full record-level cases with demographics, vitals, labs, medications, notes, labels, outcomes, and derived summary scores.
- `hc01_synthetic_icu_cases.csv`: compact summary table for quick loading or spreadsheet review.

## Schema

The JSON records are aligned with the HC01 app/backend patient model:

- `id`, `name`, `age`, `sex`, `weight`, `daysInICU`, `admitDiag`
- `vitals` with `hr`, `bpSys`, `bpDia`, `map`, `rr`, `spo2`, `temp`, `gcs`, `fio2`, `pao2`
- `labs` with `wbc`, `lactate`, `creatinine`, `platelets`, `bilirubin`, `procalcitonin`, `bun`
- `medications`, `notes`, `labels`, `trajectory`, `outcomes`

## Composition

- Sepsis cases: {counts.get('sepsis', 0)}
- ARDS cases: {counts.get('ards', 0)}
- AKI cases: {counts.get('aki', 0)}
- Cardiovascular cases: {counts.get('cardiac', 0)}
- Neurologic cases: {counts.get('neuro', 0)}
- Stable / stepdown cases: {counts.get('stable', 0)}

## Notes

- All records are synthetic and deidentified.
- The dataset is suitable for testing flows such as sepsis, ARDS, AKI, ward transfer, risk reporting, and note parsing.
- The values are intentionally ICU-like but are not copied from any real patient.
"""
    (DATA_DIR / "README.md").write_text(readme)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    patients = build_dataset()
    write_json(patients)
    write_csv(patients)
    write_readme(patients)
    print(f"Wrote {len(patients)} records to {JSON_PATH.name} and {CSV_PATH.name}")


if __name__ == "__main__":
    main()
