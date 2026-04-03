"""SOFA v2 and NEWS2 clinical scoring calculators."""

from typing import Dict, List

import numpy as np

from .data import MIMIC_IV
from .models import PatientData


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def keyword_score(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    lower = text.lower()
    hits = sum(1 for k in keywords if k.lower() in lower)
    return hits / len(keywords)


def calc_sofa(pt: PatientData) -> Dict:
    v = pt.vitals
    labs = pt.labs

    def latest(key: str) -> float:
        arr = labs.get(key, [])
        if not arr:
            return 0.0
        entry = arr[-1]
        return float(entry["v"] if isinstance(entry, dict) else entry)

    # Respiratory — skip if pao2 not measured (returns 0.0 → safe)
    pao2 = v.pao2 if v.pao2 > 0 else 0.0
    fio2 = max(v.fio2, 0.21)   # Minimum room-air FiO2
    pf = pao2 / fio2 if pao2 > 0 else None

    if pf is None:
        resp = 0   # Can't score without PaO2
    else:
        resp = 4 if pf < 100 else 3 if pf < 200 else 2 if pf < 300 else 1 if pf < 400 else 0

    plts  = latest("platelets")
    bili  = latest("bilirubin")
    cr    = latest("creatinine")

    meds_lower = [m.lower() for m in pt.medications]
    has_norepi = any("norepinephrine" in m or "norepi" in m for m in meds_lower)
    has_vaso   = any("vasopressin" in m or "phenylephrine" in m for m in meds_lower)
    has_dopa   = any("dopamine" in m or "dobutamine" in m for m in meds_lower)

    coag  = 4 if plts < 20 else 3 if plts < 50 else 2 if plts < 100 else 1 if plts < 150 else 0
    liver = 4 if bili >= 12 else 3 if bili >= 6 else 2 if bili >= 2 else 1 if bili >= 1.2 else 0
    cardio = (
        3 if (has_norepi or has_vaso) else
        2 if (has_dopa or v.map < 65) else
        1 if v.map < 70 else
        0
    )
    cns   = 4 if v.gcs < 6 else 3 if v.gcs < 10 else 2 if v.gcs < 13 else 1 if v.gcs < 15 else 0
    renal = 4 if cr >= 5.0 else 3 if cr >= 3.5 else 2 if cr >= 2.0 else 1 if cr >= 1.2 else 0

    total = resp + coag + liver + cardio + cns + renal
    mortality = MIMIC_IV["sofa_mortality_pct"].get(min(total, 15), 91.3)

    return dict(
        resp=resp, coag=coag, liver=liver, cardio=cardio, cns=cns, renal=renal,
        total=total, mortality_pct=mortality,
        pf_ratio=round(pf, 1) if pf is not None else None,
    )


def calc_news2(pt: PatientData) -> Dict:
    v = pt.vitals

    rr_s   = 3 if v.rr <= 8  else 1 if v.rr <= 11  else 0 if v.rr <= 20 else 2 if v.rr <= 24  else 3
    spo2_s = 3 if v.spo2 <= 91 else 2 if v.spo2 <= 93 else 1 if v.spo2 <= 95 else 0
    o2_s   = 2 if v.fio2 > 0.21 else 0
    bp_s   = 3 if v.bpSys <= 90 else 2 if v.bpSys <= 100 else 1 if v.bpSys <= 110 else 0 if v.bpSys <= 219 else 3
    hr_s   = 3 if v.hr <= 40   else 1 if v.hr <= 50   else 0 if v.hr <= 90  else 1 if v.hr <= 110 else 2 if v.hr <= 130 else 3
    cns_s  = 3 if v.gcs < 15 else 0
    tmp_s  = 3 if v.temp <= 35.0 else 1 if v.temp <= 36.0 else 0 if v.temp <= 38.0 else 1 if v.temp <= 39.0 else 2

    total = rr_s + spo2_s + o2_s + bp_s + hr_s + cns_s + tmp_s
    level = "HIGH" if total >= 7 else "MEDIUM" if total >= 5 else "LOW"

    return dict(
        total=total, level=level,
        breakdown=dict(RR=rr_s, SpO2=spo2_s, O2=o2_s, SBP=bp_s, HR=hr_s, GCS=cns_s, Temp=tmp_s),
    )
