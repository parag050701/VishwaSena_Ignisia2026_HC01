from typing import Dict, List

import numpy as np

MIMIC_IV = {
    "citation": "MIMIC-IV v2.2, PhysioNet, Beth Israel Deaconess Medical Center, 2008-2022",
    "icu_stays": 73141,
    "unique_patients": 50934,
    "sepsis_cohort": 35239,
    "sa_aki_cohort": 15789,
    "ards_cohort": 6390,
    "sofa_mortality_pct": {
        0: 3.2, 1: 4.5, 2: 6.0, 3: 7.5, 4: 10.0, 5: 12.5,
        6: 16.0, 7: 20.0, 8: 26.0, 9: 33.0, 10: 44.0,
        11: 52.0, 12: 62.0, 13: 74.0, 14: 83.0, 15: 91.3,
    },
    "sepsis_lactate_trajectory_mmolL": [1.1, 1.4, 1.8, 2.3, 2.9, 3.4, 3.8],
    "sepsis_wbc_trajectory_kuL": [8.2, 10.1, 12.5, 15.1, 17.8, 19.2, 20.1],
    "aki_stage1_to_stage2_rate": 0.45,
    "aki_contrast_recovery_rate": 0.70,
    "vanco_piptazo_aki_rr": 3.7,
    "lab_refs": {
        "wbc": {"low": 4.0, "high": 12.0, "critical_high": 25.0, "unit": "×10³/μL"},
        "lactate": {"low": 0.5, "high": 2.0, "critical": 4.0, "unit": "mmol/L"},
        "creatinine": {"low": 0.5, "high": 1.3, "critical_ratio": 3.0, "unit": "mg/dL"},
        "platelets": {"low": 150, "high": 400, "critical_low": 50, "unit": "×10³/μL"},
        "bilirubin": {"low": 0.1, "high": 1.2, "critical": 12.0, "unit": "mg/dL"},
        "procalcitonin": {"low": 0.0, "high": 0.5, "sepsis_thresh": 2.0, "unit": "ng/mL"},
        "bun": {"low": 7, "high": 20, "critical": 80, "unit": "mg/dL"},
    },
}

EICU = {
    "citation": "eICU Collaborative Research Database, PhysioNet, 208 hospitals, 2014-2015",
    "encounters": 200859,
    "hospitals": 208,
    "states": 33,
}

PHYSIONET_SEPSIS = {
    "citation": "PhysioNet Early Sepsis Challenge 2019, 3 hospital systems, 60,000+ records",
    "top_auroc": 0.869,
    "challenge_features": [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "Lactate", "WBC",
        "Creatinine", "Bilirubin_total", "Platelets", "PTT", "Fibrinogen",
        "pH", "PaCO2", "FiO2", "ICULOS", "Age",
    ],
}

GUIDELINES: List[Dict] = [
    {
        "id": "ssc2021-lactate",
        "source": "Surviving Sepsis Campaign 2021",
        "section": "§3.1 — Initial Resuscitation",
        "keywords": ["lactate", "sepsis", "hypoperfusion", "resuscitation"],
        "text": "Use serial lactate measurements to guide resuscitation in patients with lactate ≥2 mmol/L as marker of tissue hypoperfusion. Lactate clearance ≥10% indicates adequate response. Persistent elevation despite resuscitation is associated with ≥3× mortality increase. Target lactate normalization within 2-4 hours.",
    },
    {
        "id": "ssc2021-antibiotics",
        "source": "Surviving Sepsis Campaign 2021",
        "section": "§5.2 — Antimicrobial Therapy",
        "keywords": ["antibiotics", "antimicrobial", "septic shock", "broad spectrum", "one hour"],
        "text": "Administer broad-spectrum antimicrobials within 1 hour of recognition of septic shock. For sepsis without shock, assess over 3 hours. Blood cultures ×2 before antibiotics. Empiric double gram-negative coverage for septic shock.",
    },
    {
        "id": "ssc2021-cultures",
        "source": "Surviving Sepsis Campaign 2021",
        "section": "§5.1 — Microbiological Diagnosis",
        "keywords": ["blood cultures", "cultures", "bacteremia", "infection source"],
        "text": "Obtain at least 2 sets of blood cultures from different sites before antimicrobials if delay ≤45 minutes. Culture all potential infection sites. Failure to obtain cultures before antibiotics significantly reduces diagnostic yield and antibiogram utility.",
    },
    {
        "id": "kdigo-aki-staging",
        "source": "KDIGO AKI Guidelines 2012",
        "section": "§2.1 — Staging Criteria",
        "keywords": ["creatinine", "aki", "acute kidney injury", "renal", "urine output", "kdigo"],
        "text": "AKI Stage 1: Cr rise ≥0.3 mg/dL in 48h or 1.5-1.9× baseline; UO <0.5 mL/kg/h 6-12h. Stage 2: 2.0-2.9× baseline; UO <0.5 mL/kg/h ≥12h. Stage 3: ≥3× baseline or ≥4.0 mg/dL absolute or RRT; UO <0.3 mL/kg/h ≥24h or anuria ≥12h.",
    },
    {
        "id": "kdigo-nephrotoxins",
        "source": "KDIGO AKI Guidelines 2012",
        "section": "§3.1 — Prevention",
        "keywords": ["vancomycin", "nephrotoxic", "pip-tazo", "piperacillin", "nsaid", "contrast", "aminoglycoside"],
        "text": "Avoid nephrotoxins in AKI or high-risk patients. Vancomycin + piperacillin-tazobactam: 3.7× increased AKI risk vs vancomycin alone (MIMIC-IV 2023 cohort analysis). Target AUC/MIC vancomycin monitoring preferred over trough-only. NSAIDs contraindicated in AKI — inhibit prostaglandin-mediated renal perfusion.",
    },
    {
        "id": "sofa-criteria",
        "source": "Sepsis-3 Consensus / SOFA (Vincent et al.)",
        "section": "§1 — Organ Failure Scoring",
        "keywords": ["sofa", "organ failure", "sepsis-3", "mortality", "multi-organ"],
        "text": "SOFA ≥2 from baseline = organ dysfunction (sepsis per Sepsis-3). Mortality by maximum SOFA: 0–6 ≈ 3-18%; 7–9 ≈ 22-33%; 10–12 ≈ 44-63%; >12 ≈ 74-91.3% (de Mendonça multicentre, MIMIC-IV validation). Rapid SOFA rise (≥2 in 24h) = acute deterioration marker.",
    },
    {
        "id": "ards-berlin",
        "source": "ARDS Berlin Definition 2012",
        "section": "§1 — Diagnostic Criteria",
        "keywords": ["ards", "respiratory", "pao2", "fio2", "bilateral", "oxygenation", "infiltrates"],
        "text": "ARDS: acute onset within 1 week; bilateral opacities on CXR; not fully explained by cardiac failure; PaO2/FiO2 <300 on PEEP ≥5 cmH₂O. Severity: Mild 200-300 (27% mortality); Moderate 100-200 (32%); Severe <100 (45%). MIMIC-IV ARDS cohort: 6,390 patients.",
    },
    {
        "id": "procalcitonin-sepsis",
        "source": "IDSA/SCCM Procalcitonin Guidelines 2019",
        "section": "§2 — Diagnostic Thresholds",
        "keywords": ["procalcitonin", "pct", "biomarker", "bacterial", "infection", "sepsis marker"],
        "text": "PCT ≥0.5 ng/mL: bacterial infection likely. PCT ≥2.0 ng/mL: sepsis highly likely. PCT >10 ng/mL: severe sepsis/septic shock. Serial PCT decline >80% from peak supports antibiotic de-escalation. PCT-guided therapy reduces antibiotic duration by 1.2 days without harm (meta-analysis n=6,708).",
    },
    {
        "id": "news2-sepsis",
        "source": "NEWS2 — Royal College of Physicians 2017",
        "section": "§1 — Clinical Response Thresholds",
        "keywords": ["news2", "early warning", "deterioration", "sepsis", "rapid response", "monitoring"],
        "text": "NEWS2 ≥5 OR any single red score (3 points on one parameter): urgent review within 1h. NEWS2 ≥7: emergency response, continuous monitoring, critical care. AUC 0.80 for sepsis detection (two-cohort validation). Pooled sensitivity 0.80 (95%CI 0.71-0.86) for death prediction at NEWS ≥5.",
    },
    {
        "id": "physionet-sepsis-features",
        "source": "PhysioNet Early Sepsis Challenge 2019",
        "section": "Feature Importance for Early Sepsis Detection",
        "keywords": ["sepsis prediction", "early detection", "icu", "lactate", "wbc", "vital signs", "machine learning"],
        "text": "Top sepsis detection features (60,000+ records, 3 hospital systems, best AUROC 0.869): Lactate, HR, MAP, O2Sat, Resp rate, Creatinine, Bilirubin, Platelets, WBC, FiO2, ICULOS. Time-series trajectory more predictive than single timepoint. Generalization across hospitals a key challenge (utility drop 0.522→0.364 on unseen system).",
    },
]

_guideline_embeddings: List[np.ndarray] = []
