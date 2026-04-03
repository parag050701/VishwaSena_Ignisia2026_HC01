"""Clinical RAG corpus for HC01 ICU Diagnostic Risk Assistant.

Each entry is a guideline chunk with citation, section, and clinically useful text.
The corpus is intentionally chunked (3-5 chunks per source) for better retrieval.
"""

from __future__ import annotations

GUIDELINE_CORPUS = [
    # 1) Surviving Sepsis Campaign 2021 (general sepsis bundle)
    {
        "id": "ssc2021_lactate_1",
        "citation": "Surviving Sepsis Campaign 2021",
        "section": "Initial Resuscitation - Lactate Assessment",
        "text": "In suspected sepsis or septic shock, measure serum lactate early as a marker of tissue hypoperfusion. A lactate value of 2 mmol/L or greater is clinically concerning and should prompt close reassessment of perfusion, source control, and resuscitation response.",
    },
    {
        "id": "ssc2021_lactate_2",
        "citation": "Surviving Sepsis Campaign 2021",
        "section": "Initial Resuscitation - Lactate Re-measurement",
        "text": "If initial lactate is elevated, repeat lactate measurement to assess trajectory. Down-trending lactate suggests improved perfusion, while persistent or rising lactate despite intervention is associated with worse outcomes and should trigger escalation of hemodynamic evaluation.",
    },
    {
        "id": "ssc2021_fluids_1",
        "citation": "Surviving Sepsis Campaign 2021",
        "section": "Initial Resuscitation - Fluid Therapy",
        "text": "For sepsis-induced hypoperfusion or septic shock, begin rapid administration of 30 mL/kg crystalloid, then continue fluid resuscitation guided by dynamic perfusion variables rather than static targets alone. Reassess blood pressure, urine output, mental status, and lactate response.",
    },
    {
        "id": "ssc2021_cultures_1",
        "citation": "Surviving Sepsis Campaign 2021",
        "section": "Diagnosis - Blood Cultures and Antimicrobials",
        "text": "Obtain blood cultures before antibiotics when this does not cause substantial delay. At least two sets from separate sites improve microbiologic yield, but antimicrobial therapy should not be inappropriately delayed in unstable patients.",
    },

    # 2) KDIGO AKI 2012/2024
    {
        "id": "kdigo_aki_staging_1",
        "citation": "KDIGO AKI Guidelines 2012/2024",
        "section": "AKI Definition and Staging",
        "text": "AKI is defined by any of the following: increase in serum creatinine by at least 0.3 mg/dL within 48 hours, increase to at least 1.5 times baseline within 7 days, or urine output below 0.5 mL/kg/hour for 6 hours. These criteria support early recognition before severe kidney failure develops.",
    },
    {
        "id": "kdigo_aki_staging_2",
        "citation": "KDIGO AKI Guidelines 2012/2024",
        "section": "AKI Severity Stages",
        "text": "KDIGO staging stratifies severity: Stage 1 includes creatinine 1.5-1.9 times baseline or at least 0.3 mg/dL rise, Stage 2 includes 2.0-2.9 times baseline, and Stage 3 includes at least 3 times baseline, creatinine at least 4.0 mg/dL, or need for renal replacement therapy.",
    },
    {
        "id": "kdigo_nephrotoxins_1",
        "citation": "KDIGO AKI Guidelines 2012/2024",
        "section": "AKI Prevention - Nephrotoxic Exposure",
        "text": "Avoid or minimize nephrotoxic medications in high-risk ICU patients, especially those with sepsis, hypotension, pre-existing CKD, or rising creatinine. Medication review should include NSAIDs, aminoglycosides, contrast load, and potentially nephrotoxic antibiotic combinations.",
    },
    {
        "id": "kdigo_rrt_1",
        "citation": "KDIGO AKI Guidelines 2012/2024",
        "section": "Renal Replacement Therapy Indications",
        "text": "Consider renal replacement therapy for refractory volume overload, severe metabolic acidosis, dangerous hyperkalemia, or uremic complications not controlled with medical therapy. Clinical context and trends are more important than a single creatinine threshold alone.",
    },

    # 3) SOFA / Sepsis-3
    {
        "id": "sofa_organs_1",
        "citation": "SOFA - Vincent et al. 1996 / Sepsis-3",
        "section": "Organ System Scoring",
        "text": "The SOFA score quantifies dysfunction across respiratory, coagulation, liver, cardiovascular, CNS, and renal systems. Higher total SOFA reflects greater multi-organ injury burden and correlates with in-hospital mortality risk.",
    },
    {
        "id": "sofa_threshold_1",
        "citation": "SOFA - Vincent et al. 1996 / Sepsis-3",
        "section": "Sepsis Definition Threshold",
        "text": "Sepsis-3 defines sepsis as suspected infection plus an acute increase in SOFA of 2 points or more from baseline, representing clinically significant organ dysfunction. This threshold identifies patients at substantially higher risk of death.",
    },
    {
        "id": "sofa_trend_1",
        "citation": "SOFA - Vincent et al. 1996 / Sepsis-3",
        "section": "Sequential Change Significance",
        "text": "Sequential rise in SOFA is clinically important. Even when the absolute score is moderate, an upward trajectory over 24 hours can indicate worsening shock physiology or treatment failure and should prompt urgent reassessment.",
    },

    # 4) Berlin ARDS Definition
    {
        "id": "berlin_ards_criteria_1",
        "citation": "Berlin ARDS Definition 2012",
        "section": "Diagnostic Criteria",
        "text": "ARDS requires acute onset within one week of a known insult, bilateral opacities on chest imaging, and respiratory failure not fully explained by cardiac failure or fluid overload. Objective oxygenation impairment is required under PEEP or CPAP of at least 5 cm H2O.",
    },
    {
        "id": "berlin_ards_pf_1",
        "citation": "Berlin ARDS Definition 2012",
        "section": "Severity by PaO2/FiO2 Ratio",
        "text": "ARDS severity is classified by PaO2/FiO2: mild 200-300, moderate 100-200, and severe below 100, each measured with PEEP at least 5 cm H2O. Lower P/F ratio indicates more severe oxygenation failure and increased mortality.",
    },
    {
        "id": "berlin_ards_vent_1",
        "citation": "Berlin ARDS Definition 2012",
        "section": "Lung Protective Ventilation",
        "text": "For ARDS, lung-protective ventilation with low tidal volume around 6 mL/kg predicted body weight and plateau pressure limitation is associated with better outcomes. Avoiding volutrauma and barotrauma is central to management.",
    },

    # 5) ASHP Nephrotoxicity 2023
    {
        "id": "ashp_nephrotox_riskcombo_1",
        "citation": "ASHP Nephrotoxicity Guidelines 2023",
        "section": "High-Risk Drug Combinations",
        "text": "Concurrent nephrotoxic agents increase AKI risk substantially. ICU teams should identify high-risk combinations early and implement mitigation plans, including dose optimization, hydration strategy, and intensified renal monitoring.",
    },
    {
        "id": "ashp_nephrotox_amino_nsaid_1",
        "citation": "ASHP Nephrotoxicity Guidelines 2023",
        "section": "Aminoglycoside and NSAID Risk",
        "text": "The combination of aminoglycosides and NSAIDs can produce additive renal injury, especially in septic or hemodynamically unstable patients. Daily renal function review and de-escalation of nephrotoxic exposure are recommended.",
    },
    {
        "id": "ashp_nephrotox_vanco_auc_1",
        "citation": "ASHP Nephrotoxicity Guidelines 2023",
        "section": "Vancomycin Exposure Strategy",
        "text": "AUC-guided vancomycin dosing is preferred over trough-only monitoring to maintain efficacy while reducing nephrotoxicity. Excessive vancomycin exposure, particularly with other nephrotoxins, is linked to AKI and requires rapid dosing reassessment.",
    },

    # 6) Sepsis shock vasopressor management
    {
        "id": "ssc_shock_pressors_1",
        "citation": "Surviving Sepsis Campaign 2021 - Septic Shock Management",
        "section": "Vasopressor First-Line Choice",
        "text": "In septic shock, norepinephrine is the recommended first-line vasopressor. Start promptly when hypotension persists after initial fluid resuscitation to restore perfusion pressure and reduce prolonged tissue hypoxia.",
    },
    {
        "id": "ssc_shock_pressors_2",
        "citation": "Surviving Sepsis Campaign 2021 - Septic Shock Management",
        "section": "MAP Target",
        "text": "An initial mean arterial pressure target of about 65 mmHg is recommended for most adults with septic shock. Targets may be individualized based on chronic hypertension, perfusion markers, and end-organ response.",
    },
    {
        "id": "ssc_shock_pressors_3",
        "citation": "Surviving Sepsis Campaign 2021 - Septic Shock Management",
        "section": "Vasopressin Add-On Strategy",
        "text": "If adequate MAP is not achieved with norepinephrine alone, vasopressin can be added as a second agent to reduce norepinephrine dose requirements. Escalation should be guided by shock severity and dynamic perfusion assessment.",
    },
    {
        "id": "ssc_shock_pressors_4",
        "citation": "Surviving Sepsis Campaign 2021 - Septic Shock Management",
        "section": "Escalation Beyond Dual Vasopressors",
        "text": "For refractory shock despite norepinephrine and vasopressin, additional vasopressor or inotrope strategies may be required with urgent source control and bedside reassessment. Persistent hypotension indicates high mortality risk and mandates aggressive multidisciplinary management.",
    },
]
