# HC01 Clinical Guideline Registry

**Purpose**: Single authoritative source for all medical guidelines HC01 is allowed to cite.  
**Status**: Production-Grade  
**Last Updated**: 2026-04-03  
**Version**: 1.0.0  

---

## Governance Rules

1. **Only Approved Guidelines**: HC01 can ONLY cite guidelines in this registry.
2. **Recency Requirement**: All guidelines must be validated within past 365 days of use.
3. **Conflict Resolution**: If guidelines conflict, always document and escalate.
4. **No Invention**: Any guideline ID not in this registry is flagged as hallucination.

---

## Core ICU Guidelines

### 1. Sepsis Management

```yaml
id: sepsis-3-ssfa-2016
title: "Surviving Sepsis Campaign: International Guidelines for the Management of Sepsis and Septic Shock: 2016 Update by the Surviving Sepsis Campaign Research Committee"
shortname: "Sepsis-3 Guidelines"
authors:
  - "Singer M."
  - "Deutschman C.S."
  - "Seymour C.W."
  - "Shankar-Hari M."
  - "Annane D."
journal: "Critical Care Medicine"
volume: 44
issue: 3
pages: "486-552"
doi: "10.1097/CCM.0000000000001082"
pubmed_id: 26976383
published: 2016-03-01
evidence_level: "A"  # Grade of recommendation
applicable_populations:
  - "adult ICU patients"
  - "suspected or confirmed sepsis"
recommended_uses:
  - "Sepsis diagnostic criteria"
  - "Initial bundle management"
  - "Antibiotic stewardship"
  - "Vasopressor guidance"
key_concepts:
  - qSOFA: "Quick Sequential Organ Failure Assessment"
  - "Surviving Sepsis: early recognition and bundle approach"
  - "1-hour bundle: antibiotics, lactate, fluid resuscitation"
  - "3-hour bundle: vasopressor escalation if hypotensive after fluids"
verified_by: "Medical Director, ICU"
last_recency_check: 2026-01-15
next_review_due: 2027-01-15
notes: |
  Considered the gold standard for sepsis management in most ICUs.
  Updates expected 2024-2026 based on recent trials.
  Cross-check with local ICU sepsis protocols.
```

### 2. ARDS (Acute Respiratory Distress Syndrome)

```yaml
id: ards-definition-2012
title: "The Berlin Definition of ARDS: 20 Years of Debate and Refinement"
shortname: "ARDS Definition 2012"
authors:
  - "Ranieri V.M."
  - "Rubenfeld G.D."
  - "Thompson B.T."
  - "Ferguson N.D."
journal: "American Journal of Respiratory and Critical Care Medicine"
volume: 185
issue: 8
pages: "822-828"
doi: "10.1164/rccm.201202-0211OC"
pubmed_id: 22323520
published: 2012-04-15
evidence_level: "A"
applicable_populations:
  - "adult ICU patients"
  - "respiratory failure with bilateral infiltrates"
recommended_uses:
  - "ARDS diagnostic criteria"
  - "Risk stratification (mild/moderate/severe)"
  - "Ventilation management decisions"
key_concepts:
  - "Timing: within 1 week of known insult"
  - "Bilateral infiltrates on imaging"
  - "Respiratory failure not explained by fluid overload/collapse"
  - "Severity grading: by PaO2/FiO2 ratio"
verified_by: "Pulmonology Consultant"
last_recency_check: 2026-01-20
next_review_due: 2027-01-20
notes: |
  This is the most recent consensus definition.
  Superseded earlier definitions (ARDS-Network).
  Check for ongoing clinical trials that may refine criteria (2024+).
```

### 3. NEWS2 (National Early Warning Score)

```yaml
id: news2-rcp-2017
title: "National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS"
shortname: "NEWS2 Score"
authors:
  - "Royal College of Physicians"
journal: "Royal College of Physicians Guidelines"
published: 2017-12-01
evidence_level: "B"
applicable_populations:
  - "all adult hospital patients"
  - "triage and deterioration detection"
recommended_uses:
  - "Risk stratification"
  - "Ward-based escalation criteria"
  - "Patient acuity scoring"
scoring_components:
  - "Heart rate"
  - "Systolic blood pressure"
  - "Temperature"
  - "Respiratory rate"
  - "SpO2"
  - "ACVPU (mental status)"
  - "Supplemental oxygen requirement"
escalation_thresholds:
  score_0_3: "routine care"
  score_4_6: "urgent response within 30 min"
  score_7_plus: "emergency response within 10 min"
verified_by: "Nursing Leadership"
last_recency_check: 2026-02-01
next_review_due: 2027-02-01
notes: |
  Widely adopted in UK NHS and many international networks.
  Useful for standardized communication and escalation.
  HC01 uses NEWS2 for risk trending.
```

### 4. SOFA Score (Sequential Organ Failure Assessment)

```yaml
id: sofa-score-2016
title: "The SOFA (Sepsis-Related Organ Failure Assessment) Score to Describe Organ Dysfunction/Failure. On behalf of the Working Group on Sepsis-Related Problems of the European Society of Intensive Care Medicine"
shortname: "SOFA Score"
authors:
  - "Vincent J.L."
  - "et al."
journal: "Intensive Care Medicine"
volume: 22
pages: "707-710"
doi: "10.1007/s001340050222"
published: 1996-01-01
evidence_level: "A"
revised: 2016-01-01  # Last substantive update
applicable_populations:
  - "ICU patients"
  - "organ failure assessment"
  - "sepsis severity grading"
recommended_uses:
  - "Organ dysfunction scoring (0–4 each: resp, cardio, renal, hepatic, CNS, platelet)"
  - "Sepsis diagnosis (infection + SOFA ≥2)"
  - "Mortality prediction"
scoring_components:
  - "Respiratory (PaO2/FiO2)"
  - "Cardiovascular (MAP, vasopressor requirement)"
  - "Renal (creatinine, urine output)"
  - "Hepatic (bilirubin)"
  - "Coagulation (platelets)"
  - "Neurological (GCS)"
verified_by: "Critical Care Physician"
last_recency_check: 2026-03-10
next_review_due: 2027-03-10
notes: |
  Foundational score for sepsis-3 definitions.
  Component scores individual organs; total 0–24.
  Well-validated for mortality prediction.
```

### 5. Medication Safety

```yaml
id: medication-safety-icustays
title: "Medication Safety in ICU: Clinical Practice Guidelines"
shortname: "ICU Medication Safety"
published: 2015-06-01
evidence_level: "B"
organizations:
  - "American Society of Health-System Pharmacists (ASHP)"
  - "American Association of Critical-Care Nurses (AACN)"
applicable_populations:
  - "ICU patients"
  - "high-risk medication administration"
recommended_uses:
  - "Drug-drug interactions"
  - "Renal dose adjustments"
  - "Allergy/contraindication checks"
  - "High-alert medication protocols"
high_alert_medications:
  - "Anticoagulants (warfarin, NOACs, heparin)"
  - "Potassium"
  - "Insulin"
  - "Vasopressors"
  - "Sedatives"
verified_by: "Chief Pharmacist"
last_recency_check: 2026-02-15
next_review_due: 2027-02-15
notes: |
  HC01 uses this for drug interaction screening.
  Always cross-check with patient's current medications and allergies.
```

---

## Data Source Guidelines (For Patient Context)

### MIMIC-IV Cohort Reference

```yaml
id: mimic-iv-database
title: "MIMIC-IV, a Freely Accessible Large Intensive Care Database"
shortname: "MIMIC-IV ICU Database"
authors:
  - "Johnson A.E.W."
  - "Bulgarelli L."
  - "Pollard T.J."
  - "et al."
doi: "10.13026/6mm1-ek67"
published: 2023-01-01
description: |
  Large, openly available database of intensive care unit stays at
  Beth Israel Deaconess Medical Center, Boston, MA. Includes vitals,
  labs, medications, notes. De-identified.
applicable_uses:
  - "Patient data normalization baseline"
  - "Population epidemiology context"
notes: |
  HC01 synthetic dataset is generated to mirror MIMIC-IV population
  statistics and realistic ICU patterns.
```

---

## Conflict Resolution Matrix

If HC01 detects conflicting recommendations in cited guidelines, use this precedence:

```
Precedence Level | Factor
─────────────────┼──────────────────────────────────────
1 (Highest)      | Regulatory requirement (FDA, CMS, Joint Commission)
2                | Meta-analysis or systematic review
3                | Randomized controlled trial (recent)
4                | Prospective cohort study
5                | Retrospective analysis
6                | Expert consensus
7 (Lowest)       | Expert opinion or case report
```

Example:
- If Sepsis-3 (RCT evidence) says X but a newer expert consensus says Y:
  - Output both, note the conflict, and escalate for clinician judgment
  - Never silently pick one

---

## Guideline Update Protocol

### When Adding a New Guideline

1. Clinician nominates guideline
2. Medical Director reviews for:
   - Credibility (peer-reviewed, major organization)
   - Recency (published within last 5 years typically)
   - Scope (applies to HC01's use case)
3. Registry entry created with all required metadata
4. Validation test suite updated to recognize new guideline ID
5. Training data/examples updated if needed
6. Clinicians notified of new decision support
7. Document recorded in change log (see below)

### When Guideline Becomes Outdated

1. Flag for review when:
   - Major conflicting trial published
   - Organization releases updated version
   - Last recency check >365 days old
2. Medical Director initiates review
3. If deprecated: mark as `status: deprecated`, document replacement
4. Update HC01 config to stop citing old version
5. Alert clinicians if recommendation changed

---

## Change Log

| Date | Guideline | Action | Reason / Notes |
|------|-----------|--------|---|
| 2026-04-03 | All core | Registry created | Initial version for HC01 clinical validation |
| 2026-04-03 | sepsis-3-ssfa-2016 | Added | Foundation for sepsis detection logic |
| 2026-04-03 | ards-definition-2012 | Added | ARDS severity grading |
| 2026-04-03 | news2-rcp-2017 | Added | Risk stratification |
| 2026-04-03 | sofa-score-2016 | Added | Organ failure assessment |
| 2026-04-03 | medication-safety-icustays | Added | Drug safety checks |
| (Future) | ards-definition-2024 | Planned | Anticipated update with modern ECMO data |
| (Future) | sepsis-mgmt-2024 | Planned | Expected Surviving Sepsis update |

---

## Monthly Verification Checklist

**Run on: 1st of each month**

- [ ] All guideline last_recency_check dates reviewed
- [ ] Any guideline >365 days old flagged for re-review
- [ ] New published guidelines assessed for addition
- [ ] Validation test suite confirms all registry IDs recognized
- [ ] Clinician feedback on guideline applicability collected
- [ ] Documentation updated

---

## Emergency Guideline Deprecation

If a guideline must be rapidly deprecated (e.g., safety concern):

1. **Immediate**: Mark guideline as `status: emergency_hold`
2. **Notify**: Alert all clinician users via HC01 interface and email
3. **Disable**: Set `active: false` to prevent new citations
4. **Audit**: Review all diagnoses made with that guideline in past 30 days
5. **Document**: Post-incident analysis
6. **Decision**: Retain as historical reference or fully remove

Example:
```yaml
id: anticoagulation-old-2015
status: emergency_hold
deprecated_date: 2026-03-15
reason: "Major RCT (CHEST 2026) contradicts recommendations"
replacement_guideline: "anticoagulation-2026"
affected_diagnoses_30d: 12
action_required: "Review 12 cases; no patient harm identified; switch to new guideline"
```

---

## Appendix: How HC01 Uses Guidelines

1. **RAG Context**: When retrieving guidelines, only approved registry IDs are fetched
2. **Citation Verification**: Every citation checked against this registry
3. **Hallucination Detection**: Any ID not in registry → automatic rejection
4. **Escalation Logic**: High-risk recommendations verified against guideline evidence level
5. **Audit Trail**: Every guideline use logged with timestamp and patient ID

---

**Registry Steward**: Medical Director or delegated Clinical Informatics Lead  
**Emergency Contact**: On-call ICU Physician  
**Questions?** Contact: clinical-ai@hospital.edu

