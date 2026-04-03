# HC01 Clinical Validation Integration Guide

**Summary**: How to integrate clinical safety validation into your HC01 pipeline to ensure zero hallucination, complete evidence grounding, and risk-aware escalations.

---

## Quick Start

### 1. Run Validation After Each Diagnosis

```python
from scripts.validate_clinical_safety import ClinicalValidationSuite

suite = ClinicalValidationSuite()

# In your diagnosis workflow:
diagnosis = await hc01.diagnose(patient_data)

# Immediately validate before returning to clinician
validation_report = suite.run_quick_suite([diagnosis.as_dict()])

# Only return to clinician if validation passes
if validation_report["overall_status"] == "PASS":
    return diagnosis
else:
    log_validation_failure(diagnosis, validation_report)
    raise ClinicalSafetyException(f"Diagnosis failed validation: {validation_report}")
```

### 2. Audit Clinician Decisions

```python
from app.models import ClinicalAuditLog

audit_log = ClinicalAuditLog()

# After clinician reviews HC01 diagnosis:
audit_log.log(
    patient_id=patient.id,
    hc01_diagnosis=diagnosis.diagnosis,
    hc01_confidence=diagnosis.confidence,
    clinician_decision="accepted" | "overridden",
    clinician_reason="..." if overridden else None,
    final_diagnosis=...,
)

# Monthly: check for confidence drift
if audit_log.override_rate() > 0.15:
    alert("Model confidence drift detected. Retraining recommended.")
```

### 3. Monthly Calibration Check

Schedule this monthly task:

```bash
python scripts/validate_clinical_safety.py --mode full --audit-log-path /var/hc01/audit.db
```

---

## Integration Points

### In `app/main.py`

```python
from scripts.validate_clinical_safety import ClinicalValidationSuite

suite = ClinicalValidationSuite()

@app.post("/api/diagnose")
async def diagnose_http(req: DiagnoseHTTPRequest) -> Dict[str, Any]:
    diagnosis = await master_orchestrate(ctx)
    
    # Validate before returning
    validation = suite.run_quick_suite([diagnosis.as_dict()])
    if validation["overall_status"] != "PASS":
        # Log severity
        for error in validation["results"]:
            if error["status"] == "fail":
                log.error(f"Safety validation failed: {error['test_name']}")
        
        # Add flag to output
        diagnosis.validation_warnings = validation["results"]
    
    return diagnosis.model_dump()
```

### In `app/agents.py` (Chief Agent)

```python
from scripts.validate_clinical_safety import HallucinationChecker

hallucination_checker = HallucinationChecker()

async def chief_synthesis(patient: PatientData, rag_context: str) -> str:
    synthesis = await generate_synthesis(patient, rag_context)
    
    # Check for fabricated citations
    citations = extract_citations(synthesis)
    fabrication_check = hallucination_checker.check_no_fabricated_guidelines(citations)
    
    if fabrication_check.status == ValidationStatus.FAIL:
        # Regenerate without hallucinated guidelines
        synthesis = await regenerate_without_fabrications(patient, rag_context, fabrication_check.details)
    
    return synthesis
```

### In WebSocket Handler

```python
@app.websocket("/ws/diagnose")
async def ws_diagnose(websocket: WebSocket):
    # ... existing code ...
    
    # After orchestration completes:
    validation = suite.run_quick_suite([ctx.as_dict()])
    
    # Stream validation results to client
    await websocket.send_json({
        "type": "validation_complete",
        "status": validation["overall_status"],
        "mean_score": validation["mean_score"],
        "passed_tests": validation["passed_tests"],
    })
    
    if validation["overall_status"] != "PASS":
        # Alert clinician to review carefully
        await websocket.send_json({
            "type": "alert",
            "level": "warning",
            "message": "Some validation checks did not pass. Review evidence carefully.",
            "details": validation["results"],
        })
```

---

## Validation Rules per Recommendation Type

### High-Risk Recommendations (Require Escalation)

| Recommendation Type | Min Confidence | Must Cite | Requires | Escalate To |
|--------------------|----------------|-----------|----------|------------|
| Antibiotic (broad-spectrum) | 90% | Sepsis/ID guideline | Renal dosing check, Allergy check | Infectious Disease |
| Vasopressor escalation | 95% | Sepsis guideline | Clinical confirmation | Critical Care Physician |
| Ventilator adjustment | 92% | ARDS guidelines | Respiratory physiology | Respiratory Therapist |
| Diuretic dosing | 88% | CKD/Volume status guideline | Creatinine trend | Nephrology/ICU |
| Anticoagulation escalation | 93% | Thromboembolism guideline | Bleeding risk assessment | Hematology |

### Medium-Risk (Display with Caveat)

| Recommendation | Min Confidence | Display as |
|---|---|---|
| Monitoring adjustment | 75% | "Consider..." |
| Diagnostic test ordering | 80% | "May be useful..." |
| Supportive care change | 75% | "Could help patient..." |

### Low-Risk (Information Only)

| Information | Min Confidence |
|---|---|
| Patient summary | 60% |
| Risk factor discussion | 70% |

---

## Evidence Citation Requirements

Every clinical claim MUST follow this schema:

```json
{
  "claim": "Patient meets sepsis-3 criteria",
  "confidence": 0.94,
  "evidence": [
    {
      "type": "guideline",
      "id": "sepsis-3-ssfa-2016",
      "quote": "Suspected infection + organ dysfunction",
      "doi": "10.1097/CCM.0000000000001082",
      "page": 490,
    },
    {
      "type": "fhir_patient_data",
      "resource_type": "Observation",
      "resource_id": "obs-12345",
      "code": "qSOFA",
      "value": 2,
      "date": "2026-04-03T14:00:00Z",
    },
  ],
  "contradictions": [],
  "uncertainty_notes": "Fever source not yet identified; possible community vs. nosocomial"
}
```

---

## Handling Validation Failures

### Scenario 1: Fabricated Citation Detected

```python
def handle_hallucination_detected(prediction, fabrication_report):
    log.error(
        f"Hallucination detected in case {prediction.case_id}: "
        f"{len(fabrication_report.errors)} fabricated citations"
    )
    
    # Option A: Regenerate using only verified guidelines
    regenerated = await regenerate_with_approved_guidelines(
        patient=prediction.patient,
        approved_guidelines=list(GUIDELINE_REGISTRY.keys()),
    )
    
    # Option B: Escalate to human review
    escalate_to_clinical_review(prediction, reason="Hallucination detected")
    
    # Never silently accept the hallucinated version
    return regenerated or None
```

### Scenario 2: Confidence Drift Detected

```python
def handle_confidence_drift():
    """Triggered if clinician override rate > 15% per month."""
    alert_msg = """
    ⚠️ Model Confidence Drift Alert
    - Override rate: 23% (threshold: 15%)
    - Mean calibration error: 0.12 (target: < 0.05)
    - Recommendation: Retrain on recent data
    """
    
    # Option A: Automated retraining
    trigger_retraining_job()
    
    # Option B: Manual validation
    escalate_to_ml_team(alert_msg)
```

### Scenario 3: Contradiction Detected

```python
def handle_contradiction():
    """E.g., recommend diuretics but urine output is zero."""
    log.warn("Safety contradiction detected: will escalate and flag")
    
    output.flags.append({
        "type": "contradiction",
        "severity": "high",
        "message": "Recommendation conflicts with patient status. Clinician review required.",
        "safe_to_implement": False,
    })
    
    # Always escalate
    escalate_to_human(output)
```

---

## Monthly Dashboard

Create a dashboard showing:

```
HC01 Clinical Validation Monthly Report – April 2026
═══════════════════════════════════════════════════════

Accuracy Metrics:
  ✓ Diagnostic sensitivity (sepsis): 93% (target: ≥92%)
  ✓ Diagnostic specificity (sepsis): 89% (target: ≥88%)
  ⚠ NEWS2 calibration error: 0.08 (target: <0.05)

Hallucination Metrics:
  ✓ Fabricated guidelines: 0 in 850 diagnoses
  ✓ Citation fidelity: 100% (850/850 verified)
  ✓ Claims fully grounded: 98% (834/850)

Safety Metrics:
  ✓ High-risk escalated: 100% (24/24 recommendations)
  ✓ Contradictions flagged: 100% (3/3 detected)
  ✓ Clinician override rate: 12% (within tolerance)

Confidence Calibration:
  70% bucket: 71% actual ✓
  80% bucket: 79% actual ✓
  90% bucket: 91% actual ✓
  95% bucket: 94% actual ⚠ (drift toward underconfidence)

Recommendations:
  1. Monitor 95% bucket closely
  2. Retrain if drift continues next month
  3. Expand sepsis-3 validation set to 100 cases
```

---

## Running Full Validation Suite

```bash
# Quick validation (essential checks)
python scripts/validate_clinical_safety.py --mode quick

# Full validation with audit log analysis
python scripts/validate_clinical_safety.py --mode full --audit-log /var/hc01/audit.db

# Generate PDF report
python scripts/validate_clinical_safety.py --mode full --output-html validation_report.html
```

---

## Next Steps

1. **Implement guideline registry**: Populate `GUIDELINE_REGISTRY` in `validate_clinical_safety.py` with real guidelines + DOIs
2. **Create gold-standard test set**: Work with ICU physicians to build 50–100 verified cases
3. **Set up audit logging**: Wire `ClinicalAuditLog` into production workflow
4. **Monthly calibration**: Schedule monthly validation runs and review drift
5. **Escalation workflow**: Integrate escalation alerts into hospital alerting system

---

**Version**: 1.0.0  
**Last Updated**: April 2026  
**Next Review**: Quarterly or after major model update

