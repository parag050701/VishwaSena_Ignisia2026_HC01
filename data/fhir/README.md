# HC01 FHIR Synthetic Export

This folder contains FHIR R4-compliant synthetic data generated from `data/hc01_synthetic_icu_dataset.json`.

## Files

- `hc01_synthetic_fhir_bundle.json`: single FHIR `Bundle` (`type=transaction`) for one-shot import.
- `hc01_synthetic_fhir_ndjson/`: per-resource NDJSON files.

## Resource counts

- Patient: 120
- Encounter: 120
- Condition: 120
- Observation: 3528
- MedicationRequest: 375
- DocumentReference: 361
- Total bundle entries: 4624

## Import examples

### 1) HAPI FHIR (transaction bundle)

```bash
curl -X POST   -H "Content-Type: application/fhir+json"   --data @data/fhir/hc01_synthetic_fhir_bundle.json   https://hapi.fhir.org/baseR4
```

### 2) Local FHIR server

```bash
curl -X POST   -H "Content-Type: application/fhir+json"   --data @data/fhir/hc01_synthetic_fhir_bundle.json   http://localhost:8080/fhir
```

## Notes

- All data is synthetic and deidentified.
- Observations use LOINC coding for vitals and labs used by `app/ehr.py` mapping.
- DocumentReference note content is stored in base64 text/plain attachments.
