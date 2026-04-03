# HC01 Synthetic ICU Data

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

- Sepsis cases: 20
- ARDS cases: 20
- AKI cases: 20
- Cardiovascular cases: 20
- Neurologic cases: 20
- Stable / stepdown cases: 20

## Notes

- All records are synthetic and deidentified.
- The dataset is suitable for testing flows such as sepsis, ARDS, AKI, ward transfer, risk reporting, and note parsing.
- The values are intentionally ICU-like but are not copied from any real patient.
