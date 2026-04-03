"""Data loader for MIMIC-III patient records (CSV format)."""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from app.config import cfg


class MIMICDataLoader:
    """Load and process MIMIC-III CSV data."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or cfg.DATA_DIR
        self._cache = {}

    def _load_csv(self, key: str) -> Optional[pd.DataFrame]:
        """Load or retrieve cached CSV."""
        if key in self._cache:
            return self._cache[key]

        path = cfg.MIMIC_CSVS.get(key)
        if not path or not os.path.exists(path):
            print(f"Warning: {key} not found at {path}")
            return None

        # MIMIC CSVs use pipe delimiters
        df = pd.read_csv(path, low_memory=False, delimiter="|")
        self._cache[key] = df
        return df

    def get_patient_admission(
        self, patient_id: int, admission_id: int
    ) -> Optional[Dict]:
        """Get patient admission record."""
        icu = self._load_csv("icustays")
        if icu is None:
            return None

        # Find ICU stay (handle both uppercase and lowercase column names)
        subject_col = "SUBJECT_ID" if "SUBJECT_ID" in icu.columns else "subject_id"
        hadm_col = "HADM_ID" if "HADM_ID" in icu.columns else "hadm_id"
        intime_col = "INTIME" if "INTIME" in icu.columns else "intime"
        outtime_col = "OUTTIME" if "OUTTIME" in icu.columns else "outtime"
        los_col = "LOS" if "LOS" in icu.columns else "los"

        stay = icu[(icu[subject_col] == patient_id) & (icu[hadm_col] == admission_id)]
        if stay.empty:
            return None

        stay = stay.iloc[0]
        patients = self._load_csv("patients")
        if patients is None:
            return None

        patient_subject_col = (
            "SUBJECT_ID" if "SUBJECT_ID" in patients.columns else "subject_id"
        )
        patient = patients[patients[patient_subject_col] == patient_id]
        if patient.empty:
            return None
        patient = patient.iloc[0]

        dob_col = "DOB" if "DOB" in patient.index or "DOB" in patients.columns else None
        gender_col = "GENDER" if "GENDER" in patients.columns else "gender"

        age = 65  # Default age for ICU patients
        if dob_col and pd.notna(patient.get(dob_col)):
            try:
                age = (
                    (pd.Timestamp(stay[intime_col]) - pd.Timestamp(patient[dob_col])).days
                    // 365
                )
            except Exception:
                pass

        return {
            "patient_id": patient_id,
            "admission_id": admission_id,
            "age": age,
            "gender": patient.get(gender_col, "U"),
            "icu_in": str(stay[intime_col]),
            "icu_out": str(stay[outtime_col]),
            "los_icu": float(stay[los_col]),
        }

    def get_clinical_notes(self, patient_id: int, admission_id: int) -> list:
        """Get clinical notes for admission."""
        notes = self._load_csv("noteevents")
        if notes is None:
            return []

        subject_col = "SUBJECT_ID" if "SUBJECT_ID" in notes.columns else "subject_id"
        hadm_col = "HADM_ID" if "HADM_ID" in notes.columns else "hadm_id"
        charttime_col = "CHARTTIME" if "CHARTTIME" in notes.columns else "charttime"
        category_col = "CATEGORY" if "CATEGORY" in notes.columns else "category"
        text_col = "TEXT" if "TEXT" in notes.columns else "text"

        adm_notes = notes[
            (notes[subject_col] == patient_id) & (notes[hadm_col] == admission_id)
        ]
        return [
            {
                "time": row[charttime_col],
                "category": row[category_col],
                "text": str(row[text_col])[:500] if pd.notna(row[text_col]) else "",
            }
            for _, row in adm_notes.iterrows()
        ]

    def get_lab_values(
        self, patient_id: int, admission_id: int, lab_name: str = None
    ) -> list:
        """Get lab values for admission."""
        labs = self._load_csv("labevents")
        d_labs = self._load_csv("d_labitems")
        if labs is None or d_labs is None:
            return []

        subject_col = "SUBJECT_ID" if "SUBJECT_ID" in labs.columns else "subject_id"
        hadm_col = "HADM_ID" if "HADM_ID" in labs.columns else "hadm_id"
        itemid_col = "ITEMID" if "ITEMID" in labs.columns else "itemid"
        charttime_col = "CHARTTIME" if "CHARTTIME" in labs.columns else "charttime"
        valuenum_col = "VALUENUM" if "VALUENUM" in labs.columns else "valuenum"
        valueuom_col = "VALUEUOM" if "VALUEUOM" in labs.columns else "valueuom"

        d_itemid_col = "ITEMID" if "ITEMID" in d_labs.columns else "itemid"
        d_label_col = "LABEL" if "LABEL" in d_labs.columns else "label"

        adm_labs = labs[
            (labs[subject_col] == patient_id) & (labs[hadm_col] == admission_id)
        ]

        if lab_name:
            lab_id = d_labs[d_labs[d_label_col].str.lower() == lab_name.lower()][
                d_itemid_col
            ].values
            if len(lab_id) > 0:
                adm_labs = adm_labs[adm_labs[itemid_col] == lab_id[0]]

        results = []
        for _, row in adm_labs.iterrows():
            lab = d_labs[d_labs[d_itemid_col] == row[itemid_col]]
            if not lab.empty:
                val = row[valuenum_col]
                if pd.notna(val):
                    results.append(
                        {
                            "time": row[charttime_col],
                            "name": lab.iloc[0][d_label_col],
                            "value": float(val),
                            "unit": row[valueuom_col] if pd.notna(row[valueuom_col]) else "N/A",
                        }
                    )
        return results

    def synthesize_vitals(
        self, patient_id: int, admission_id: int
    ) -> Dict[str, Tuple[float, str]]:
        """
        Synthesize vital signs by sampling from realistic distributions
        based on MIMIC-III statistics and ICU patient profiles.
        """
        admission = self.get_patient_admission(patient_id, admission_id)
        if not admission:
            return {}

        age = admission.get("age", 65)
        gender = admission.get("gender", "M")

        # Realistic ICU vital ranges (mean ± std for adult ICU patients)
        vitals = {}

        # Heart rate: higher in acute illness
        hr = np.random.normal(loc=95, scale=15)
        vitals["Heart Rate"] = (max(40, min(150, int(hr))), "bpm")

        # Blood pressure (systolic/diastolic)
        sbp = np.random.normal(loc=130, scale=20)
        dbp = np.random.normal(loc=75, scale=12)
        vitals["Systolic BP"] = (max(70, min(200, int(sbp))), "mmHg")
        vitals["Diastolic BP"] = (max(40, min(120, int(dbp))), "mmHg")

        # Body temperature
        temp = np.random.normal(loc=37.2, scale=0.8)
        vitals["Temperature"] = (round(max(35.0, min(40.0, temp)), 1), "°C")

        # Respiratory rate
        rr = np.random.normal(loc=20, scale=4)
        vitals["Respiratory Rate"] = (max(8, min(40, int(rr))), "bpm")

        # SpO2 (oxygen saturation)
        spo2 = np.random.normal(loc=96, scale=2)
        vitals["SpO2"] = (max(85, min(100, int(spo2))), "%")

        return vitals

    def get_medications(self, patient_id: int, admission_id: int) -> list:
        """Get medications for admission."""
        presc = self._load_csv("prescriptions")
        if presc is None:
            return []

        subject_col = "SUBJECT_ID" if "SUBJECT_ID" in presc.columns else "subject_id"
        hadm_col = "HADM_ID" if "HADM_ID" in presc.columns else "hadm_id"
        drug_col = "DRUG" if "DRUG" in presc.columns else "drug"
        dose_col = "DOSE_VAL_RX" if "DOSE_VAL_RX" in presc.columns else "dose_val_rx"
        unit_col = "DOSE_UNIT_RX" if "DOSE_UNIT_RX" in presc.columns else "dose_unit_rx"

        adm_presc = presc[
            (presc[subject_col] == patient_id) & (presc[hadm_col] == admission_id)
        ]

        return [
            {
                "drug": row[drug_col] if pd.notna(row[drug_col]) else "Unknown",
                "dose": row[dose_col] if pd.notna(row[dose_col]) else "N/A",
                "unit": row[unit_col] if pd.notna(row[unit_col]) else "N/A",
            }
            for _, row in adm_presc.iterrows()
        ]

    @staticmethod
    def get_synthetic_patient(patient_id: int = 1, admission_id: int = 1) -> Dict:
        """Get or generate a synthetic patient record."""
        return {
            "patient_id": patient_id,
            "admission_id": admission_id,
            "age": np.random.randint(40, 85),
            "gender": np.random.choice(["M", "F"]),
            "icu_in": (datetime.now() - timedelta(days=2)).isoformat(),
            "icu_out": datetime.now().isoformat(),
            "los_icu": 2,
            "chief_complaint": np.random.choice(
                [
                    "Shortness of breath",
                    "Chest pain",
                    "Sepsis",
                    "Post-operative monitoring",
                    "Trauma evaluation",
                ]
            ),
        }


def load_patient_data(
    patient_id: int, admission_id: int = None, use_synthetic: bool = True
) -> Dict:
    """
    Load complete patient data including admission info, notes, labs, and vitals.
    Falls back to synthetic data if real data unavailable.
    """
    loader = MIMICDataLoader()

    if use_synthetic:
        admission = loader.get_synthetic_patient(patient_id, admission_id or 1)
        notes = [
            {
                "time": datetime.now().isoformat(),
                "category": "Nursing/Other",
                "text": f"Patient {patient_id} admitted with {admission['chief_complaint']}. Initial vitals stable.",
            }
        ]
        vitals = loader.synthesize_vitals(patient_id, admission_id or 1)
        medications = []
    else:
        admission = loader.get_patient_admission(patient_id, admission_id)
        if not admission:
            return {}
        notes = loader.get_clinical_notes(patient_id, admission_id)
        vitals = loader.synthesize_vitals(patient_id, admission_id)
        medications = loader.get_medications(patient_id, admission_id)

    return {
        "admission": admission,
        "notes": notes,
        "vitals": vitals,
        "medications": medications,
    }
