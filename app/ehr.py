"""
FHIR R4 EHR Integration for HC01.
Converts FHIR resources → PatientData using httpx (no aiohttp dependency).
Supports SMART on FHIR OAuth2 client-credentials flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .audit import log_ehr_query
from .config import cfg
from .models import PatientData, PatientVitals

log = logging.getLogger("HC01.ehr")

# ─── LOINC → HC01 lab key mapping ────────────────────────────────────────────
# Reference: https://loinc.org/  (subset used in critical care)
_LOINC_TO_LAB: Dict[str, str] = {
    "2160-0":   "creatinine",    # Creatinine [Mass/volume] in Serum
    "32693-4":  "lactate",       # Lactate [Moles/volume] in Blood
    "26464-8":  "wbc",           # Leukocytes [#/volume] in Blood
    "778-1":    "platelets",     # Platelets [#/volume] in Blood
    "1975-2":   "bilirubin",     # Bilirubin, total [Mass/volume]
    "3094-0":   "bun",           # Urea nitrogen [Mass/volume] in Serum
    "2823-3":   "potassium",     # Potassium [Moles/volume] in Serum
    "2951-2":   "sodium",        # Sodium [Moles/volume] in Serum
    "33959-8":  "procalcitonin", # Procalcitonin [Mass/volume] in Serum
    "2345-7":   "glucose",       # Glucose [Mass/volume] in Serum
    "718-7":    "hemoglobin",    # Hemoglobin [Mass/volume] in Blood
    "4544-3":   "hematocrit",    # Hematocrit [Volume Fraction] of Blood
    "2703-7":   "pao2",          # Oxygen [Partial pressure] in Arterial blood
    "6299-2":   "bun",           # BUN alternate code
    "48065-7":  "fibrinogen",    # Fibrinogen [Mass/volume] in Platelet poor plasma
    "14979-9":  "pt",            # Prothrombin time (PT)
    "5902-2":   "inr",           # INR
    "30341-2":  "esr",           # Erythrocyte sedimentation rate
}

# LOINC → vital sign key
_LOINC_TO_VITAL: Dict[str, str] = {
    "8867-4":  "hr",     # Heart rate
    "8480-6":  "bpSys",  # Systolic BP
    "8462-4":  "bpDia",  # Diastolic BP
    "8478-0":  "map",    # Mean arterial pressure
    "9279-1":  "rr",     # Respiratory rate
    "59408-5": "spo2",   # SpO2
    "8310-5":  "temp",   # Body temperature
    "9269-2":  "gcs",    # GCS total score
    "19994-3": "fio2",   # FiO2
    "2703-7":  "pao2",   # PaO2
    "55284-4": "bp",     # BP panel (handle separately)
}

# FHIR Observation category → whether to treat as lab or vital
_VITAL_CATEGORIES = {"vital-signs", "vitals"}


# ─── OAuth2 token management ──────────────────────────────────────────────────

class _TokenCache:
    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._expires_at: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def get(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: str,
    ) -> str:
        async with self._lock:
            if self._token and self._expires_at and datetime.now(tz=timezone.utc) < self._expires_at:
                return self._token
            return await self._refresh(client_id, client_secret, token_url, scope)

    async def _refresh(self, client_id, client_secret, token_url, scope) -> str:
        log.info("Requesting new SMART on FHIR OAuth token…")
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                token_url,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     client_id,
                    "client_secret": client_secret,
                    "scope":         scope,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OAuth token error {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            self._token      = data["access_token"]
            expires_in       = int(data.get("expires_in", 3600))
            self._expires_at = datetime.now(tz=timezone.utc) + timedelta(seconds=expires_in - 60)
            log.info("SMART OAuth token acquired, expires in %ds", expires_in)
            return self._token


# ─── FHIR client ─────────────────────────────────────────────────────────────

class FHIRClient:
    """
    Async FHIR R4 client.
    Supports:
    - Anonymous access (public HAPI sandbox)
    - SMART on FHIR client-credentials OAuth2
    - Full PatientData conversion
    - Audit logging for compliance
    """

    HAPI_PUBLIC = "https://hapi.fhir.org/baseR4"

    def __init__(
        self,
        server_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        oauth_url: Optional[str] = None,
        scope: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.server_url = (server_url or cfg.FHIR_SERVER_URL or self.HAPI_PUBLIC).rstrip("/")
        self.client_id     = client_id     or cfg.FHIR_CLIENT_ID
        self.client_secret = client_secret or cfg.FHIR_CLIENT_SECRET
        self.oauth_url     = oauth_url     or cfg.FHIR_OAUTH_URL
        self.scope         = scope         or cfg.FHIR_SCOPE
        self.timeout       = timeout
        self._token_cache  = _TokenCache()
        self._cache: Dict[str, Tuple[Any, float]] = {}

    @property
    def is_public(self) -> bool:
        return not (self.client_id and self.client_secret and self.oauth_url)

    async def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"}
        if not self.is_public:
            token = await self._token_cache.get(
                self.client_id, self.client_secret, self.oauth_url, self.scope
            )
            h["Authorization"] = f"Bearer {token}"
        return h

    async def _get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        url = f"{self.server_url}/{path.lstrip('/')}"
        cache_key = f"{url}?{json.dumps(params or {}, sort_keys=True)}"

        # Cache hit (5-min TTL)
        if cache_key in self._cache:
            data, ts = self._cache[cache_key]
            if datetime.now().timestamp() - ts < cfg.FHIR_CACHE_TTL:
                return data

        try:
            headers = await self._headers()
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()
                self._cache[cache_key] = (data, datetime.now().timestamp())
                return data
        except httpx.HTTPStatusError as exc:
            log.error("FHIR GET %s → HTTP %s", path, exc.response.status_code)
            return None
        except Exception as exc:
            log.error("FHIR GET %s error: %s", path, exc)
            return None

    # ── Resource accessors ────────────────────────────────────────────────

    async def get_patient_resource(
        self,
        fhir_id: Optional[str] = None,
        name: Optional[str] = None,
        mrn: Optional[str] = None,
    ) -> Optional[Dict]:
        if fhir_id:
            res = await self._get(f"Patient/{fhir_id}")
            log_ehr_query(self.server_url, "Patient", {"id": fhir_id}, 1 if res else 0)
            return res
        params: Dict[str, str] = {}
        if name:
            params["name"] = name
        if mrn:
            params["identifier"] = mrn
        bundle = await self._get("Patient", params)
        log_ehr_query(self.server_url, "Patient", params,
                      len(bundle.get("entry", [])) if bundle else 0)
        if bundle and bundle.get("entry"):
            return bundle["entry"][0]["resource"]
        return None

    async def get_observations(
        self, patient_id: str, category: Optional[str] = None, code: Optional[str] = None
    ) -> List[Dict]:
        params: Dict[str, str] = {"patient": patient_id, "_sort": "-date", "_count": "200"}
        if category:
            params["category"] = category
        if code:
            params["code"] = code
        bundle = await self._get("Observation", params)
        log_ehr_query(self.server_url, "Observation", params,
                      len(bundle.get("entry", [])) if bundle else 0)
        if bundle and bundle.get("entry"):
            return [e["resource"] for e in bundle["entry"]]
        return []

    async def get_medication_requests(self, patient_id: str) -> List[Dict]:
        # Try MedicationRequest first, then MedicationStatement
        for resource in ("MedicationRequest", "MedicationStatement"):
            params = {"patient": patient_id, "status": "active", "_count": "50"}
            bundle = await self._get(resource, params)
            log_ehr_query(self.server_url, resource, params,
                          len(bundle.get("entry", [])) if bundle else 0)
            if bundle and bundle.get("entry"):
                return [e["resource"] for e in bundle["entry"]]
        return []

    async def get_conditions(self, patient_id: str) -> List[Dict]:
        params = {"patient": patient_id, "clinical-status": "active", "_count": "20"}
        bundle = await self._get("Condition", params)
        log_ehr_query(self.server_url, "Condition", params,
                      len(bundle.get("entry", [])) if bundle else 0)
        if bundle and bundle.get("entry"):
            return [e["resource"] for e in bundle["entry"]]
        return []

    async def get_encounters(self, patient_id: str) -> List[Dict]:
        params = {"patient": patient_id, "_sort": "-date", "_count": "5"}
        bundle = await self._get("Encounter", params)
        if bundle and bundle.get("entry"):
            return [e["resource"] for e in bundle["entry"]]
        return []

    async def get_document_references(self, patient_id: str) -> List[Dict]:
        params = {"patient": patient_id, "_sort": "-date", "_count": "10"}
        bundle = await self._get("DocumentReference", params)
        if bundle and bundle.get("entry"):
            return [e["resource"] for e in bundle["entry"]]
        return []

    # ── Full record pull ──────────────────────────────────────────────────

    async def get_full_record(self, patient_id: str) -> Dict[str, Any]:
        """Pull all clinical resources for a patient in parallel."""
        results = await asyncio.gather(
            self.get_patient_resource(fhir_id=patient_id),
            self.get_observations(patient_id),
            self.get_medication_requests(patient_id),
            self.get_conditions(patient_id),
            self.get_encounters(patient_id),
            self.get_document_references(patient_id),
            return_exceptions=True,
        )
        return {
            "patient":    results[0] if not isinstance(results[0], Exception) else None,
            "observations": results[1] if not isinstance(results[1], Exception) else [],
            "medications":  results[2] if not isinstance(results[2], Exception) else [],
            "conditions":   results[3] if not isinstance(results[3], Exception) else [],
            "encounters":   results[4] if not isinstance(results[4], Exception) else [],
            "documents":    results[5] if not isinstance(results[5], Exception) else [],
            "retrieved_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    # ── FHIR → PatientData converter ─────────────────────────────────────

    async def to_patient_data(
        self, fhir_id: str, user_id: str = "anonymous"
    ) -> Optional[PatientData]:
        """
        Convert FHIR resources for a patient into a PatientData object
        ready for the HC01 diagnostic pipeline.
        """
        record = await self.get_full_record(fhir_id)
        if not record.get("patient"):
            log.warning("FHIR patient %s not found", fhir_id)
            return None

        mapper = _FHIRMapper(record)
        pd = mapper.build()
        from .audit import log_patient_access
        log_patient_access(fhir_id, "ehr_pull", "fhir", user_id,
                           details={"server": self.server_url,
                                    "resources": [k for k, v in record.items()
                                                  if v and k != "retrieved_at"]})
        return pd

    async def capability_statement(self) -> Optional[Dict]:
        """Fetch server capability statement (useful for validation)."""
        return await self._get("metadata")


# ─── FHIR resource → PatientData mapper ──────────────────────────────────────

class _FHIRMapper:
    """Converts a raw FHIR record dict into PatientData."""

    def __init__(self, record: Dict[str, Any]) -> None:
        self.rec = record
        self.patient    = record.get("patient") or {}
        self.obs        = record.get("observations") or []
        self.meds       = record.get("medications") or []
        self.conditions = record.get("conditions") or []
        self.encounters = record.get("encounters") or []
        self.documents  = record.get("documents") or []

    def build(self) -> PatientData:
        fhir_id  = self.patient.get("id", "fhir-unknown")
        name     = self._name()
        age, sex = self._demographics()
        weight   = self._weight()
        los      = self._los_days()
        diag     = self._admit_diag()
        labs     = self._labs()
        vitals   = self._vitals(labs)
        meds     = self._medications()
        notes    = self._notes()

        return PatientData(
            id=f"fhir-{fhir_id}",
            name=name,
            age=age,
            sex=sex,
            weight=weight,
            daysInICU=los,
            admitDiag=diag,
            vitals=vitals,
            labs=labs,
            medications=meds,
            notes=notes if notes else [{
                "time": datetime.now().isoformat(),
                "author": "FHIR Import",
                "text": f"Patient imported from FHIR server. Admission: {diag}",
            }],
        )

    def _name(self) -> str:
        for name_entry in self.patient.get("name", []):
            given  = " ".join(name_entry.get("given", []))
            family = name_entry.get("family", "")
            if given or family:
                return f"{given} {family}".strip()
        return f"FHIR-{self.patient.get('id', 'Unknown')}"

    def _demographics(self) -> Tuple[int, str]:
        gender = self.patient.get("gender", "unknown")[0].upper()
        if gender not in ("M", "F"):
            gender = "U"
        dob_str = self.patient.get("birthDate", "")
        age = 65
        if dob_str:
            try:
                dob = datetime.strptime(dob_str[:10], "%Y-%m-%d")
                age = max(0, min(130, (datetime.now() - dob).days // 365))
            except ValueError:
                pass
        return age, gender

    def _weight(self) -> float:
        """Pull weight from Observation (LOINC 29463-7) or default."""
        for obs in self.obs:
            codes = _obs_codes(obs)
            if "29463-7" in codes:  # Body weight
                v = _obs_value(obs)
                if v and 20 < v < 500:
                    # FHIR weight may be in kg or lbs
                    unit = _obs_unit(obs).lower()
                    return round(v if "kg" in unit else v * 0.453592, 1)
        return 70.0  # Default

    def _los_days(self) -> float:
        for enc in self.encounters:
            period = enc.get("period", {})
            start  = period.get("start")
            end    = period.get("end")
            if start:
                try:
                    t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(end.replace("Z", "+00:00")) if end else datetime.now(tz=timezone.utc)
                    return round((t1 - t0).total_seconds() / 86400, 2)
                except Exception:
                    pass
        return 1.0

    def _admit_diag(self) -> str:
        for cond in self.conditions:
            code_entry = cond.get("code", {})
            for coding in code_entry.get("coding", []):
                display = coding.get("display", "")
                if display:
                    return display[:120]
            text = code_entry.get("text", "")
            if text:
                return text[:120]
        for enc in self.encounters:
            for reason in enc.get("reasonCode", []):
                for coding in reason.get("coding", []):
                    disp = coding.get("display", "")
                    if disp:
                        return disp[:120]
        return "ICU Admission"

    def _labs(self) -> Dict[str, List[Dict]]:
        result: Dict[str, List[Dict]] = {}
        for obs in self.obs:
            categories = [
                c.get("coding", [{}])[0].get("code", "")
                for c in obs.get("category", [])
            ]
            # Skip vital-sign observations
            if any(c in _VITAL_CATEGORIES for c in categories):
                continue

            loinc_code = _obs_loinc(obs)
            lab_key    = _LOINC_TO_LAB.get(loinc_code) if loinc_code else None

            # Fallback: try display name matching
            if not lab_key:
                display = (_obs_display(obs) or "").lower()
                for substr, key in {
                    "lactate": "lactate", "creatinine": "creatinine",
                    "white blood": "wbc", "platelet": "platelets",
                    "bilirubin": "bilirubin", "urea": "bun",
                    "procalcitonin": "procalcitonin",
                }.items():
                    if substr in display:
                        lab_key = key
                        break

            if not lab_key:
                continue

            val = _obs_value(obs)
            if val is None or val < 0:
                continue

            t_str = obs.get("effectiveDateTime", obs.get("issued", datetime.now().isoformat()))
            result.setdefault(lab_key, []).append({"t": t_str[:19], "v": round(float(val), 3)})

        # Preserve insertion order from the FHIR bundle (already chronological).
        # Cap at last 8 readings per lab.
        for k in result:
            result[k] = result[k][-8:]

        return result

    def _vitals(self, labs: Dict) -> PatientVitals:
        vital_map: Dict[str, float] = {}
        for obs in self.obs:
            loinc = _obs_loinc(obs)
            if not loinc:
                continue
            vital_key = _LOINC_TO_VITAL.get(loinc)
            if not vital_key:
                continue
            val = _obs_value(obs)
            if val is None:
                continue

            # Special handling for BP panel (component-based)
            if vital_key == "bp":
                for comp in obs.get("component", []):
                    c_loinc = _obs_loinc(comp)
                    c_key   = _LOINC_TO_VITAL.get(c_loinc, "")
                    c_val   = _obs_value(comp)
                    if c_key and c_val:
                        vital_map[c_key] = float(c_val)
            else:
                if vital_key not in vital_map:
                    vital_map[vital_key] = float(val)

        # Convert temperature F→C if needed (FHIR usually in Celsius but verify)
        temp = vital_map.get("temp", 37.0)
        if temp > 45:  # Likely Fahrenheit
            temp = round((temp - 32) * 5 / 9, 1)

        # Compute MAP if missing
        sbp = vital_map.get("bpSys", 120.0)
        dbp = vital_map.get("bpDia", 70.0)
        map_ = vital_map.get("map") or round((sbp + 2 * dbp) / 3, 1)

        # pao2 fallback from labs
        pao2_arr = labs.get("pao2", [])
        pao2 = float(pao2_arr[-1]["v"]) if pao2_arr else vital_map.get("pao2", 0.0)

        return PatientVitals(
            hr=vital_map.get("hr", 80.0),
            bpSys=sbp,
            bpDia=dbp,
            map=map_,
            rr=vital_map.get("rr", 16.0),
            spo2=vital_map.get("spo2", 97.0),
            temp=max(30.0, min(45.0, temp)),
            gcs=vital_map.get("gcs", 15.0),
            fio2=vital_map.get("fio2", 0.21),
            pao2=pao2,
        )

    def _medications(self) -> List[str]:
        result = []
        seen   = set()
        for med in self.meds:
            # MedicationRequest
            mc = med.get("medicationCodeableConcept") or med.get("medication", {})
            name = ""
            for coding in mc.get("coding", []):
                name = coding.get("display", "")
                if name:
                    break
            if not name:
                name = mc.get("text", "")
            if not name:
                # MedicationStatement has medicationReference
                ref = med.get("medicationReference", {}).get("display", "")
                name = ref

            if name and name.lower() not in seen:
                seen.add(name.lower())
                # Dosage instruction
                for dosage in (med.get("dosageInstruction") or med.get("dosage") or []):
                    dose_text = dosage.get("text", "").strip()
                    if dose_text and dose_text.lower() != name.lower():
                        name = f"{name} — {dose_text}"
                    break
                result.append(name[:100])

        return result[:30]

    def _notes(self) -> List[Dict[str, str]]:
        notes = []
        for doc in self.documents:
            content_list = doc.get("content", [])
            for content in content_list:
                attachment = content.get("attachment", {})
                text = attachment.get("data", "")  # base64 in FHIR
                if text:
                    try:
                        import base64
                        text = base64.b64decode(text).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                title   = doc.get("description", doc.get("type", {}).get("text", "Clinical Note"))
                created = doc.get("date", "")
                if text:
                    notes.append({
                        "time":   created[:19] if created else datetime.now().isoformat(),
                        "author": title,
                        "text":   text[:800],
                    })
        return notes[:10]


# ─── FHIR observation helper functions ───────────────────────────────────────

def _obs_codes(obs: Dict) -> List[str]:
    return [c.get("code", "") for c in obs.get("code", {}).get("coding", [])]


def _obs_loinc(obs: Dict) -> Optional[str]:
    for coding in obs.get("code", {}).get("coding", []):
        system = coding.get("system", "")
        if "loinc" in system.lower():
            return coding.get("code")
    # Fallback: any coding code
    codes = _obs_codes(obs)
    return codes[0] if codes else None


def _obs_display(obs: Dict) -> Optional[str]:
    for coding in obs.get("code", {}).get("coding", []):
        d = coding.get("display")
        if d:
            return d
    return obs.get("code", {}).get("text")


def _obs_value(obs: Dict) -> Optional[float]:
    # valueQuantity
    vq = obs.get("valueQuantity")
    if vq:
        return vq.get("value")
    # valueString → try parse
    vs = obs.get("valueString")
    if vs:
        try:
            return float(vs)
        except ValueError:
            return None
    # component-based (BP): return first component value
    for comp in obs.get("component", []):
        v = _obs_value(comp)
        if v is not None:
            return v
    return None


def _obs_unit(obs: Dict) -> str:
    vq = obs.get("valueQuantity", {})
    return vq.get("unit", vq.get("code", ""))


# ─── Module-level singleton ───────────────────────────────────────────────────

_fhir_client: Optional[FHIRClient] = None


def get_fhir_client() -> FHIRClient:
    global _fhir_client
    if _fhir_client is None:
        _fhir_client = FHIRClient(
            server_url=cfg.FHIR_SERVER_URL or FHIRClient.HAPI_PUBLIC,
            client_id=cfg.FHIR_CLIENT_ID,
            client_secret=cfg.FHIR_CLIENT_SECRET,
            oauth_url=cfg.FHIR_OAUTH_URL,
            scope=cfg.FHIR_SCOPE,
            timeout=float(cfg.FHIR_TIMEOUT),
        )
    return _fhir_client
