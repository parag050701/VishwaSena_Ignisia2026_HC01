"""
EHR Integration Module - FHIR API Support
Handles real-time patient data retrieval from EHR systems
"""

import os
import asyncio
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
import json

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from fhirclient.models.patient import Patient
    from fhirclient.models.encounter import Encounter
    from fhirclient.models.observation import Observation
except ImportError:
    Patient = Encounter = Observation = None

logger = logging.getLogger(__name__)


class EHRConfig:
    """EHR Configuration"""
    FHIR_SERVER_URL = os.getenv("FHIR_SERVER_URL", "")
    FHIR_CLIENT_ID = os.getenv("FHIR_CLIENT_ID", "")
    FHIR_CLIENT_SECRET = os.getenv("FHIR_CLIENT_SECRET", "")
    FHIR_OAUTH_URL = os.getenv("FHIR_OAUTH_URL", "")
    FHIR_SCOPE = os.getenv("FHIR_SCOPE", "user/Patient.read user/Observation.read user/Encounter.read")
    FHIR_TIMEOUT = 30  # seconds
    CACHE_TTL = 300  # 5 minutes


class FHIRAuth:
    """FHIR OAuth2 Authentication"""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        oauth_url: str,
        scope: str = EHRConfig.FHIR_SCOPE,
    ):
        """Initialize FHIR authentication
        
        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret
            oauth_url: OAuth token endpoint
            scope: OAuth scope
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.oauth_url = oauth_url
        self.scope = scope
        self.access_token = None
        self.token_expiry = None
    
    async def get_token(self) -> str:
        """Get valid OAuth token, refresh if needed
        
        Returns:
            Access token
        """
        # Check if token is still valid
        if self.access_token and self.token_expiry:
            if datetime.now() < self.token_expiry:
                return self.access_token
        
        try:
            logger.info("Requesting OAuth token from FHIR server")
            
            if aiohttp is None:
                raise ImportError("aiohttp not installed")
            
            async with aiohttp.ClientSession() as session:
                data = {
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "scope": self.scope,
                }
                
                async with session.post(
                    self.oauth_url,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=EHRConfig.FHIR_TIMEOUT),
                ) as response:
                    if response.status != 200:
                        raise Exception(f"OAuth error: {response.status}")
                    
                    token_data = await response.json()
                    self.access_token = token_data["access_token"]
                    
                    # Set expiry to token_expiry - 60 seconds (refresh early)
                    expires_in = token_data.get("expires_in", 3600)
                    from datetime import timedelta
                    self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                    
                    logger.info("OAuth token obtained successfully")
                    return self.access_token
        
        except Exception as e:
            logger.error(f"Failed to get OAuth token: {e}")
            raise


class FHIRClient:
    """FHIR API Client for EHR Integration"""
    
    def __init__(
        self,
        server_url: str,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        oauth_url: Optional[str] = None,
    ):
        """Initialize FHIR client
        
        Args:
            server_url: FHIR server base URL
            client_id: OAuth client ID (optional for public servers)
            client_secret: OAuth client secret
            oauth_url: OAuth token endpoint
        """
        self.server_url = server_url.rstrip("/")
        self.auth = None
        self.cache = {}
        
        if client_id and client_secret and oauth_url:
            self.auth = FHIRAuth(
                client_id=client_id,
                client_secret=client_secret,
                oauth_url=oauth_url,
            )
        
        if aiohttp is None:
            logger.warning("aiohttp not installed - EHR queries may not work")
    
    def _get_cache_key(self, resource_type: str, query: Dict) -> str:
        """Generate cache key for query"""
        query_str = json.dumps(query, sort_keys=True)
        return f"{resource_type}:{query_str}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.cache:
            return False
        
        timestamp = self.cache[cache_key]["timestamp"]
        age = datetime.now().timestamp() - timestamp
        return age < EHRConfig.CACHE_TTL
    
    async def query_resource(
        self,
        resource_type: str,
        query_params: Dict[str, str],
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Query FHIR resource
        
        Args:
            resource_type: FHIR resource type (Patient, Encounter, Observation, etc.)
            query_params: Query parameters (e.g., {"given": "John", "family": "Doe"})
            use_cache: Use cache if available
            
        Returns:
            FHIR Bundle or resource
        """
        cache_key = self._get_cache_key(resource_type, query_params)
        
        # Check cache
        if use_cache and self._is_cache_valid(cache_key):
            logger.info(f"Cache hit for {resource_type}")
            return self.cache[cache_key]["data"]
        
        try:
            if aiohttp is None:
                raise ImportError("aiohttp required for FHIR queries")
            
            # Build URL
            url = f"{self.server_url}/{resource_type}"
            
            # Get auth token if needed
            headers = {
                "Accept": "application/fhir+json",
                "Content-Type": "application/fhir+json",
            }
            
            if self.auth:
                token = await self.auth.get_token()
                headers["Authorization"] = f"Bearer {token}"
            
            logger.info(f"Querying FHIR: {resource_type} with params {query_params}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=query_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=EHRConfig.FHIR_TIMEOUT),
                ) as response:
                    if response.status != 200:
                        logger.error(f"FHIR query failed: {response.status}")
                        return None
                    
                    data = await response.json()
                    
                    # Cache the result
                    if use_cache:
                        self.cache[cache_key] = {
                            "data": data,
                            "timestamp": datetime.now().timestamp(),
                        }
                    
                    return data
        
        except Exception as e:
            logger.error(f"FHIR query error: {e}")
            return None
    
    async def get_patient(
        self,
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
        mrn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get patient information
        
        Args:
            patient_id: FHIR Patient ID
            patient_name: Patient name (given or family)
            mrn: Medical Record Number
            
        Returns:
            Patient resource or None
        """
        if patient_id:
            # Direct ID lookup
            url = f"{self.server_url}/Patient/{patient_id}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
            except:
                pass
        
        # Search by parameters
        query_params = {}
        if patient_name:
            query_params["name"] = patient_name
        if mrn:
            query_params["identifier"] = mrn
        
        if query_params:
            result = await self.query_resource("Patient", query_params)
            if result and result.get("entry"):
                return result["entry"][0]["resource"]
        
        return None
    
    async def get_patient_encounters(
        self,
        patient_id: str,
        status: str = "finished",  # or "in-progress"
    ) -> List[Dict[str, Any]]:
        """Get patient encounters (admissions/visits)
        
        Args:
            patient_id: FHIR Patient ID
            status: Encounter status filter
            
        Returns:
            List of encounters
        """
        result = await self.query_resource(
            "Encounter",
            {
                "patient": patient_id,
                "status": status,
                "_sort": "-date",
                "_count": "50",
            },
        )
        
        if result and result.get("entry"):
            return [entry["resource"] for entry in result["entry"]]
        
        return []
    
    async def get_patient_observations(
        self,
        patient_id: str,
        code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get patient observations (vital signs, lab results)
        
        Args:
            patient_id: FHIR Patient ID
            code: LOINC code filter (e.g., "8480-6" for systolic BP)
            
        Returns:
            List of observations
        """
        params = {
            "patient": patient_id,
            "_sort": "-date",
            "_count": "100",
        }
        
        if code:
            params["code"] = code
        
        result = await self.query_resource("Observation", params)
        
        if result and result.get("entry"):
            return [entry["resource"] for entry in result["entry"]]
        
        return []
    
    async def get_patient_medications(
        self,
        patient_id: str,
    ) -> List[Dict[str, Any]]:
        """Get patient medications
        
        Args:
            patient_id: FHIR Patient ID
            
        Returns:
            List of medication statements
        """
        result = await self.query_resource(
            "MedicationStatement",
            {
                "patient": patient_id,
                "_sort": "-date",
                "_count": "50",
            },
        )
        
        if result and result.get("entry"):
            return [entry["resource"] for entry in result["entry"]]
        
        return []
    
    async def get_patient_allergies(
        self,
        patient_id: str,
    ) -> List[Dict[str, Any]]:
        """Get patient allergies
        
        Args:
            patient_id: FHIR Patient ID
            
        Returns:
            List of allergy intolerance records
        """
        result = await self.query_resource(
            "AllergyIntolerance",
            {
                "patient": patient_id,
                "_sort": "-date",
            },
        )
        
        if result and result.get("entry"):
            return [entry["resource"] for entry in result["entry"]]
        
        return []
    
    async def get_patient_conditions(
        self,
        patient_id: str,
    ) -> List[Dict[str, Any]]:
        """Get patient conditions (diagnoses)
        
        Args:
            patient_id: FHIR Patient ID
            
        Returns:
            List of conditions
        """
        result = await self.query_resource(
            "Condition",
            {
                "patient": patient_id,
                "clinical-status": "active",
                "_sort": "-onset-date",
            },
        )
        
        if result and result.get("entry"):
            return [entry["resource"] for entry in result["entry"]]
        
        return []
    
    async def get_complete_patient_record(
        self,
        patient_id: str,
    ) -> Dict[str, Any]:
        """Get complete patient clinical record
        
        Args:
            patient_id: FHIR Patient ID
            
        Returns:
            Comprehensive patient record
        """
        logger.info(f"Retrieving complete record for patient {patient_id}")
        
        # Parallel queries for efficiency
        tasks = [
            self.get_patient(patient_id=patient_id),
            self.get_patient_encounters(patient_id),
            self.get_patient_observations(patient_id),
            self.get_patient_medications(patient_id),
            self.get_patient_allergies(patient_id),
            self.get_patient_conditions(patient_id),
        ]
        
        try:
            results = await asyncio.gather(*tasks)
            
            return {
                "patient": results[0],
                "encounters": results[1],
                "observations": results[2],
                "medications": results[3],
                "allergies": results[4],
                "conditions": results[5],
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error retrieving complete record: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    async def test_ehr():
        """Test EHR functionality"""
        # Note: This requires a real FHIR server
        client = FHIRClient(
            server_url="https://fhir.hospital.com/api/FHIR/R4",
            client_id="your_client_id",
            client_secret="your_client_secret",
            oauth_url="https://oauth.hospital.com/token",
        )
        
        # Example query
        patient = await client.get_patient(patient_name="John Doe")
        if patient:
            print(f"Found patient: {patient['name']}")
            
            patient_id = patient["id"]
            record = await client.get_complete_patient_record(patient_id)
            print(f"Patient has {len(record['encounters'])} encounters")
    
    # Run test (requires real FHIR server)
    # asyncio.run(test_ehr())
