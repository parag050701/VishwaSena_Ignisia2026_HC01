"""
Voice Workflow Engine - Orchestrates complete voice-to-voice clinical workflows
Handles: STT → EHR → Clinical Analysis → TTS
"""

import asyncio
import logging
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import json

from app.stt import WhisperSTT, ClinicalSpeechParser
from app.tts import TTSManager, ClinicalResponseFormatter
from app.ehr import FHIRClient
from app.nim_client import ChiefModelClient, FallbackModelClient
from app.config import Config

logger = logging.getLogger(__name__)


@dataclass
class VoiceMessage:
    """Voice message in conversation"""
    role: str  # "user" or "assistant"
    content: str
    audio_bytes: Optional[bytes] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VoiceSession:
    """Voice session context"""
    session_id: str
    patient_id: Optional[str] = None
    clinician_id: Optional[str] = None
    messages: List[VoiceMessage] = None
    context: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.context is None:
            self.context = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class VoiceWorkflowEngine:
    """Orchestrates complete voice-driven clinical workflows"""
    
    def __init__(
        self,
        fhir_server_url: Optional[str] = None,
        fhir_client_id: Optional[str] = None,
        fhir_client_secret: Optional[str] = None,
        fhir_oauth_url: Optional[str] = None,
        use_chief_model: bool = True,
    ):
        """Initialize Voice Workflow Engine
        
        Args:
            fhir_server_url: FHIR server URL
            fhir_client_id: OAuth client ID
            fhir_client_secret: OAuth client secret
            fhir_oauth_url: OAuth token endpoint
            use_chief_model: Use Chief model (True) or Fallback model (False)
        """
        self.stt = WhisperSTT(model_name="base")
        self.tts = TTSManager(preferred_engine="kokoro")
        
        # Initialize EHR client if credentials provided
        self.ehr = None
        if fhir_server_url:
            self.ehr = FHIRClient(
                server_url=fhir_server_url,
                client_id=fhir_client_id,
                client_secret=fhir_client_secret,
                oauth_url=fhir_oauth_url,
            )
        
        # Initialize clinical AI models
        if use_chief_model:
            self.ai_model = ChiefModelClient()
        else:
            self.ai_model = FallbackModelClient()
        
        self.use_chief_model = use_chief_model
        
        # Session management
        self.sessions: Dict[str, VoiceSession] = {}
        self.callbacks: Dict[str, List[Callable]] = {
            "on_speech_started": [],
            "on_transcription": [],
            "on_ehr_loaded": [],
            "on_analysis_started": [],
            "on_analysis_complete": [],
            "on_tts_started": [],
            "on_response_ready": [],
        }
        
        logger.info("Voice Workflow Engine initialized")
    
    def register_callback(self, event: str, callback: Callable):
        """Register event callback
        
        Args:
            event: Event name
            callback: Async callback function
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    async def _trigger_event(self, event: str, data: Any = None):
        """Trigger event callbacks"""
        if event in self.callbacks:
            tasks = [
                callback(data) if asyncio.iscoroutinefunction(callback)
                else callback(data)
                for callback in self.callbacks[event]
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def create_session(
        self,
        session_id: str,
        patient_id: Optional[str] = None,
        clinician_id: Optional[str] = None,
    ) -> VoiceSession:
        """Create new voice session
        
        Args:
            session_id: Unique session identifier
            patient_id: Optional patient identifier
            clinician_id: Optional clinician identifier
            
        Returns:
            VoiceSession
        """
        session = VoiceSession(
            session_id=session_id,
            patient_id=patient_id,
            clinician_id=clinician_id,
        )
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id}")
        return session
    
    async def process_voice_input(
        self,
        session_id: str,
        audio_bytes: bytes,
    ) -> Dict[str, Any]:
        """Process voice input and generate voice response
        
        Args:
            session_id: Session ID
            audio_bytes: Raw audio bytes
            
        Returns:
            Response with transcription, analysis, and audio output
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        logger.info(f"Processing voice input for session {session_id}")
        await self._trigger_event("on_speech_started", {"session_id": session_id})
        
        try:
            # Step 1: Speech-to-Text
            logger.info("Step 1: Transcribing audio...")
            transcription = await self.stt.transcribe_bytes(audio_bytes)
            user_text = transcription["text"]
            
            logger.info(f"Transcribed: {user_text[:100]}...")
            await self._trigger_event("on_transcription", {
                "session_id": session_id,
                "text": user_text,
            })
            
            # Step 2: Extract clinical context
            logger.info("Step 2: Extracting clinical context...")
            clinical_context = ClinicalSpeechParser.extract_clinical_context(user_text)
            expanded_text = ClinicalSpeechParser.expand_abbreviations(user_text)
            
            session.context.update({
                "last_query": expanded_text,
                "clinical_context": clinical_context,
                "query_type": clinical_context["query_type"],
                "urgency": clinical_context["urgency"],
            })
            
            # Step 3: Load EHR data if patient specified
            patient_data = None
            if session.patient_id and self.ehr:
                logger.info("Step 3: Loading EHR data...")
                try:
                    patient_data = await self.ehr.get_complete_patient_record(
                        session.patient_id
                    )
                    session.context["ehr_data"] = patient_data
                    
                    logger.info(f"Loaded EHR data for patient {session.patient_id}")
                    await self._trigger_event("on_ehr_loaded", {
                        "session_id": session_id,
                        "patient_id": session.patient_id,
                    })
                except Exception as e:
                    logger.error(f"EHR data load failed: {e}")
                    patient_data = None
            
            # Step 4: Prepare AI analysis input
            logger.info("Step 4: Preparing AI analysis...")
            await self._trigger_event("on_analysis_started", {
                "session_id": session_id,
                "model": "chief" if self.use_chief_model else "fallback",
            })
            
            # Build clinical context for AI
            ehr_summary = self._summarize_ehr_data(patient_data) if patient_data else ""
            ai_prompt = self._build_ai_prompt(
                user_query=expanded_text,
                clinical_context=clinical_context,
                ehr_summary=ehr_summary,
                session_context=session.context,
            )
            
            # Step 5: Clinical AI Analysis
            logger.info("Step 5: Running clinical analysis...")
            if self.use_chief_model:
                # Use Chief model with extended thinking
                ai_response = await self.ai_model.reason(
                    prompt=ai_prompt,
                    reasoning_budget=8000,
                    max_tokens=2000,
                )
            else:
                # Use Fallback model for quick response
                ai_response = await self.ai_model.document(
                    prompt=ai_prompt,
                    max_tokens=1024,
                )
            
            logger.info(f"Analysis complete: {ai_response[:100]}...")
            await self._trigger_event("on_analysis_complete", {
                "session_id": session_id,
                "response_length": len(ai_response),
            })
            
            # Step 6: Format response for speech
            logger.info("Step 6: Formatting response for speech...")
            speech_text = ClinicalResponseFormatter.prepare_for_speech(ai_response)
            
            # Step 7: Text-to-Speech
            logger.info("Step 7: Synthesizing speech...")
            await self._trigger_event("on_tts_started", {
                "session_id": session_id,
                "text_length": len(speech_text),
            })
            
            response_audio = await self.tts.synthesize(
                speech_text,
                speed=1.0,
            )
            
            logger.info(f"Speech synthesis complete: {len(response_audio)} bytes")
            await self._trigger_event("on_response_ready", {
                "session_id": session_id,
                "audio_size": len(response_audio),
            })
            
            # Step 8: Store in session history
            user_msg = VoiceMessage(
                role="user",
                content=user_text,
                audio_bytes=audio_bytes,
                metadata=clinical_context,
            )
            assistant_msg = VoiceMessage(
                role="assistant",
                content=ai_response,
                audio_bytes=response_audio,
                metadata={"model": "chief" if self.use_chief_model else "fallback"},
            )
            
            session.messages.append(user_msg)
            session.messages.append(assistant_msg)
            session.updated_at = datetime.now()
            
            return {
                "success": True,
                "session_id": session_id,
                "user_input": user_text,
                "user_context": clinical_context,
                "ai_response_text": ai_response,
                "response_audio_bytes": response_audio,
                "response_audio_length_seconds": len(response_audio) / (24000 * 2),  # Kokoro is 24kHz 16-bit
                "model_used": "chief" if self.use_chief_model else "fallback",
                "processing_complete": True,
            }
        
        except Exception as e:
            logger.error(f"Voice processing error: {e}", exc_info=True)
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e),
            }
    
    def _summarize_ehr_data(self, patient_data: Dict[str, Any]) -> str:
        """Summarize EHR data for AI context
        
        Args:
            patient_data: FHIR patient record
            
        Returns:
            Text summary of key EHR data
        """
        lines = []
        
        try:
            # Patient demographics
            patient = patient_data.get("patient", {})
            if patient:
                name = self._extract_name(patient)
                if name:
                    lines.append(f"Patient: {name}")
            
            # Active conditions
            conditions = patient_data.get("conditions", [])
            if conditions:
                condition_names = [
                    c.get("code", {}).get("coding", [{}])[0].get("display", "Unknown")
                    for c in conditions[:5]
                ]
                lines.append(f"Active conditions: {', '.join(condition_names)}")
            
            # Current medications
            medications = patient_data.get("medications", [])
            if medications:
                med_names = [
                    m.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display")
                    or m.get("medicationReference", {}).get("display", "Unknown")
                    for m in medications[:5]
                ]
                lines.append(f"Current medications: {', '.join(med_names)}")
            
            # Recent vitals
            observations = patient_data.get("observations", [])
            if observations:
                vitals = []
                for obs in observations[:10]:
                    code = obs.get("code", {}).get("coding", [{}])[0].get("display", "")
                    value = obs.get("valueQuantity", {}).get("value", "")
                    if value:
                        vitals.append(f"{code}: {value}")
                if vitals:
                    lines.append(f"Recent vitals: {', '.join(vitals)}")
            
            # Allergies
            allergies = patient_data.get("allergies", [])
            if allergies:
                allergy_names = [
                    a.get("code", {}).get("coding", [{}])[0].get("display", "Unknown")
                    for a in allergies
                ]
                lines.append(f"Allergies: {', '.join(allergy_names)}")
        
        except Exception as e:
            logger.error(f"Error summarizing EHR data: {e}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _extract_name(patient: Dict[str, Any]) -> str:
        """Extract patient name from FHIR resource"""
        names = patient.get("name", [])
        if names:
            name = names[0]
            given = " ".join(name.get("given", []))
            family = name.get("family", "")
            return f"{given} {family}".strip()
        return ""
    
    def _build_ai_prompt(
        self,
        user_query: str,
        clinical_context: Dict[str, Any],
        ehr_summary: str,
        session_context: Dict[str, Any],
    ) -> str:
        """Build comprehensive AI prompt from all context
        
        Args:
            user_query: User's query text
            clinical_context: Extracted clinical context
            ehr_summary: EHR data summary
            session_context: Session-level context
            
        Returns:
            Formatted prompt for clinical AI
        """
        prompt = f"""You are an expert clinical decision support system. A clinician is asking about a patient.

## Clinical Conversation History
{chr(10).join([f"- {msg.role.upper()}: {msg.content[:100]}" for msg in self.sessions.get(
    list(self.sessions.keys())[0], VoiceSession(session_id="")
).messages[-4:] if hasattr(self.sessions, 'get')])}

## Current Query
{user_query}

## Query Context
- Type: {clinical_context.get('query_type', 'general')}
- Urgency: {clinical_context.get('urgency', 'routine')}

## Patient Clinical Information
{ehr_summary if ehr_summary else "No EHR data available - using knowledge-based analysis"}

## Your Response
Provide a clinically accurate, concise response:
1. Direct answer to the clinician's question
2. Key clinical findings relevant to the query
3. Risk assessment (if applicable)
4. Recommended next steps or monitoring
5. Clinical guidelines cited (if applicable)

Be direct and professional. Use standard medical terminology. Highlight critical findings.
"""
        return prompt
    
    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [
            {
                "session_id": s.session_id,
                "patient_id": s.patient_id,
                "clinician_id": s.clinician_id,
                "messages": len(s.messages),
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in self.sessions.values()
        ]


# Example usage
if __name__ == "__main__":
    async def demo_voice_workflow():
        """Demo voice workflow"""
        # Initialize engine (without EHR for this demo)
        engine = VoiceWorkflowEngine(
            use_chief_model=True,
        )
        
        # Create session
        session = engine.create_session(
            session_id="demo_123",
            patient_id=None,  # No real patient
        )
        
        print("Voice Workflow Engine Demo")
        print("=" * 50)
        print(f"Session created: {session.session_id}")
        print(f"Status: Ready to process voice input")
        print("=" * 50)
    
    asyncio.run(demo_voice_workflow())
