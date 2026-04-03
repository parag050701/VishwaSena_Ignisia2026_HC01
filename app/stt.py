"""
STT Module - Speech-to-Text using OpenAI Whisper
Handles audio transcription for clinical voice input
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import io

try:
    import whisper
except ImportError:
    whisper = None

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

import numpy as np

logger = logging.getLogger(__name__)


class STTConfig:
    """STT Configuration"""
    MODEL = os.getenv("STT_MODEL", "base")  # tiny, base, small, medium, large
    DEVICE = os.getenv("STT_DEVICE", "cuda")  # cuda or cpu
    LANGUAGE = os.getenv("STT_LANGUAGE", "en")
    CHUNK_SIZE = int(os.getenv("STT_CHUNK_SIZE", "30"))  # seconds
    COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
    TEMPERATURE = 0.0  # Deterministic transcription
    WORD_LEVEL_TIMESTAMPS = True


class WhisperSTT:
    """OpenAI Whisper STT Client"""
    
    def __init__(
        self,
        model_name: str = STTConfig.MODEL,
        device: str = STTConfig.DEVICE,
        use_faster: bool = True,
    ):
        """Initialize Whisper STT client
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: "cuda" or "cpu"
            use_faster: Use faster-whisper if available (recommended)
        """
        self.model_name = model_name
        self.device = device
        self.use_faster = use_faster and FASTER_WHISPER_AVAILABLE
        
        logger.info(f"Loading STT model: {model_name} ({device})")
        try:
            if self.use_faster:
                logger.info("Using faster-whisper for optimized inference")
                self.model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=STTConfig.COMPUTE_TYPE,
                )
            else:
                logger.info("Using standard openai-whisper")
                self.model = whisper.load_model(model_name, device=device)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    async def transcribe_audio(
        self,
        audio_path: str,
        language: str = STTConfig.LANGUAGE,
        include_timestamps: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file (wav, mp3, m4a, flac, etc.)
            language: Language code (e.g., 'en', 'es', 'fr')
            include_timestamps: Include word-level timestamps
            
        Returns:
            Dict with transcription results:
            {
                "text": "full transcription",
                "language": "en",
                "segments": [...],  # individual segments with timing
                "confidence": 0.95,
                "duration": 30.5,
                "words": [...]  # word-level with timing
            }
        """
        loop = asyncio.get_event_loop()
        
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            if self.use_faster:
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: self.model.transcribe(
                        audio=audio_path,
                        language=language,
                        word_level_timestamps=STTConfig.WORD_LEVEL_TIMESTAMPS,
                        temperature=STTConfig.TEMPERATURE,
                    ),
                )
                
                result = {
                    "text": "".join([seg.text for seg in segments]),
                    "language": info.language,
                    "segments": [
                        {
                            "id": i,
                            "seek": 0,
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text,
                            "avg_logprob": seg.avg_logprob if hasattr(seg, 'avg_logprob') else -0.5,
                            "compression_ratio": info.compression_ratio,
                            "no_speech_prob": 0.0,
                        }
                        for i, seg in enumerate(segments)
                    ],
                    "confidence": info.compression_ratio,  # Proxy for confidence
                    "duration": info.duration,
                }
            else:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.model.transcribe(
                        audio=audio_path,
                        language=language,
                        word_level_timestamps=STTConfig.WORD_LEVEL_TIMESTAMPS,
                        temperature=STTConfig.TEMPERATURE,
                    ),
                )
            
            logger.info(
                f"Transcription complete: {result['text'][:50]}... "
                f"(duration: {result.get('duration', 'unknown')}s)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str = STTConfig.LANGUAGE,
    ) -> Dict[str, Any]:
        """Transcribe audio from bytes
        
        Args:
            audio_bytes: Raw audio bytes
            language: Language code
            
        Returns:
            Transcription result dictionary
        """
        # Save bytes to temporary file
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            result = await self.transcribe_audio(tmp_path, language)
            return result
        finally:
            os.unlink(tmp_path)
    
    async def stream_transcribe(
        self,
        audio_chunks: list,
        language: str = STTConfig.LANGUAGE,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Transcribe streaming audio chunks
        
        Args:
            audio_chunks: List of audio chunk bytes
            language: Language code
            sample_rate: Audio sample rate (Hz)
            
        Returns:
            Transcription result
        """
        # Concatenate all chunks
        full_audio = b"".join(audio_chunks)
        return await self.transcribe_bytes(full_audio, language)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model": self.model_name,
            "device": self.device,
            "backend": "faster-whisper" if self.use_faster else "openai-whisper",
            "language": STTConfig.LANGUAGE,
            "max_chunk_duration": STTConfig.CHUNK_SIZE,
        }


class ClinicalSpeechParser:
    """Parse clinical speech patterns and extract entities"""
    
    # Common clinical abbreviations
    CLINICAL_ABBREVS = {
        "BP": "blood pressure",
        "HR": "heart rate",
        "RR": "respiratory rate",
        "O2": "oxygen",
        "SpO2": "oxygen saturation",
        "WBC": "white blood cell",
        "RBC": "red blood cell",
        "Hgb": "hemoglobin",
        "Hct": "hematocrit",
        "PLT": "platelet",
        "Cr": "creatinine",
        "BUN": "blood urea nitrogen",
        "Na": "sodium",
        "K": "potassium",
        "Cl": "chloride",
        "HCO3": "bicarbonate",
        "DBP": "diastolic blood pressure",
        "SBP": "systolic blood pressure",
        "IV": "intravenous",
        "PO": "per oral",
        "IM": "intramuscular",
        "NPO": "nothing by mouth",
        "QID": "four times daily",
        "TID": "three times daily",
        "BID": "twice daily",
        "QD": "once daily",
        "PRN": "as needed",
        "AMS": "altered mental status",
        "SOB": "shortness of breath",
        "MI": "myocardial infarction",
        "CVA": "cerebrovascular accident",
        "PE": "pulmonary embolism",
        "DVT": "deep vein thrombosis",
        "UTI": "urinary tract infection",
        "COPD": "chronic obstructive pulmonary disease",
        "CHF": "congestive heart failure",
        "CAD": "coronary artery disease",
        "DM": "diabetes mellitus",
        "HTN": "hypertension",
        "GERD": "gastroesophageal reflux disease",
        "PUD": "peptic ulcer disease",
        "IBD": "inflammatory bowel disease",
        "CKD": "chronic kidney disease",
        "ESRD": "end-stage renal disease",
        "ICU": "intensive care unit",
        "ED": "emergency department",
        "OR": "operating room",
        "PACU": "post-anesthesia care unit",
    }
    
    @staticmethod
    def expand_abbreviations(text: str) -> str:
        """Expand clinical abbreviations in transcribed text"""
        for abbrev, expansion in ClinicalSpeechParser.CLINICAL_ABBREVS.items():
            # Match case-insensitively but preserve context
            pattern = rf"\b{abbrev}\b"
            import re
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text
    
    @staticmethod
    def extract_clinical_context(text: str) -> Dict[str, Any]:
        """Extract clinical context from transcribed speech
        
        Returns:
            {
                "patient_identifiers": [...],
                "vital_signs": [...],
                "medications": [...],
                "procedures": [...],
                "diagnoses": [...],
                "query_type": "status_check|medication_review|risk_assessment|...",
                "urgency": "routine|urgent|critical",
            }
        """
        # This is a simplified version - would integrate spaCy/transformers in production
        result = {
            "patient_identifiers": [],
            "vital_signs": [],
            "medications": [],
            "procedures": [],
            "diagnoses": [],
            "query_type": "general",
            "urgency": "routine",
        }
        
        # Urgency detection
        urgent_keywords = ["urgent", "critical", "emergency", "stat", "immediately"]
        if any(kw in text.lower() for kw in urgent_keywords):
            result["urgency"] = "critical"
        
        # Query type detection
        if any(kw in text.lower() for kw in ["status", "how is", "what is"]):
            result["query_type"] = "status_check"
        elif any(kw in text.lower() for kw in ["medication", "drug", "prescribe"]):
            result["query_type"] = "medication_review"
        elif any(kw in text.lower() for kw in ["risk", "score", "prognosis"]):
            result["query_type"] = "risk_assessment"
        
        return result


# Example usage
if __name__ == "__main__":
    async def test_stt():
        """Test STT functionality"""
        stt = WhisperSTT(model_name="base")
        
        # Test with sample audio file if available
        test_audio = "test_audio.wav"
        if os.path.exists(test_audio):
            result = await stt.transcribe_audio(test_audio)
            print("Transcription:", result["text"])
            print("Language:", result["language"])
            print("Duration:", result.get("duration"), "seconds")
        
        # Test abbreviation expansion
        text = "Patient has elevated BP and HR, SpO2 normal"
        expanded = ClinicalSpeechParser.expand_abbreviations(text)
        print("Original:", text)
        print("Expanded:", expanded)
    
    # Run test
    asyncio.run(test_stt())
