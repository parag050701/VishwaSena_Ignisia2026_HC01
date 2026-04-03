"""
TTS Module - Text-to-Speech with Coqui TTS or Kokoro
Handles voice synthesis for clinical responses
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, Union
import io
from pathlib import Path

logger = logging.getLogger(__name__)


class TTSConfig:
    """TTS Configuration"""
    ENGINE = os.getenv("TTS_ENGINE", "kokoro")  # "kokoro" or "coqui"
    VOICE = os.getenv("TTS_VOICE", "female")  # "female" or "male"
    SPEED = float(os.getenv("TTS_SPEED", "1.0"))
    DEVICE = os.getenv("TTS_DEVICE", "cuda")  # "cuda" or "cpu"
    LANGUAGE = os.getenv("TTS_LANGUAGE", "en")
    SAMPLE_RATE = 24000  # Hz


class KokoroTTS:
    """Kokoro TTS Client - Modern, natural speech synthesis"""
    
    def __init__(self, device: str = TTSConfig.DEVICE):
        """Initialize Kokoro TTS
        
        Args:
            device: "cuda" or "cpu"
        """
        self.device = device
        self.voice_name = "af"  # Default female voice
        
        try:
            logger.info(f"Loading Kokoro TTS model ({device})")
            # Kokoro model loading
            try:
                from kokoro import generate
                self.generate = generate
                self.is_available = True
                logger.info("Kokoro TTS loaded successfully")
            except ImportError:
                logger.warning("Kokoro not installed. Install with: pip install git+https://github.com/hexgrad/Kokoro.git")
                self.is_available = False
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro: {e}")
            self.is_available = False
    
    async def synthesize(
        self,
        text: str,
        voice: str = "af",  # "af", "am", "bf", "bm" (female/male variants)
        speed: float = TTSConfig.SPEED,
        return_bytes: bool = True,
    ) -> Union[bytes, str]:
        """Synthesize text to speech
        
        Args:
            text: Clinical text to synthesize
            voice: Voice variant (af=american female, am=american male, etc.)
            speed: Speech speed multiplier (0.5-2.0)
            return_bytes: Return audio bytes (True) or file path (False)
            
        Returns:
            Audio bytes or file path
        """
        if not self.is_available:
            logger.error("Kokoro TTS not available")
            return None
        
        loop = asyncio.get_event_loop()
        
        try:
            logger.info(f"Synthesizing with Kokoro: {text[:50]}...")
            
            # Run synthesis in executor (blocking operation)
            audio = await loop.run_in_executor(
                None,
                lambda: self.generate(
                    text,
                    voice=voice,
                    speed=speed,
                ),
            )
            
            logger.info(f"Synthesis complete: {len(audio)} bytes")
            
            return audio if return_bytes else audio
            
        except Exception as e:
            logger.error(f"Kokoro synthesis error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "engine": "kokoro",
            "device": self.device,
            "available": self.is_available,
            "sample_rate": TTSConfig.SAMPLE_RATE,
            "voices": ["af", "am", "bf", "bm"],  # American female, male, British female, male
        }


class CoquiTTS:
    """Coqui TTS Client - Flexible, open-source speech synthesis"""
    
    def __init__(self, device: str = TTSConfig.DEVICE):
        """Initialize Coqui TTS
        
        Args:
            device: "cuda" or "cpu"
        """
        self.device = device
        
        try:
            logger.info(f"Loading Coqui TTS model ({device})")
            from TTS.api import TTS
            self.tts = TTS(
                model_name="tts_models/en/ljspeech/glow-tts",
                device=device,
                progress_bar=False,
            )
            self.is_available = True
            logger.info("Coqui TTS loaded successfully")
        except ImportError:
            logger.warning("Coqui TTS not installed. Install with: pip install TTS")
            self.tts = None
            self.is_available = False
        except Exception as e:
            logger.error(f"Failed to initialize Coqui: {e}")
            self.is_available = False
    
    async def synthesize(
        self,
        text: str,
        speed: float = TTSConfig.SPEED,
        emotion: str = "neutral",
        return_bytes: bool = True,
    ) -> Union[bytes, str]:
        """Synthesize text to speech
        
        Args:
            text: Clinical text to synthesize
            speed: Speech speed multiplier (0.5-2.0)
            emotion: Speech emotion (neutral, happy, sad, angry)
            return_bytes: Return audio bytes (True) or file path (False)
            
        Returns:
            Audio bytes or file path
        """
        if not self.is_available or self.tts is None:
            logger.error("Coqui TTS not available")
            return None
        
        loop = asyncio.get_event_loop()
        
        try:
            logger.info(f"Synthesizing with Coqui: {text[:50]}...")
            
            # Run synthesis in executor
            output_path = await loop.run_in_executor(
                None,
                lambda: self.tts.tts_to_file(
                    text=text,
                    file_path="tmp_tts_output.wav",
                    speed=speed,
                ),
            )
            
            if return_bytes:
                # Read file and return bytes
                with open(output_path, "rb") as f:
                    audio = f.read()
                os.unlink(output_path)
                logger.info(f"Synthesis complete: {len(audio)} bytes")
                return audio
            else:
                logger.info(f"Synthesis complete: {output_path}")
                return output_path
            
        except Exception as e:
            logger.error(f"Coqui synthesis error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "engine": "coqui",
            "device": self.device,
            "available": self.is_available,
            "model": "glow-tts",
            "sample_rate": 22050,
        }


class TTSManager:
    """Manages TTS engine selection and fallback"""
    
    def __init__(self, preferred_engine: str = TTSConfig.ENGINE):
        """Initialize TTS Manager
        
        Args:
            preferred_engine: "kokoro" (default) or "coqui"
        """
        self.preferred_engine = preferred_engine
        self.kokoro = None
        self.coqui = None
        self.active_engine = None
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize both TTS engines, fall back if needed"""
        # Try preferred engine first
        if self.preferred_engine == "kokoro":
            self.kokoro = KokoroTTS(device=TTSConfig.DEVICE)
            if self.kokoro.is_available:
                self.active_engine = "kokoro"
                logger.info("Using Kokoro TTS engine")
            else:
                logger.warning("Kokoro unavailable, trying Coqui...")
                self.coqui = CoquiTTS(device=TTSConfig.DEVICE)
                if self.coqui.is_available:
                    self.active_engine = "coqui"
                    logger.info("Fallback to Coqui TTS engine")
        else:
            self.coqui = CoquiTTS(device=TTSConfig.DEVICE)
            if self.coqui.is_available:
                self.active_engine = "coqui"
                logger.info("Using Coqui TTS engine")
            else:
                logger.warning("Coqui unavailable, trying Kokoro...")
                self.kokoro = KokoroTTS(device=TTSConfig.DEVICE)
                if self.kokoro.is_available:
                    self.active_engine = "kokoro"
                    logger.info("Fallback to Kokoro TTS engine")
        
        if not self.active_engine:
            logger.error("No TTS engine available!")
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = TTSConfig.SPEED,
        **kwargs
    ) -> bytes:
        """Synthesize text with active engine
        
        Args:
            text: Text to synthesize
            voice: Voice name (optional)
            speed: Speech speed
            **kwargs: Engine-specific parameters
            
        Returns:
            Audio bytes
        """
        if not self.active_engine:
            raise RuntimeError("No TTS engine available")
        
        try:
            if self.active_engine == "kokoro" and self.kokoro:
                return await self.kokoro.synthesize(
                    text,
                    voice=voice or "af",
                    speed=speed,
                    **kwargs
                )
            elif self.active_engine == "coqui" and self.coqui:
                return await self.coqui.synthesize(
                    text,
                    speed=speed,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Synthesis with {self.active_engine} failed: {e}")
            # Fallback to alternative engine if available
            if self.active_engine == "kokoro" and self.coqui and self.coqui.is_available:
                logger.info("Falling back to Coqui TTS")
                self.active_engine = "coqui"
                return await self.coqui.synthesize(text, speed=speed, **kwargs)
            elif self.active_engine == "coqui" and self.kokoro and self.kokoro.is_available:
                logger.info("Falling back to Kokoro TTS")
                self.active_engine = "kokoro"
                return await self.kokoro.synthesize(text, voice=voice or "af", speed=speed, **kwargs)
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get TTS manager status"""
        return {
            "active_engine": self.active_engine,
            "kokoro_available": self.kokoro.is_available if self.kokoro else False,
            "coqui_available": self.coqui.is_available if self.coqui else False,
            "kokoro_info": self.kokoro.get_model_info() if self.kokoro else None,
            "coqui_info": self.coqui.get_model_info() if self.coqui else None,
        }


class ClinicalResponseFormatter:
    """Format AI responses for natural speech synthesis"""
    
    @staticmethod
    def prepare_for_speech(text: str) -> str:
        """Prepare text for TTS synthesis by normalizing clinical content
        
        Args:
            text: Raw clinical response text
            
        Returns:
            Text optimized for natural speech
        """
        # Expand common abbreviations that weren't expanded earlier
        replacements = {
            r"\b(\d+)/(\d+)\b": r"\1 over \2",  # 3/10 → 3 over 10
            r"\b(\d+)x(\d+)\b": r"\1 by \2",    # 3x5 → 3 by 5
            "±": "plus or minus",
            "→": "leads to",
            "↑": "up",
            "↓": "down",
            "approx.": "approximately",
            "etc.": "et cetera",
            "vs.": "versus",
        }
        
        import re
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove markdown formatting
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold** → bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)      # *italic* → italic
        text = re.sub(r"`(.+?)`", r"\1", text)        # `code` → code
        
        # Clean up excessive punctuation
        text = re.sub(r"\s+", " ", text)  # Multiple spaces → single space
        text = text.strip()
        
        return text


# Example usage
if __name__ == "__main__":
    async def test_tts():
        """Test TTS functionality"""
        manager = TTSManager(preferred_engine="kokoro")
        
        # Test synthesis
        text = "Patient admitted with elevated blood pressure and rapid heart rate."
        
        status = manager.get_status()
        print("TTS Status:", status)
        
        try:
            audio = await manager.synthesize(text, speed=1.0)
            print(f"Synthesized {len(audio)} bytes of audio")
            
            # Save to file for testing
            with open("test_output.wav", "wb") as f:
                f.write(audio)
            print("Audio saved to test_output.wav")
        except Exception as e:
            print(f"TTS error: {e}")
    
    # Run test
    asyncio.run(test_tts())
