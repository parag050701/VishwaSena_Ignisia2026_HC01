import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


class Config:
    # ═══════════════════════════════════════════════════════════
    # CLINICAL AI MODELS
    # ═══════════════════════════════════════════════════════════
    OLLAMA_BASE: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    NIM_BASE: str = "https://integrate.api.nvidia.com/v1"

    NOTE_MODEL: str = os.getenv("NOTE_MODEL", "qwen3:4b")
    RAG_EXPLAIN_MODEL: str = os.getenv("RAG_EXPLAIN_MODEL", "qwen3:4b")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

    CHIEF_MODEL: str = os.getenv(
        "CHIEF_MODEL", "nvidia/nemotron-3-super-120b-a12b"
    )
    FALLBACK_MODEL: str = os.getenv(
        "FALLBACK_MODEL", "qwen/qwen2.5-7b-instruct"
    )

    # API keys — never hardcoded; must be set in .env
    NIM_API_KEY_CHIEF: str = os.getenv("NIM_API_KEY_CHIEF", "")
    NIM_API_KEY_FALLBACK: str = os.getenv("NIM_API_KEY_FALLBACK", "")
    # Fast assistant model (Llama 3 8B) — lower latency for ward queries
    NIM_API_KEY_FAST: str = os.getenv("NIM_API_KEY_FAST", "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")
    FAST_MODEL: str = os.getenv("FAST_MODEL", "meta/llama3-8b-instruct")
    NIM_EMBED_MODEL: str = os.getenv("NIM_EMBED_MODEL", "bge-large:335m")

    # ═══════════════════════════════════════════════════════════
    # PIPELINE TUNING
    # ═══════════════════════════════════════════════════════════
    OUTLIER_Z: float = float(os.getenv("OUTLIER_Z", "2.5"))
    TOP_K_GUIDELINES: int = int(os.getenv("TOP_K_GUIDELINES", "5"))
    OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "180.0"))
    NIM_TIMEOUT: float = float(os.getenv("NIM_TIMEOUT", "45.0"))
    NIM_RETRY_ATTEMPTS: int = int(os.getenv("NIM_RETRY_ATTEMPTS", "1"))

    # Optional LLM council consult (Machine Theory / Andrej Karpathy project)
    ENABLE_LLM_COUNCIL: bool = os.getenv("ENABLE_LLM_COUNCIL", "false").lower() == "true"
    LLM_COUNCIL_SPACE: str = os.getenv("LLM_COUNCIL_SPACE", "burtenshaw/karpathy-llm-council")
    LLM_COUNCIL_TIMEOUT: float = float(os.getenv("LLM_COUNCIL_TIMEOUT", "180.0"))

    # ═══════════════════════════════════════════════════════════
    # VOICE SETTINGS
    # ═══════════════════════════════════════════════════════════
    STT_MODEL: str = os.getenv("STT_MODEL", "base")
    STT_DEVICE: str = os.getenv("STT_DEVICE", "cuda")
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "en")
    STT_CHUNK_SIZE: int = int(os.getenv("STT_CHUNK_SIZE", "30"))

    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "kokoro")
    TTS_VOICE: str = os.getenv("TTS_VOICE", "female")
    TTS_SPEED: float = float(os.getenv("TTS_SPEED", "1.0"))
    TTS_DEVICE: str = os.getenv("TTS_DEVICE", "cuda")
    TTS_SAMPLE_RATE: int = 24000

    # NIM Speech API keys
    NIM_STT_API_KEY: str = os.getenv("NIM_STT_API_KEY", "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")
    NIM_TTS_API_KEY: str = os.getenv("NIM_TTS_API_KEY", "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")
    NIM_TTS_VOICE: str = os.getenv("NIM_TTS_VOICE", "Magpie-Multilingual.EN-US.Aria")
    NIM_SPEECH_BASE: str = "https://integrate.api.nvidia.com/v1"

    # ═══════════════════════════════════════════════════════════
    # EHR / FHIR
    # ═══════════════════════════════════════════════════════════
    FHIR_SERVER_URL: str = os.getenv("FHIR_SERVER_URL", "")
    FHIR_CLIENT_ID: str = os.getenv("FHIR_CLIENT_ID", "")
    FHIR_CLIENT_SECRET: str = os.getenv("FHIR_CLIENT_SECRET", "")
    FHIR_OAUTH_URL: str = os.getenv("FHIR_OAUTH_URL", "")
    FHIR_SCOPE: str = os.getenv(
        "FHIR_SCOPE",
        "user/Patient.read user/Observation.read user/Encounter.read",
    )
    FHIR_TIMEOUT: int = int(os.getenv("FHIR_TIMEOUT", "30"))
    FHIR_CACHE_TTL: int = int(os.getenv("FHIR_CACHE_TTL", "300"))

    # ═══════════════════════════════════════════════════════════
    # DATA PATHS
    # ═══════════════════════════════════════════════════════════
    DATA_DIR: str = os.getenv("HC01_DATA_DIR", ".")
    MEDICAL_GUIDELINES_DIR: str = os.getenv("MEDICAL_GUIDELINES_DIR", "data/medical_guidelines")
    MEDICAL_RAG_DB_DIR: str = os.getenv("MEDICAL_RAG_DB_DIR", "data/medical_rag_db")
    MIMIC_CSVS: dict = {
        "noteevents":    os.path.join(os.getenv("HC01_DATA_DIR", "."), "NOTEEVENTS.csv"),
        "labevents":     os.path.join(os.getenv("HC01_DATA_DIR", "."), "LABEVENTS.csv"),
        "d_labitems":    os.path.join(os.getenv("HC01_DATA_DIR", "."), "D_LABITEMS.csv"),
        "icustays":      os.path.join(os.getenv("HC01_DATA_DIR", "."), "ICUSTAYS.csv"),
        "patients":      os.path.join(os.getenv("HC01_DATA_DIR", "."), "PATIENTS.csv"),
        "prescriptions": os.path.join(os.getenv("HC01_DATA_DIR", "."), "PRESCRIPTIONS.csv"),
    }

    # ═══════════════════════════════════════════════════════════
    # FEATURE FLAGS
    # ═══════════════════════════════════════════════════════════
    ENABLE_VOICE: bool = os.getenv("ENABLE_VOICE", "true").lower() == "true"
    ENABLE_EHR: bool = bool(os.getenv("FHIR_SERVER_URL", ""))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    def has_nim_key(self, key_type: str = "fallback") -> bool:
        """Check if a NIM key is configured."""
        if key_type == "chief":
            return bool(self.NIM_API_KEY_CHIEF)
        return bool(self.NIM_API_KEY_FALLBACK)

    def nim_key(self, key_type: str = "fallback") -> str:
        """Return the best available NIM key."""
        if key_type == "chief" and self.NIM_API_KEY_CHIEF:
            return self.NIM_API_KEY_CHIEF
        if self.NIM_API_KEY_FALLBACK:
            return self.NIM_API_KEY_FALLBACK
        if self.NIM_API_KEY_CHIEF:
            return self.NIM_API_KEY_CHIEF
        return ""


cfg = Config()
