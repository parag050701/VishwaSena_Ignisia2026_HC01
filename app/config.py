import os

class Config:
    # ═══════════════════════════════════════════════════════════
    # CLINICAL AI MODELS
    # ═══════════════════════════════════════════════════════════
    OLLAMA_BASE = "http://localhost:11434"
    NIM_BASE = "https://integrate.api.nvidia.com/v1"

    NOTE_MODEL = "qwen2.5:7b-q4_k_m"
    RAG_EXPLAIN_MODEL = "qwen2.5:7b-q4_k_m"
    EMBED_MODEL = "bge-m3"

    CHIEF_MODEL = "nvidia/nemotron-3-super-120b-a12b"
    FALLBACK_MODEL = "qwen/qwen2.5-7b-instruct"

    # NIM API Keys (set via environment or config)
    NIM_API_KEY_CHIEF = os.getenv("NIM_API_KEY_CHIEF", "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")
    NIM_API_KEY_FALLBACK = os.getenv("NIM_API_KEY_FALLBACK", "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")

    OUTLIER_Z = 2.5
    TOP_K_GUIDELINES = 5
    OLLAMA_TIMEOUT = 120.0
    NIM_TIMEOUT = 90.0
    
    # ═══════════════════════════════════════════════════════════
    # VOICE SETTINGS (NEW)
    # ═══════════════════════════════════════════════════════════
    # Speech-to-Text (STT) Configuration
    STT_MODEL = os.getenv("STT_MODEL", "base")  # tiny, base, small, medium, large
    STT_DEVICE = os.getenv("STT_DEVICE", "cuda")  # cuda or cpu
    STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")
    STT_CHUNK_SIZE = int(os.getenv("STT_CHUNK_SIZE", "30"))  # seconds
    
    # Text-to-Speech (TTS) Configuration
    TTS_ENGINE = os.getenv("TTS_ENGINE", "kokoro")  # kokoro or coqui
    TTS_VOICE = os.getenv("TTS_VOICE", "female")  # female or male
    TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))  # 0.5-2.0
    TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda")  # cuda or cpu
    TTS_SAMPLE_RATE = 24000  # Hz (Kokoro native)
    
    # ═══════════════════════════════════════════════════════════
    # EHR/FHIR CONFIGURATION (NEW)
    # ═══════════════════════════════════════════════════════════
    FHIR_SERVER_URL = os.getenv("FHIR_SERVER_URL", "")
    FHIR_CLIENT_ID = os.getenv("FHIR_CLIENT_ID", "")
    FHIR_CLIENT_SECRET = os.getenv("FHIR_CLIENT_SECRET", "")
    FHIR_OAUTH_URL = os.getenv("FHIR_OAUTH_URL", "")
    FHIR_SCOPE = os.getenv("FHIR_SCOPE", "user/Patient.read user/Observation.read user/Encounter.read")
    FHIR_TIMEOUT = 30  # seconds
    FHIR_CACHE_TTL = 300  # 5 minutes
    
    # ═══════════════════════════════════════════════════════════
    # DATA PATHS
    # ═══════════════════════════════════════════════════════════
    DATA_DIR = os.getenv("HC01_DATA_DIR", ".")
    MIMIC_CSVS = {
        "noteevents": os.path.join(DATA_DIR, "NOTEEVENTS.csv"),
        "labevents": os.path.join(DATA_DIR, "LABEVENTS.csv"),
        "d_labitems": os.path.join(DATA_DIR, "D_LABITEMS.csv"),
        "icustays": os.path.join(DATA_DIR, "ICUSTAYS.csv"),
        "patients": os.path.join(DATA_DIR, "PATIENTS.csv"),
        "prescriptions": os.path.join(DATA_DIR, "PRESCRIPTIONS.csv"),
    }
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE FLAGS
    # ═══════════════════════════════════════════════════════════
    ENABLE_VOICE = True  # Enable voice capabilities
    ENABLE_EHR = bool(FHIR_SERVER_URL)  # Enable EHR only if configured
    USE_CHIEF_MODEL_FOR_VOICE = True  # Use advanced reasoning for voice
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"


cfg = Config()
