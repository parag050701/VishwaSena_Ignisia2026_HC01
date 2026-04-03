import os

class Config:
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
    
    # Data paths
    DATA_DIR = os.getenv("HC01_DATA_DIR", ".")
    MIMIC_CSVS = {
        "noteevents": os.path.join(DATA_DIR, "NOTEEVENTS.csv"),
        "labevents": os.path.join(DATA_DIR, "LABEVENTS.csv"),
        "d_labitems": os.path.join(DATA_DIR, "D_LABITEMS.csv"),
        "icustays": os.path.join(DATA_DIR, "ICUSTAYS.csv"),
        "patients": os.path.join(DATA_DIR, "PATIENTS.csv"),
        "prescriptions": os.path.join(DATA_DIR, "PRESCRIPTIONS.csv"),
    }


cfg = Config()
