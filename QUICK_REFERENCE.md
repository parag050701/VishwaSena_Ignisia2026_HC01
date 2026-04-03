# IGNISIA Quick Reference

## Start Here

### Prerequisites
- Python 3.8+ (already installed)
- All dependencies (`pip install -r requirements.txt` - done)
- Ollama running (`ollama serve`)
- NIM API keys configured

### One-Line Setup
```bash
export NIM_API_KEY_CHIEF="nvapi-xxx" && \
export NIM_API_KEY_FALLBACK="nvapi-yyy" && \
python demo.py
```

---

## Service URLs

| Service | URL | Status |
|---------|-----|--------|
| Ollama Local | http://localhost:11434 | Local (start with `ollama serve`) |
| NIM Remote | https://integrate.api.nvidia.com/v1 | Cloud (requires API key) |

---

## Key Files

```
demo.py                  → Run the full pipeline demo
app/config.py           → API keys, model names, timeouts
app/data_loader.py      → Load MIMIC patient data
app/clients.py          → Ollama & NIM API wrappers
app/data.py             → Clinical guidelines reference
SETUP.md                → Detailed setup guide
README.md               → Architecture & usage docs
```

---

## Common Commands

### Run Demo
```bash
python demo.py
```

### Test Ollama
```bash
curl http://localhost:11434/api/tags
ollama pull qwen2.5:7b-q4_k_m
```

### Test NIM API
```bash
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
  -H "Authorization: Bearer $NIM_API_KEY_FALLBACK" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen/qwen2.5-7b-instruct","messages":[{"role":"user","content":"test"}]}'
```

### Quick Python Test
```python
import asyncio
from app.data_loader import load_patient_data

# Load real patient from MIMIC
data = load_patient_data(93810, 193810)
print(data)
```

---

## API Keys

### Get Keys
1. Go to https://build.nvidia.com/
2. Create account/sign in
3. Generate API key for each model

### Set Keys (Choose One)

**Temporarily (current session only)**
```bash
export NIM_API_KEY_CHIEF="nvapi-xxx"
export NIM_API_KEY_FALLBACK="nvapi-yyy"
```

**Permanently (bashrc/zshrc)**
```bash
echo 'export NIM_API_KEY_CHIEF="nvapi-xxx"' >> ~/.bashrc
echo 'export NIM_API_KEY_FALLBACK="nvapi-yyy"' >> ~/.bashrc
source ~/.bashrc
```

**In Code (app/config.py)**
```python
NIM_API_KEY_CHIEF = "nvapi-xxx"
NIM_API_KEY_FALLBACK = "nvapi-yyy"
```

---

## Clinical Workflow

```
1. Load Patient Data
   ↓
   [Patient vitals, notes, labs from MIMIC CSV]
   
2. Extract Information (Ollama)
   ↓
   [Chief complaint, vital signs status, key findings]
   
3. Retrieve Guidelines (RAG + Local Embedding)
   ↓
   [Relevant clinical protocols & evidence]
   
4. Clinical Reasoning (Chief Model via NIM)
   ↓
   [Assessment, risk factors, initial care plan]
   
5. Documentation (Fallback Model via NIM)
   ↓
   [Clinical note in standard medical notation]
```

---

## Performance Targets

| Stage | Model | Latency | Location |
|-------|-------|---------|----------|
| Note Extraction | Ollama (Qwen 7B) | 5-15s | Local |
| Guideline Retrieval | Ollama (BGE-M3) | <1s | Local |
| Clinical Reasoning | NIM (Nemotron 120B) | 15-30s | Remote |
| Documentation | NIM (Qwen 7B) | 10-20s | Remote |
| **Total** | - | **~40-65s** | Mixed |

---

## Troubleshooting Quick Fixes

| Problem | Fix |
|---------|-----|
| Ollama not connecting | `ollama serve` in another terminal |
| Model missing | `ollama pull qwen2.5:7b-q4_k_m` |
| NIM auth fails | Check API key is set: `echo $NIM_API_KEY_CHIEF` |
| Timeout | Increase timeout in `config.py` |
| CUDA OOM | Use smaller model or reduce batch size |
| Data not found | Set `HC01_DATA_DIR=/path/to/mimic` |

---

## Patient Data

### Real MIMIC Data
```python
from app.data_loader import load_patient_data

# Use actual MIMIC patient IDs
# Available: 93810, 24592, 13278, 46048, ... (150 patients)
data = load_patient_data(patient_id=93810, admission_id=193810)
```

### Synthetic Data (When no real data)
```python
data = load_patient_data(patient_id=12345, admission_id=1, use_synthetic=True)
# Returns realistic synthetic vitals + patient demographics
```

### Data Structure
```python
{
    'admission': {
        'patient_id': int,
        'admission_id': int,
        'age': int,
        'gender': str,
        'icu_in': datetime,
        'icu_out': datetime,
        'los_icu': float,
    },
    'vitals': {
        'Heart Rate': (value, 'bpm'),
        'Systolic BP': (value, 'mmHg'),
        'Temperature': (value, '°C'),
        # ... more vitals
    },
    'notes': [
        {'time': str, 'category': str, 'text': str},
        # ... more notes
    ],
    'medications': [
        {'drug': str, 'dose': float, 'unit': str},
        # ... more meds
    ],
}
```

---

## Clinical Reference

### Included Guidelines
- Sepsis diagnosis & treatment (SSC 2021)
- Antimicrobial therapy (SSC 2021)
- Blood cultures (SSC 2021)
- AKI staging (KDIGO 2012)
- Nephrotoxin avoidance (KDIGO 2012)
- SOFA scoring (Sepsis-3)
- ARDS criteria (Berlin 2012)

### Clinical Scores Supported
- **SOFA**: Multi-organ failure assessment
- **NEWS2**: Early warning score for deterioration

---

## Module Reference

### app/config.py
```python
cfg.OLLAMA_BASE        # Local Ollama endpoint
cfg.NIM_BASE           # Remote NIM endpoint
cfg.NOTE_MODEL         # Local note extraction model
cfg.EMBED_MODEL        # Local embedding model
cfg.CHIEF_MODEL        # Remote reasoning model
cfg.FALLBACK_MODEL     # Remote documentation model
cfg.MIMIC_CSVS         # Data file paths
```

### app/data_loader.py
```python
load_patient_data()        # Load complete patient record
MIMICDataLoader().get_patient_admission()
MIMICDataLoader().get_clinical_notes()
MIMICDataLoader().get_lab_values()
MIMICDataLoader().synthesize_vitals()
MIMICDataLoader().get_medications()
```

### app/clients.py
```python
ollama = OllamaClient()
ollama.chat()                # Chat with Ollama model
ollama.embed()               # Get embeddings
ollama.is_online()           # Check if running
ollama.available_models()    # List models

nim = NIMClient()
nim.chat()                   # Chat with NIM model
```

### app/data.py
```python
GUIDELINES              # List of clinical guidelines
MIMIC_IV               # MIMIC-III statistics & reference ranges
```

---

## Notes

- **Privacy**: Local processing (Ollama) handles sensitive data
- **Scalability**: Async architecture supports concurrent requests
- **Extensibility**: Add custom agents in `app/agents.py`
- **Validation**: Always validate AI outputs with medical professionals
- **Compliance**: Ensure HIPAA/GDPR compliance for production

---

## Useful Links

- Build NIM Keys: https://build.nvidia.com/
- MIMIC Documentation: https://mimic.mit.edu/
- Ollama Models: https://ollama.ai/library
- NIM API Docs: https://docs.nvidia.com/ai-enterprise/nim/

---

## Next Steps

1. ✓ Setup complete
2. → Run `python demo.py`
3. → Customize workflows in `app/agents.py`
4. → Integrate with your systems
5. → Monitor & validate outputs
