# IGNISIA Setup & Configuration Guide

## Quick Start

### 1. Install Dependencies
All dependencies are already installed. Verify with:
```bash
cd /home/admin-/Desktop/ignisia
pip install -r requirements.txt
```

### 2. Start Ollama Server
Open a new terminal:
```bash
ollama serve
```

In another terminal, ensure models are available:
```bash
ollama pull qwen2.5:7b-q4_k_m  # Clinical note extraction (~4.4 GB)
ollama pull bge-m3:latest       # RAG embeddings (~7 GB)
```

### 3. Configure NIM API Keys
The system uses NVIDIA NIM APIs for powerful clinical reasoning. Set your keys:

**Option A: Environment Variables** (Recommended)
```bash
export NIM_API_KEY_CHIEF="nvapi-your-chief-model-key"
export NIM_API_KEY_FALLBACK="nvapi-your-fallback-model-key"
python demo.py
```

**Option B: Edit config directly**
Edit `app/config.py` and replace the default keys with your own:
```python
NIM_API_KEY_CHIEF = "nvapi-your-key-here"
NIM_API_KEY_FALLBACK = "nvapi-your-key-here"
```

### 4. Run the Demo
```bash
python demo.py
```

The demo will:
1. ✓ Check Ollama connectivity (local)
2. ✓ Check NIM API connectivity (remote)
3. ✓ Load real patient data from MIMIC-III CSVs
4. ✓ Extract information from clinical notes
5. ✓ Retrieve relevant clinical guidelines via RAG
6. ✓ Generate clinical reasoning with Chief Model
7. ✓ Generate clinical notes with Fallback Model

## Architecture Overview

```
User Input
    ↓
Data Loader → Patient admission data, vitals, notes from MIMIC CSVs
    ↓
Ollama (Local) ← Note extraction, guideline embeddings
    ↓
RAG System ← Clinical guidelines retrieval
    ↓
Inference Engine
    ├─ Chief Model (NIM) ← Clinical reasoning
    └─ Fallback Model (NIM) ← Documentation generation
    ↓
Clinical Output → Assessment, care plan, clinical notes
```

## Project Structure

```
ignisia/
├── app/
│   ├── __init__.py
│   ├── agents.py           # Clinical agent workflows
│   ├── clients.py          # Ollama & NIM API clients
│   ├── config.py           # Configuration & settings
│   ├── data.py             # Clinical reference data (guidelines)
│   ├── data_loader.py      # MIMIC CSV loader
│   ├── main.py             # Application entry
│   ├── models.py           # Data models
│   ├── scoring.py          # Clinical scoring (SOFA, NEWS2)
├── docs/                   # Documentation
├── MIMIC Data Files        # Patient data
│   ├── NOTEEVENTS.csv
│   ├── LABEVENTS.csv
│   ├── ICUSTAYS.csv
│   ├── PATIENTS.csv
│   ├── PRESCRIPTIONS.csv
│   └── D_LABITEMS.csv
├── demo.py                 # Demonstration script
├── requirements.txt        # Python dependencies
└── README.md              # Main documentation
```

## Data Files

The system includes sample MIMIC-III data:
- **NOTEEVENTS.csv**: Clinical notes
- **LABEVENTS.csv**: Laboratory values
- **ICUSTAYS.csv**: ICU stay information
- **PATIENTS.csv**: Patient demographics
- **PRESCRIPTIONS.csv**: Medication data
- **D_LABITEMS.csv**: Lab item definitions

### Using Different Data
To use your own MIMIC data, set the environment variable:
```bash
export HC01_DATA_DIR="/path/to/your/mimic/csvs"
python demo.py
```

## API Keys & Security

### Getting NIM API Keys
1. Visit https://build.nvidia.com/
2. Create an account or sign in
3. Navigate to "API Keys"
4. Create new API keys for:
   - Chief Model (Nemotron 3 120B)
   - Fallback Model (Qwen 2.5 7B Instruct)

### Best Practices
- **Never commit API keys** to version control
- **Use environment variables** for local development
- **Rotate keys regularly** for production
- **Monitor API usage** to detect unusual activity

## Configuration Details

### Local Models (Ollama)
- **Base URL**: `http://localhost:11434`
- **Note Model**: `qwen2.5:7b-q4_k_m` (4.4 GB)
- **Embedding Model**: `bge-m3` (~7 GB)
- **Timeout**: 120 seconds

### Remote Models (NIM)
- **Base URL**: `https://integrate.api.nvidia.com/v1`
- **Chief Model**: `nvidia/nemotron-3-super-120b-a12b` (Advanced reasoning)
- **Fallback Model**: `qwen/qwen2.5-7b-instruct` (Documentation)
- **Timeout**: 90 seconds

### Clinical Configuration
- **OUTLIER_Z**: 2.5 (Z-score threshold for vital sign anomalies)
- **TOP_K_GUIDELINES**: 5 (Guidelines to retrieve)

## Usage Examples

### Basic Patient Analysis
```python
import asyncio
from app.data_loader import load_patient_data

# Load patient data
data = load_patient_data(patient_id=93810, admission_id=193810)
print(data)
```

### Using the Inference Engine
```python
import asyncio
from app.clients import nim
from app.config import cfg

async def analyze():
    messages = [{"role": "user", "content": "Patient with SOB and fever..."}]
    result = await nim.chat(
        model=cfg.CHIEF_MODEL,
        messages=messages,
        api_key=cfg.NIM_API_KEY_CHIEF,
        max_tokens=500
    )
    print(result)

asyncio.run(analyze())
```

### RAG-Based Guideline Retrieval
```python
from app.demo import retrieve_relevant_guidelines
import asyncio

async def get_guidelines():
    guidelines = await retrieve_relevant_guidelines("sepsis with lactate elevation")
    for g in guidelines:
        print(f"- {g['id']}: {g['text'][:100]}...")

asyncio.run(get_guidelines())
```

## Troubleshooting

### Ollama Connection Failed
**Error**: `Connection refused to localhost:11434`
**Solution**: 
1. Start Ollama: `ollama serve`
2. Check it's running: `curl http://localhost:11434/api/tags`

### Model Not Found
**Error**: `ollama: model not found` or similar
**Solution**:
```bash
ollama pull qwen2.5:7b-q4_k_m
ollama pull bge-m3:latest
```

### NIM Authorization Failed
**Error**: `401 Unauthorized` or `403 Forbidden`
**Solution**:
1. Check API keys are set correctly
2. Verify keys haven't expired
3. Test with a simple request:
```bash
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
  -H "Authorization: Bearer $NIM_API_KEY_FALLBACK" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen/qwen2.5-7b-instruct","messages":[{"role":"user","content":"test"}]}'
```

### Timeout Errors
**Error**: `Request timeout after 90 seconds`
**Solution**: Increase timeouts in `config.py`
```python
NIM_TIMEOUT = 180.0  # Increase from 90
OLLAMA_TIMEOUT = 240.0  # Increase from 120
```

### Out of Memory
**Error**: `CUDA out of memory` or similar
**Solution**: Use smaller models or enable quantization
```bash
ollama pull qwen2.5:3b-q3_k_m  # Smaller variant
```

### Data Files Not Found
**Error**: `Warning: noteevents not found at ./NOTEEVENTS.csv`
**Solution**: Set the data directory
```bash
export HC01_DATA_DIR="/path/to/mimic/csvs"
python demo.py
```

## Performance Optimization

### Caching
The system caches loaded CSVs in memory. For large datasets:
```python
from app.data_loader import MIMICDataLoader
loader = MIMICDataLoader()
# First call loads from disk
data1 = loader.get_clinical_notes(patient_id=93810, admission_id=193810)
# Second call uses cache
data2 = loader.get_lab_values(patient_id=93810, admission_id=193810)
```

### Batch Processing
Process multiple patients efficiently:
```python
import asyncio
from app.demo import demo_patient_analysis

async def batch_analyze():
    patients = [(93810, 193810), (24592, 124592), (13278, 113278)]
    tasks = [demo_patient_analysis(pid, aid) for pid, aid in patients]
    await asyncio.gather(*tasks)

asyncio.run(batch_analyze())
```

### Model Selection
- Use Ollama (local) for fast, frequent operations
- Use NIM Chief Model only for complex clinical reasoning
- Use NIM Fallback Model for general documentation

## Development

### Adding Custom Agents
Edit `app/agents.py` to add new clinical workflows:
```python
async def agent_custom_analysis(ctx: AgentContext):
    """Your custom agent."""
    await ctx.log("CUSTOM", "Running custom analysis...")
    # Your implementation
```

### Extending Data Models
Add new data models in `app/models.py`:
```python
class CustomMetric(BaseModel):
    value: float
    unit: str
    timestamp: str
```

### Adding Clinical Guidelines
Update `app/data.py`:
```python
GUIDELINES.append({
    "id": "custom-guideline",
    "source": "Your Source",
    "keywords": ["keyword1", "keyword2"],
    "text": "Guideline text..."
})
```

## Advanced Configuration

### Custom Ollama Models
Replace default models in `config.py`:
```python
NOTE_MODEL = "llama2:13b"  # Your custom model
EMBED_MODEL = "nomic-embed-text"  # Your embedding model
```

### Custom NIM Models
Check available models:
```bash
curl "https://integrate.api.nvidia.com/v1/models" \
  -H "Authorization: Bearer $NIM_API_KEY_FALLBACK"
```

### Proxy Configuration
If behind a corporate proxy:
```python
# In config.py
import httpx
PROXY = "http://proxy.company.com:8080"

# In clients.py modifications
async with httpx.AsyncClient(proxy=PROXY, ...) as client:
    ...
```

## Next Steps

1. **Test the demo**: `python demo.py`
2. **Review clinical guidelines**: Check `app/data.py` for reference protocols
3. **Customize for your use case**: Modify agents and workflows in `app/agents.py`
4. **Integrate with your systems**: Use the API interfaces for external integration
5. **Monitor and evaluate**: Track accuracy and clinical appropriateness

## Resources

- [MIMIC-III Documentation](https://mimic.mit.edu/)
- [Ollama Models](https://ollama.ai/library)
- [NVIDIA NIM Documentation](https://docs.nvidia.com/ai-enterprise/nim/)
- [Clinical News2 Scoring](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6641821/)
- [SOFA Score](https://www.ncbi.nlm.nih.gov/pubmed/8331348)

## Support

For issues or questions:
1. Review the Troubleshooting section
2. Check the demo script for working examples
3. Examine logs in the console output
4. Verify all services are running and accessible

## License & Disclaimer

This system is provided for research and educational purposes. When using for clinical decision support:

- **Always validate** AI-generated recommendations with qualified medical professionals
- **Never rely solely** on AI recommendations for clinical decisions
- **Consider patient context** beyond what the system can process
- **Document & audit** all clinical decisions
- **Comply with regulations** (HIPAA, GDPR, local regulations)
