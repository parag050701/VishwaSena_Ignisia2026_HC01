# Voice System Setup Guide

## Overview
This guide covers complete setup of IGNISIA's voice capabilities including:
- **STT** (Speech-to-Text): Whisper
- **TTS** (Text-to-Speech): Coqui or Kokoro  
- **EHR Integration**: FHIR API support
- **Clinical Workflows**: Voice-driven clinical analysis

---

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows with WSL
- **Python**: 3.11+ (conda hc01 already has this)
- **RAM**: 16GB+ (for speech models)
- **GPU**: Optional but recommended (CUDA 11.8+)
- **Audio Device**: Microphone for input, speakers for output

### Audio System Check
```bash
# Check audio devices on Linux
arecord -l        # Recording devices
aplay -l          # Playback devices

# Check audio devices on macOS
system_profiler SPAudioDataType

# Check on Windows
Get-AudioDevice -List  # PowerShell
```

---

## Installation Steps

### Step 1: Activate Environment
```bash
conda activate hc01
```

### Step 2: Install Voice Dependencies
```bash
# Navigate to project directory
cd /home/admin-/Desktop/ignisia

# Install all voice packages
pip install -r requirements-voice.txt

# Install spaCy medical NLP model (optional, for clinical NER)
python -m spacy download en_core_sci_md
```

### Step 3: Download Models (First Time)

#### Whisper STT
```bash
# Whisper will auto-download on first use
# Available sizes: tiny (39M), base (140M), small (440M), medium (1.5G), large (2.9G)
export STT_MODEL=base      # Recommended balance
export STT_DEVICE=cuda     # or "cpu" if no GPU
```

#### Kokoro TTS (Recommended)
```bash
# Install Kokoro
pip install git+https://github.com/hexgrad/Kokoro.git

# Model auto-downloads on first use (~500MB)
export TTS_ENGINE=kokoro
export TTS_DEVICE=cuda
```

#### Coqui TTS (Alternative)
```bash
# Installation happens in app/tts.py on first use
# Model size: ~300MB
export TTS_ENGINE=coqui
```

---

## Configuration

### Step 1: STT Configuration (.env)
```bash
# Speech-to-Text Settings
STT_MODEL=base                    # tiny, base, small, medium, large
STT_DEVICE=cuda                   # cuda or cpu
STT_LANGUAGE=en                   # Language code
STT_CHUNK_SIZE=30                 # Max audio chunk in seconds
```

### Step 2: TTS Configuration (.env)
```bash
# Text-to-Speech Settings
TTS_ENGINE=kokoro                 # kokoro or coqui
TTS_VOICE=female                  # female or male
TTS_SPEED=1.0                     # 0.5-2.0
TTS_DEVICE=cuda                   # cuda or cpu
TTS_LANGUAGE=en
```

### Step 3: EHR Configuration (.env - Optional)
```bash
# FHIR Server Configuration (if using real EHR)
FHIR_SERVER_URL=https://fhir.hospital.com/api/FHIR/R4
FHIR_CLIENT_ID=your_client_id
FHIR_CLIENT_SECRET=your_client_secret
FHIR_OAUTH_URL=https://oauth.hospital.com/token
FHIR_SCOPE="user/Patient.read user/Observation.read user/Encounter.read"
```

### Step 4: Complete .env File Example
```bash
# Existing NIM settings (already configured)
NIM_API_KEY_CHIEF=nvapi-...
NIM_API_KEY_FALLBACK=nvapi-...

# New Voice Settings
# STT (Whisper)
STT_MODEL=base
STT_DEVICE=cuda
STT_LANGUAGE=en
STT_CHUNK_SIZE=30

# TTS (Kokoro/Coqui)
TTS_ENGINE=kokoro
TTS_VOICE=female
TTS_SPEED=1.0
TTS_DEVICE=cuda

# EHR (Optional - for production use)
FHIR_SERVER_URL=
FHIR_CLIENT_ID=
FHIR_CLIENT_SECRET=
FHIR_OAUTH_URL=

# Other settings
HC01_DATA_DIR=.
DEBUG=false
```

---

## Verification

### Quick Test: STT Module
```bash
cd /home/admin-/Desktop/ignisia

python -c "
import asyncio
from app.stt import WhisperSTT

async def test():
    stt = WhisperSTT(model_name='base')
    print('✓ STT module loaded successfully')
    print(stt.get_model_info())

asyncio.run(test())
"
```

### Quick Test: TTS Module
```bash
python -c "
import asyncio
from app.tts import TTSManager

async def test():
    manager = TTSManager()
    print('✓ TTS manager initialized')
    print(manager.get_status())
    
    # Test synthesis
    audio = await manager.synthesize('Patient stable')
    print(f'✓ Synthesis complete: {len(audio)} bytes')

asyncio.run(test())
"
```

### Quick Test: Voice Workflow
```bash
python -c "
import asyncio
from app.voice_workflow import VoiceWorkflowEngine

async def test():
    engine = VoiceWorkflowEngine(use_chief_model=True)
    session = engine.create_session('test_session')
    print(f'✓ Voice engine ready: {session.session_id}')

asyncio.run(test())
"
```

### Full System Verification
```bash
bash verify_voice_setup.sh
```

---

## Usage Examples

### Example 1: Basic Voice Query
```python
import asyncio
from app.voice_workflow import VoiceWorkflowEngine

async def basic_voice_query():
    # Initialize engine
    engine = VoiceWorkflowEngine(use_chief_model=True)
    
    # Create session
    session = engine.create_session("clinical_session_001")
    
    # Load sample audio (use real mic in production)
    with open("sample_audio.wav", "rb") as f:
        audio_bytes = f.read()
    
    # Process voice input → get voice output
    result = await engine.process_voice_input("clinical_session_001", audio_bytes)
    
    if result["success"]:
        print(f"User: {result['user_input']}")
        print(f"AI: {result['ai_response_text']}")
        
        # Get response audio
        response_audio = result['response_audio_bytes']
        with open("response.wav", "wb") as f:
            f.write(response_audio)
    else:
        print(f"Error: {result['error']}")

asyncio.run(basic_voice_query())
```

### Example 2: With EHR Integration
```python
async def voice_query_with_ehr():
    # Initialize with FHIR EHR
    engine = VoiceWorkflowEngine(
        fhir_server_url="https://fhir.hospital.com/api/FHIR/R4",
        fhir_client_id="app_client_id",
        fhir_client_secret="app_secret",
        fhir_oauth_url="https://oauth.hospital.com/token",
        use_chief_model=True,  # Advanced reasoning
    )
    
    # Create session with patient context
    session = engine.create_session(
        session_id="patient_case_001",
        patient_id="12345",  # FHIR Patient ID
        clinician_id="dr_smith"
    )
    
    # Process voice input (engine automatically fetches EHR data)
    result = await engine.process_voice_input(
        "patient_case_001",
        audio_bytes
    )
    
    # Result includes both EHR data and AI analysis
    print(f"Patient context: {result}")
```

### Example 3: Session Management
```python
# Create multiple sessions for different patients
engine = VoiceWorkflowEngine()

# Patient A - Fast response
session_a = engine.create_session("fast_consult", patient_id="A123")

# Patient B - Deep analysis
session_b = engine.create_session("complex_case", patient_id="B456")

# List all active sessions
sessions = engine.list_sessions()
for s in sessions:
    print(f"Session {s['session_id']}: {s['messages']} messages")
```

### Example 4: Real-Time Microphone Input
```python
import pyaudio
import numpy as np

async def voice_chat_from_mic():
    engine = VoiceWorkflowEngine(use_chief_model=True)
    session = engine.create_session("live_session")
    
    # Record audio from microphone
    def record_audio(duration=10):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
        )
        
        frames = []
        for _ in range(0, int(16000 / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return b''.join(frames)
    
    # Record user query
    print("Recording... (10 seconds)")
    audio = record_audio(10)
    
    # Process
    result = await engine.process_voice_input("live_session", audio)
    
    # Play response
    print("AI Response:", result['ai_response_text'])
```

---

## Troubleshooting

### Issue: STT not initialized
```
Solution: pip install openai-whisper faster-whisper
```

### Issue: TTS engine unavailable
```
Solution: 
  - For Kokoro: pip install git+https://github.com/hexgrad/Kokoro.git
  - For Coqui: pip install TTS
```

### Issue: CUDA not detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, use CPU instead
export STT_DEVICE=cpu
export TTS_DEVICE=cpu
```

### Issue: Audio device not found
```bash
# List available audio devices
python -m sounddevice --list

# Configure default device in code if needed
```

### Issue: FHIR connection refused
```
Solution:
1. Verify FHIR_SERVER_URL is correct
2. Check OAuth credentials (FHIR_CLIENT_ID/SECRET)
3. Ensure network access to FHIR server
4. Test OAuth token endpoint: curl FHIR_OAUTH_URL
```

---

## Performance Tuning

### For Fast Response (< 20 seconds)
```bash
# Use fallback model (Qwen 7B)
export TTS_ENGINE=kokoro
export STT_MODEL=tiny      # Faster but less accurate
export TTS_SPEED=1.2       # Slightly faster speech
```

### For High Accuracy (> 40 seconds)
```bash
# Use chief model (Nemotron 120B)
export STT_MODEL=small     # Better accuracy
export TTS_SPEED=1.0       # Natural speech
# Extended thinking enabled by default
```

### GPU Optimization
```bash
# Use mixed precision
export TORCH_DTYPE=float16
export CUDA_VISIBLE_DEVICES=0
```

---

## Security Considerations

✓ **API Keys**: Never commit .env to git  
✓ **HIPAA Compliance**: All patient data stays local (except NIM API calls)  
✓ **Audio Logging**: Enable only for debugging, disable in production  
✓ **OAuth**: Use secure secret management for FHIR credentials  
✓ **Audit Trail**: All voice interactions should be logged with timestamps

---

## Next Steps

1. **Run Demo**: `python demo_voice.py`
2. **Test Integration**: `python test_voice_integration.py`
3. **Explore Examples**: See `VOICE_WORKFLOWS.md`
4. **Deploy**: Review `DEPLOYMENT.md`

---

## Support

For issues or questions:
1. Check `ENHANCED_ARCHITECTURE.md` for system design
2. Review example files in repository
3. Check logs for error details
4. Consult FHIR documentation at https://www.hl7.org/fhir/

---

**Status**: Voice system ready for clinical deployment! 🚀
