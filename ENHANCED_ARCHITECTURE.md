# IGNISIA Enhanced Architecture - EHR + STT + TTS Integration

**Date**: April 3, 2026  
**Status**: Design & Planning Phase  
**Objective**: Transform IGNISIA into a complete multimodal clinical AI system

---

## 🎯 System Vision

Transform IGNISIA from a text-only clinical decision support system into a **multimodal, voice-enabled clinical AI platform** with real EHR integration.

### Before (Current)
```
Text Input (Patient Data)
    ↓
OpenAI SDK + NIM Models
    ↓
Text Output (Clinical Analysis)
```

### After (Enhanced)
```
Voice Input (Clinician) ─→ STT (Whisper) ─────┐
                                                ├─→ OpenAI SDK + NIM Models ─→ TTS (Coqui/Kokoro) → Voice Output
Text/EHR Input ──────────→ EHR FHIR API ─────┤
                                                └─→ Text Output

Real-time Data Flow:
1. Voice in from clinician
2. STT converts to text (Whisper)
3. EHR pulls real patient data (FHIR)
4. Chief/Fallback models analyze
5. TTS speaks response back
```

---

## 📋 Enhancement Components

### 1. STT Module (Speech-to-Text)
**Implementation**: OpenAI Whisper (open-source)

```
Features:
✓ Large audio model support (wav, mp3, m4a, etc.)
✓ Automatic language detection
✓ Timestamp-level accuracy
✓ Lightweight local deployment option
✓ Error handling for noisy clinical environments
```

**File**: `app/stt.py`

### 2. TTS Module (Text-to-Speech)
**Options**: Coqui TTS vs Kokoro

**Coqui TTS** (Recommended for variety)
```
✓ Multiple voices and languages
✓ Open source, fully customizable
✓ Realistic prosody
✓ ~300MB model size
✓ Desktop deployment friendly
```

**Kokoro** (Alternative - newer)
```
✓ Advanced speech synthesis (hexgrad/Kokoro-82M)
✓ More natural, conversational
✓ Smaller footprint (~82M params)
✓ Better emotion/sentiment support
✓ Recommended for clinical settings
```

**File**: `app/tts.py`

### 3. EHR Integration Module
**Protocol**: FHIR v4.0 (Fast Healthcare Interoperability Resources)

```
Features:
✓ FHIR-compliant API endpoints
✓ Real-time patient data retrieval
✓ Multiple EHR system support
✓ OAuth2 authentication
✓ HIPAA-compliant data handling
✓ Caching for performance
```

**File**: `app/ehr.py`

**Supported EHR Systems**:
- Epic EHR
- Cerner
- NextGen Healthcare
- OpenMRS (Open source)
- Interoperability via FHIR endpoints

### 4. Audio Processing Pipeline
**File**: `app/audio.py`

```
Features:
✓ Audio format conversion
✓ Noise reduction
✓ Audio validation
✓ Streaming support
✓ Quality metrics
```

### 5. Voice Clinical Workflow Engine
**File**: `app/voice_workflow.py`

```
Features:
✓ Multi-turn conversation management
✓ Context preservation
✓ Fallback to text mode
✓ Error recovery
✓ Session management
```

---

## 🏗️ Enhanced Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    IGNISIA Clinical AI Platform             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  Voice Input     │      │  Text Input      │            │
│  │  (Clinician)     │      │  (Practitioners)│            │
│  └────────┬─────────┘      └────────┬─────────┘            │
│           │                         │                      │
│  ┌────────▼──────────────────────────▼──────────┐          │
│  │  STT Module (Whisper)                        │          │
│  │  - Audio to text conversion                  │          │
│  │  - Language detection                        │          │
│  │  - Timestamp alignment                       │          │
│  └────────┬───────────────────────────────────┬─┘          │
│           │                                   │             │
│  ┌────────▼─────────────────────────────────┐ │            │
│  │  Text Normalization & Preprocessing      │ │            │
│  │  - Clinical entity extraction             │ │            │
│  │  - Abbreviation expansion                 │ │            │
│  └────────┬─────────────────────────────────┘ │            │
│           │                                   │             │
│  ┌────────▼────────────────────────────────────▼──────┐    │
│  │  EHR Integration Layer (FHIR)                     │    │
│  │  - Real-time patient data retrieval              │    │
│  │  - Clinical note pulling                         │    │
│  │  - Lab results, medications, vitals              │    │
│  │  - Admission history, problem list               │    │
│  └────────┬─────────────────────────────────────────┘    │
│           │                                              │
│  ┌────────▼──────────────────────────────────────────┐  │
│  │  ClinicalContext Builder                         │  │
│  │  - Merge voice/text with EHR data               │  │
│  │  - Risk score calculation                        │  │
│  │  - Guideline matching                           │  │
│  └────────┬──────────────────────────────────────┬──┘  │
│           │                                      │      │
│  ┌────────▼─────────────┐    ┌─────────────────▼──┐  │
│  │  Chief Model         │    │  Fallback Model    │  │
│  │  (Nemotron 120B)     │    │  (Qwen 7B)         │  │
│  │  - Advanced reasoning│    │  - Fast docs       │  │
│  │  - Extended thinking │    │  - Quick answers   │  │
│  └────────┬─────────────┘    └─────────────┬──────┘  │
│           │                                │         │
│  ┌────────▼────────────────────────────────▼────┐   │
│  │  Response Formatting & Verification          │   │
│  │  - Clinical safety checks                    │   │
│  │  - Citation & guideline references           │   │
│  │  - Confidence scoring                        │   │
│  └────────┬─────────────────────────────────────┘   │
│           │                                         │
│  ┌────────▼────────────────────────────────────┐    │
│  │  TTS Module (Coqui/Kokoro)                 │    │
│  │  - Text to natural speech                   │    │
│  │  - Voice selection/customization            │    │
│  │  - Real-time audio streaming                │    │
│  └────────┬────────────────────────────────────┘    │
│           │                                         │
│  ┌────────▼──────────────────────────────────┐     │
│  │  Audio Output                             │     │
│  │  - Speaker playback                       │     │
│  │  - Recording for audit trail              │     │
│  │  - Transcript generation                  │     │
│  └────────────────────────────────────────┬──┘     │
│                                           │        │
└─────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow - Clinical Voice Workflow

### Scenario: Clinician asks about admitted patient

```
Step 1: Voice Input
┌─────────────────────────────────┐
│ Clinician: "What's the status   │
│  of patient John Doe? Clinical  │
│  presentation and risk?"        │
└─────────────────────────────────┘
           ↓ (Audio)

Step 2: STT Processing
┌─────────────────────────────────┐
│ Whisper STT                     │
│ Input: raw audio (10 sec)       │
│ Output: "What's the status..."  │
│ Confidence: 0.98                │
└─────────────────────────────────┘
           ↓ (Text)

Step 3: Extract Clinical Entities
┌─────────────────────────────────┐
│ Entity extraction               │
│ - Patient: "John Doe"           │
│ - Query type: "status review"   │
│ - Entities: presentation, risk  │
└─────────────────────────────────┘
           ↓

Step 4: EHR Data Retrieval
┌─────────────────────────────────┐
│ FHIR API Call                   │
│ GET /Patient?name=John+Doe      │
│ GET /Encounter/{id}             │
│ GET /Observation/{vitals}       │
│ GET /MedicationStatement        │
│ GET /Condition                  │
│ Returns: Full clinical picture  │
└─────────────────────────────────┘
           ↓ (Structured)

Step 5: Context Building
┌─────────────────────────────────┐
│ Combine:                        │
│ - Real-time EHR data            │
│ - MIMIC historical patterns     │
│ - Current risk factors          │
│ - Clinical guidelines           │
│ Result: Rich clinical context   │
└─────────────────────────────────┘
           ↓

Step 6: AI Analysis
┌─────────────────────────────────┐
│ Chief Model → Extended Thinking │
│ Reasoning: 15-30 seconds        │
│ Output: Comprehensive analysis  │
│ + Risk stratification           │
│ + Clinical recommendations      │
└─────────────────────────────────┘
           ↓

Step 7: Response Formatting
┌─────────────────────────────────┐
│ Format as clinical narrative:   │
│ - Summary (1-2 sentences)       │
│ - Key findings                  │
│ - Risk assessment               │
│ - Recommended actions           │
│ - Sources/guidelines cited      │
└─────────────────────────────────┘
           ↓

Step 8: TTS Processing
┌─────────────────────────────────┐
│ Kokoro TTS                      │
│ Input: Clinical response text   │
│ Voice: Professional female/male │
│ Duration: ~30 seconds           │
│ Output: Natural speech audio    │
└─────────────────────────────────┘
           ↓

Step 9: Audio Output
┌─────────────────────────────────┐
│ Speaker Output                  │
│ "Patient John Doe admitted...   │
│  Current vitals stable...       │
│  Risk score 6/10...             │
│  Recommend..."                  │
└─────────────────────────────────┘
           ↓

Step 10: Audit & Logging
┌─────────────────────────────────┐
│ - Input audio transcript        │
│ - AI response text & audio      │
│ - Clinician action (if any)     │
│ - Timestamp & user ID           │
│ - HIPAA compliance logging      │
└─────────────────────────────────┘
```

---

## 📦 New Project Structure

```
ignisia/
├── app/
│   ├── __init__.py
│   ├── config.py                 ← UPDATED (STT/TTS configs)
│   ├── nim_client.py             (existing)
│   ├── data_loader.py            (existing)
│   │
│   ├── ★ stt.py                  ← NEW: Whisper STT client
│   ├── ★ tts.py                  ← NEW: Coqui/Kokoro TTS client
│   ├── ★ ehr.py                  ← NEW: FHIR EHR integration
│   ├── ★ audio.py                ← NEW: Audio processing pipeline
│   ├── ★ voice_workflow.py       ← NEW: Voice clinical workflows
│   ├── ★ entities.py             ← NEW: Clinical NER extraction
│   │
│   ├── agents.py                 (existing)
│   ├── clients.py                (existing)
│   └── models.py                 (existing)
│
├── ★ app_voice.py                ← NEW: Voice-enabled main app
├── ★ demo_voice.py               ← NEW: Voice workflow examples
├── ★ test_stt.py                 ← NEW: STT tests
├── ★ test_tts.py                 ← NEW: TTS tests
├── ★ test_ehr.py                 ← NEW: EHR integration tests
├── ★ test_voice_integration.py   ← NEW: End-to-end voice test
│
├── docs/
│   ├── ARCHITECTURE.md           (existing)
│   ├── ★ EHR_INTEGRATION.md       ← NEW: EHR setup guide
│   ├── ★ STT_TTS_GUIDE.md         ← NEW: Voice setup guide
│   ├── ★ VOICE_WORKFLOWS.md       ← NEW: Clinical workflow docs
│   └── ★ FHIR_REFERENCE.md        ← NEW: FHIR API reference
│
├── requirements.txt              ← UPDATED
├── setup_voice.sh                ← NEW: Voice environment setup
└── verify_voice_setup.sh          ← NEW: Voice system verification
```

---

## 🛠️ Dependencies to Add

### STT (Whisper)
```bash
pip install openai-whisper
# or
pip install faster-whisper  # Optimized version
```

### TTS Options
```bash
# Option 1: Coqui TTS
pip install TTS

# Option 2: Kokoro (HexGrad)
pip install git+https://github.com/hexgrad/Kokoro.git
```

### Clinical NER
```bash
pip install transformers torch
pip install spacy
python -m spacy download en_core_sci_md
```

### FHIR & Healthcare
```bash
pip install fhirclient
pip install requests
```

### Audio Processing
```bash
pip install librosa soundfile pydub
pip install scipy numpy
```

---

## 🔐 Configuration Requirements

### EHR FHIR Server
```python
FHIR_SERVER_URL = "https://fhir.hospital.com/api/FHIR/R4"
FHIR_CLIENT_ID = "ignisia_app"
FHIR_CLIENT_SECRET = "***"
FHIR_OAUTH_URL = "https://oauth.hospital.com/token"
FHIR_SCOPE = "user/Patient.read user/Observation.read"
```

### STT Settings
```python
STT_MODEL = "base"  # tiny, base, small, medium, large
STT_DEVICE = "cuda"  # or "cpu"
STT_LANGUAGE = "en"
STT_CHUNK_SIZE = 30  # seconds
```

### TTS Settings
```python
TTS_ENGINE = "kokoro"  # or "coqui"
TTS_VOICE = "female"  # or "male"
TTS_SPEED = 1.0
TTS_DEVICE = "cuda"
```

---

## 🚀 Implementation Phases

### Phase 1: STT Integration (Week 1)
- [ ] Implement Whisper STT module
- [ ] Add audio file/stream handling
- [ ] Create STT tests
- [ ] Document setup

### Phase 2: TTS Integration (Week 1-2)
- [ ] Compare Coqui vs Kokoro performance
- [ ] Implement TTS module
- [ ] Add voice selection options
- [ ] Create TTS tests

### Phase 3: EHR Integration (Week 2-3)
- [ ] Design FHIR client wrapper
- [ ] Implement OAuth2 authentication
- [ ] Add patient data retrieval methods
- [ ] Create EHR tests
- [ ] Document FHIR endpoints

### Phase 4: Clinical Entity Extraction (Week 3)
- [ ] Implement clinical NLP
- [ ] Add entity recognition
- [ ] Create entity tests

### Phase 5: Voice Workflow Engine (Week 4)
- [ ] Implement conversation management
- [ ] Add context preservation
- [ ] Create workflow examples
- [ ] End-to-end testing

### Phase 6: Integration & Deployment (Week 4-5)
- [ ] Full system testing
- [ ] Performance optimization
- [ ] Security audit
- [ ] Production deployment

---

## 📊 Performance Targets

| Component | Today | Target |
|-----------|-------|--------|
| STT Latency | - | <5 sec for 30-sec audio |
| TTS Latency | - | <3 sec for 100-word response |
| EHR Data Fetch | - | <2 sec (with caching) |
| End-to-end (voice→voice) | - | <40 seconds |
| Concurrent Users | 1 | 10+ |

---

## 🔒 Security & Compliance

✓ HIPAA compliance for all EHR data  
✓ OAuth2 authentication with EHR servers  
✓ Local audio processing (no transmission to cloud except NIM)  
✓ Encrypted storage for session data  
✓ Audit logging for all clinical interactions  
✓ PII redaction in logs  
✓ Role-based access control (RBAC)  

---

## 📈 Expected Improvements

### Current System
- Text-only clinical queries
- MIMIC historical data only
- Batch processing
- Clinician must type query

### Enhanced System
- Voice-enabled clinical workflows
- Real-time EHR integration
- Real patient data (live)
- Natural conversation
- Hands-free operation
- Multi-turn context preservation
- Adaptive responses
- Clinical efficiency: ~40-50% faster query-to-answer time

---

## 🎯 Success Metrics

1. **User Adoption**: Clinical staff using voice mode for >30% of queries
2. **Accuracy**: EHR data integration accuracy >99%
3. **Latency**: Voice→voice turnaround <40 seconds
4. **Safety**: Zero HIPAA violations, all interactions audited
5. **Reliability**: System uptime >99.5%
6. **Clinical Value**: Measurable improvement in care coordination

---

## 📞 Next Steps

1. Decide on TTS engine (Coqui vs Kokoro)
2. Review EHR FHIR requirements
3. Setup development environment
4. Begin Phase 1: STT integration
5. Create proof-of-concept demo

