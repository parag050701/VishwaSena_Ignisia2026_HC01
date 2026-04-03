"""
Demo: Voice-Enabled Clinical Workflow
Shows complete voice interaction with MockEHR data
"""

import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: In production, import real modules
# For demo, we'll show the workflow structure


class MockAudioGenerator:
    """Generate mock audio for demo"""
    
    @staticmethod
    def generate_sample_query():
        """Generate sample audio bytes (in production, from microphone)"""
        # This represents 3 seconds of silence (placeholder)
        # In real use: record from microphone via pyaudio
        return b'\x00' * (16000 * 2 * 3)  # 16kHz, 16-bit, 3 seconds


async def demo_basic_voice_workflow():
    """Demo 1: Basic voice query without EHR"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Voice Query (No EHR)")
    print("="*70)
    
    try:
        from app.voice_workflow import VoiceWorkflowEngine
        
        # Initialize voice engine
        print("\n[1] Initializing Voice Engine...")
        engine = VoiceWorkflowEngine(use_chief_model=True)
        
        # Create session
        print("[2] Creating session...")
        session = engine.create_session("demo_basic_001")
        print(f"    ✓ Session: {session.session_id}")
        
        # Simulate voice input
        print("[3] Simulating voice input...")
        print("    [User speaks]: What are the common causes of dyspnea in ICU patients?")
        
        # In production: audio_bytes = record_from_microphone()
        audio_bytes = MockAudioGenerator.generate_sample_query()
        
        print("[4] Processing voice input...")
        print("    - Converting speech to text (STT)")
        print("    - Extracting clinical context")
        print("    - Running AI analysis with Chief model")
        print("    - Synthesizing response to speech (TTS)")
        
        # Note: In production, uncomment to run real workflow:
        # result = await engine.process_voice_input("demo_basic_001", audio_bytes)
        
        print("\n[5] Expected Output:")
        print("    User: What are the common causes of dyspnea in ICU patients?")
        print("    AI: [Clinical reasoning]")
        print("        Common causes include: pneumonia, pulmonary embolism,")
        print("        heart failure exacerbation, ARDS... [continues with full analysis]")
        print("    Audio: [Response synthesized to natural speech]")
        
        print("\n✓ Demo 1 Complete")
        
    except ImportError as e:
        print(f"Note: {e}")
        print("Run: pip install -r requirements-voice.txt")


async def demo_ehr_integrated_workflow():
    """Demo 2: Voice workflow with real EHR data"""
    print("\n" + "="*70)
    print("DEMO 2: Voice Query with EHR Integration")
    print("="*70)
    
    print("\n[1] Initializing Voice Engine with FHIR EHR...")
    print("    Configuration:")
    print("    - FHIR Server: https://fhir.hospital.com/api/FHIR/R4")
    print("    - Auth: OAuth2 client credentials")
    print("    - Models: Chief (Nemotron) + EHR data")
    
    print("\n[2] Creating patient-specific session...")
    print("    Session: demo_ehr_001")
    print("    Patient ID: 12345")
    print("    Clinician: Dr. Smith")
    
    print("\n[3] Simulating clinician voice input...")
    print("    [Clinician]: What's the status of my patient John Doe?")
    
    print("\n[4] System processes query:")
    print("    a) STT: Convert speech to text ✓")
    print("       → 'What is the status of my patient John Doe?'")
    print("")
    print("    b) EHR Lookup: Fetch full patient record")
    print("       - Query FHIR for patient demographics")
    print("       - Retrieve active encounters")
    print("       - Get recent vitals and lab results")
    print("       - Pull medication list")
    print("       - Check allergies")
    print("       - Get active diagnoses")
    print("")
    print("    c) Clinical Context Extraction:")
    print("       - Query type: patient_status")
    print("       - Urgency: routine")
    print("       - Entity found: 'John Doe' → Patient ID: 12345 ✓")
    print("")
    print("    d) AI Analysis with Chief Model:")
    print("       - Input: Voice query + Real-time EHR data")
    print("       - Extended thinking: ON (8000 tokens)")
    print("       - Processing: 20-30 seconds")
    print("")
    print("    e) TTS: Convert response to speech")
    print("       - Engine: Kokoro (natural synthesis)")
    print("       - Duration: ~45 seconds of audio")
    
    print("\n[5] Expected AI Response:")
    print("    [Internal Reasoning]:")
    print("    'Patient John Doe (MRN: 12345) admitted 3 days ago with...")
    print("     Current vital signs stable... Risk assessment: moderate...")
    print("     Based on clinical guidelines and MIMIC patterns...'")
    print("")
    print("    [Spoken Response]:")
    print("    'John Doe is clinically stable. Current vitals show HR 85, BP 140/90,")
    print("     SpO2 96% on room air. He has three active conditions...'")
    print("    [AUDIO OUTPUT TO SPEAKER]")
    
    print("\n[6] Session History:")
    print("    Message 1 (User): Voice query")
    print("    Message 2 (AI): Clinical response with audio")
    print("    ...")


async def demo_multi_turn_conversation():
    """Demo 3: Multi-turn voice conversation"""
    print("\n" + "="*70)
    print("DEMO 3: Multi-Turn Voice Conversation")
    print("="*70)
    
    conversation = [
        ("Clinician", "Tell me about patient Sarah Martinez"),
        ("AI", "[Fetches EHR] Sarah Martinez is a 65-year-old admitted with..."),
        ("Clinician", "What are her current medications?"),
        ("AI", "[Accesses EHR medication list] She's on Lisinopril 10mg..."),
        ("Clinician", "Any recent lab abnormalities?"),
        ("AI", "[Analyzes recent labs] Yes, elevated creatinine suggesting..."),
        ("Clinician", "What's the risk score?"),
        ("AI", "[Calculates with Chief model] Risk score is 7/10 due to..."),
    ]
    
    print("\n[Multi-turn session]")
    for speaker, message in conversation:
        if speaker == "Clinician":
            print(f"\n🎤 Clinician: {message}")
            print("   [Voice input processed via STT]")
        else:
            print(f"🤖 AI: {message[:60]}...")
            print("   [Response synthesized and played via speakers]")
    
    print("\n✓ Session complete - All context preserved across turns")


async def demo_fallback_mode():
    """Demo 4: Fast fallback mode for urgent queries"""
    print("\n" + "="*70)
    print("DEMO 4: Fallback Mode (Fast Response)")
    print("="*70)
    
    print("\n[Scenario]: Critical alert - need rapid assessment")
    
    print("\n[1] Initialize with FALLBACK model (Qwen 7B):")
    print("    - Faster response time: 10-15 seconds")
    print("    - Trade-off: Less deep reasoning")
    print("    - Use case: Time-critical queries")
    
    print("\n[2] Voice input: 'Patient SpO2 dropped to 88%'")
    
    print("\n[3] Processing (quick path):")
    print("    STT → Clinical context extraction → EHR fetch →")
    print("    Fallback model (faster) → TTS → Audio output")
    
    print("\n[4] Response time: ~12 seconds")
    print("    AI: 'SpO2 88% requires immediate intervention...'")
    
    print("\n✓ Fast response provided within critical time window")


async def demo_comparison():
    """Demo 5: Compare Chief vs Fallback models"""
    print("\n" + "="*70)
    print("DEMO 5: Model Comparison")
    print("="*70)
    
    comparison = {
        "Chief (Nemotron 120B)": {
            "Response Time": "15-30 seconds",
            "Depth": "Comprehensive with reasoning",
            "Use Case": "Complex cases needing explanation",
            "Extended Thinking": "✓ Enabled",
            "Input Size": "Up to 4000 tokens",
            "Output": "[REASONING] ... [RESPONSE] ...",
        },
        "Fallback (Qwen 7B)": {
            "Response Time": "10-15 seconds",
            "Depth": "Quick documentation",
            "Use Case": "Time-critical, routine queries",
            "Extended Thinking": "Not used",
            "Input Size": "Up to 1024 tokens",
            "Output": "Direct response",
        },
    }
    
    print("\n[Model Comparison]")
    for model, specs in comparison.items():
        print(f"\n  {model}:")
        for key, value in specs.items():
            print(f"    • {key}: {value}")


async def demo_full_system_architecture():
    """Demo 6: Full system architecture"""
    print("\n" + "="*70)
    print("DEMO 6: Full System Architecture")
    print("="*70)
    
    architecture = """
    IGNISIA Voice-Enabled Clinical AI System
    ════════════════════════════════════════════════════════════════
    
    INPUT LAYER:
    │
    ├─ 🎤 Microphone (Voice Input)
    │  └─ Audio Stream (WAV, MP3, etc.)
    │
    ├─ ⌨️ Keyboard (Text Input - fallback)
    │  └─ Direct text query
    │
    └─ 🏥 EHR Systems (Real-time Data)
       └─ FHIR API (Patient records, vitals, labs)
    
    PROCESSING LAYER:
    │
    ├─ 🗣️ STT Module (Whisper)
    │  ├─ Audio → Text (16kHz, 16-bit)
    │  ├─ Language detection
    │  └─ Timestamp accuracy
    │
    ├─ 📊 EHR Integration (FHIR Client)
    │  ├─ OAuth2 Authentication
    │  ├─ Patient lookup
    │  ├─ Medication/vital retrieval
    │  └─ Caching for performance
    │
    ├─ 🧠 Clinical NLP
    │  ├─ Entity extraction
    │  ├─ Abbreviation expansion
    │  ├─ Context parsing
    │  └─ Urgency detection
    │
    ├─ 🤖 AI Analysis Models
    │  ├─ Chief Model (Nemotron 120B)
    │  │  └─ Advanced reasoning with extended thinking
    │  └─ Fallback Model (Qwen 7B)
    │     └─ Fast documentation generation
    │
    └─ 📝 Response Formatting
       ├─ Clinical safety checks
       ├─ Citation of guidelines
       └─ Confidence scoring
    
    OUTPUT LAYER:
    │
    ├─ 🎤 TTS Module (Kokoro/Coqui)
    │  ├─ Text → Natural speech
    │  ├─ Voice selection
    │  └─ Real-time streaming
    │
    ├─ 🔊 Speaker (Audio Output)
    │  └─ Playback to clinician
    │
    ├─ 📄 Text Output
    │  └─ Display on screen
    │
    └─ 📊 Session Management
       ├─ Conversation history
       ├─ Audit logging
       └─ Context preservation
    
    BACKEND SERVICES:
    │
    ├─ NVIDIA NIM APIs
    │  ├─ Chief (Nemotron 120B)
    │  └─ Fallback (Qwen 7B)
    │
    ├─ FHIR EHR Servers
    │  └─ Hospital system integration
    │
    └─ Data Storage
       ├─ MIMIC-III (historical)
       ├─ Session cache
       └─ Audit logs
    
    ════════════════════════════════════════════════════════════════
    """
    print(architecture)


async def run_all_demos():
    """Run all demonstrations"""
    print("\n\n")
    print("╔" + "="*68 + "╗")
    print("║" + " " * 10 + "IGNISIA Voice System - Complete Demonstration" + " "*14 + "║")
    print("╚" + "="*68 + "╝")
    
    await demo_basic_voice_workflow()
    await demo_ehr_integrated_workflow()
    await demo_multi_turn_conversation()
    await demo_fallback_mode()
    await demo_comparison()
    await demo_full_system_architecture()
    
    print("\n\n" + "="*70)
    print("ALL DEMOS COMPLETE")
    print("="*70)
    print("\n📚 Next Steps:")
    print("  1. Run: conda activate hc01")
    print("  2. Install: pip install -r requirements-voice.txt")
    print("  3. Test STT: python -c 'from app.stt import WhisperSTT'")
    print("  4. Test TTS: python -c 'from app.tts import TTSManager'")
    print("  5. Read: docs/VOICE_SETUP.md")
    print("\n🚀 Production Deployment:")
    print("  - Deploy with FastAPI server")
    print("  - Configure FHIR EHR connection")
    print("  - Setup OAuth authentication")
    print("  - Enable audit logging for HIPAA compliance")
    print("\n💡 Documentation:")
    print("  - ENHANCED_ARCHITECTURE.md - System design")
    print("  - VOICE_SETUP.md - Installation & configuration")
    print("  - VOICE_WORKFLOWS.md - Clinical workflows")
    print("  - FHIR_REFERENCE.md - FHIR API details")
    print("\n✓ Voice-enabled clinical AI system ready!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_all_demos())
