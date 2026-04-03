#!/usr/bin/env python3
"""
IGNISIA OpenAI SDK Quick Start
Using NVIDIA NIM with OpenAI-compatible Python client
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# EXAMPLE 1: Simple Documentation Generation (Fallback Model)
# ============================================================================

async def example_fast_documentation():
    """Generate clinical documentation quickly."""
    from app.nim_client import FallbackModelClient
    
    print("[EXAMPLE 1] Fast Clinical Documentation")
    print("-" * 70)
    
    client = FallbackModelClient()
    
    prompt = """
    Patient: 65-year-old female
    Chief Complaint: Shortness of breath
    Vitals: HR 102, BP 142/88, RR 24, SpO2 91%
    
    Write a brief clinical assessment.
    """
    
    response = await client.document(prompt, max_tokens=500)
    print(response)
    print()


# ============================================================================
# EXAMPLE 2: Advanced Clinical Reasoning (Chief Model)
# ============================================================================

async def example_advanced_reasoning():
    """Generate clinical reasoning with extended thinking."""
    from app.nim_client import ChiefModelClient
    
    print("[EXAMPLE 2] Advanced Clinical Reasoning")
    print("-" * 70)
    
    client = ChiefModelClient()
    
    prompt = """
    You are an expert ICU clinician. Analyze:
    
    PATIENT: 68-year-old male
    PRESENTATION: 24-hour fever, chills
    VITALS: HR 110, BP 145/88, Temp 38.5°C, RR 22, SpO2 94%
    LABS: WBC 18.2, Lactate 2.1, Creatinine 1.8 (baseline 1.0)
    
    Provide:
    1. Top 3 differential diagnoses
    2. Key risk factors
    3. Recommended immediate interventions
    """
    
    response = await client.reason(
        prompt=prompt,
        reasoning_budget=6000,
        max_tokens=1500
    )
    print(response)
    print()


# ============================================================================
# EXAMPLE 3: Load Patient Data & Use with NIM
# ============================================================================

async def example_patient_workflow():
    """Load patient data and generate clinical analysis."""
    from app.data_loader import load_patient_data
    from app.nim_client import FallbackModelClient
    
    print("[EXAMPLE 3] Patient Data + Clinical Analysis")
    print("-" * 70)
    
    # Load real MIMIC patient
    patient = load_patient_data(patient_id=93810, admission_id=193810)
    
    print(f"Patient ID: {patient['admission']['patient_id']}")
    print(f"Age: {patient['admission']['age']}, Gender: {patient['admission']['gender']}")
    print(f"LOS: {patient['admission']['los_icu']} days")
    print(f"Vitals: HR={patient['vitals'].get('Heart Rate', ('N/A', ''))[0]}, "
          f"Temp={patient['vitals'].get('Temperature', ('N/A', ''))[0]}°C")
    
    # Generate clinical assessment
    vitals_str = ", ".join([f"{k}={v[0]}{v[1]}" for k, v in patient['vitals'].items()])
    
    prompt = f"""
    Clinical Assessment:
    
    Patient: {patient['admission']['age']}-year-old {patient['admission']['gender']}
    Vitals: {vitals_str}
    LOS: {patient['admission']['los_icu']} days ICU
    
    Provide a brief clinical assessment.
    """
    
    client = FallbackModelClient()
    response = await client.document(prompt, max_tokens=400)
    print("\nClinical Assessment:")
    print(response)
    print()


# ============================================================================
# EXAMPLE 4: Direct NIM Client with Custom Parameters
# ============================================================================

async def example_custom_parameters():
    """Use NIM client with custom parameters."""
    from app.nim_client import NIMMModelClient
    
    print("[EXAMPLE 4] Custom Parameters & Streaming")
    print("-" * 70)
    
    client = NIMMModelClient(api_key=os.getenv("NIM_API_KEY_FALLBACK"))
    
    messages = [
        {"role": "system", "content": "You are an expert clinical pharmacist."},
        {"role": "user", "content": "What are the key concerns with vancomycin in AKI?"},
    ]
    
    response = await client.chat(
        model="qwen/qwen2.5-7b-instruct",
        messages=messages,
        temperature=0.3,
        top_p=0.8,
        max_tokens=600,
        stream=True  # Enable streaming
    )
    
    print(response)
    print()


# ============================================================================
# EXAMPLE 5: Error Handling & Retry Logic
# ============================================================================

async def example_error_handling():
    """Demonstrate error handling."""
    from app.nim_client import FallbackModelClient
    
    print("[EXAMPLE 5] Error Handling")
    print("-" * 70)
    
    client = FallbackModelClient()
    
    try:
        # Normal operation
        response = await client.document(
            "Generate a clinical note.", 
            max_tokens=100
        )
        print("✓ Request succeeded")
        print(f"  Response length: {len(response)} characters")
        
    except Exception as e:
        print(f"✗ Request failed: {e}")
        print("  Fallback: Using cached response or template")
    
    print()


# ============================================================================
# EXAMPLE 6: Batch Processing Multiple Patients
# ============================================================================

async def example_batch_patients():
    """Process multiple patients efficiently."""
    from app.data_loader import load_patient_data
    from app.nim_client import FallbackModelClient
    
    print("[EXAMPLE 6] Batch Processing")
    print("-" * 70)
    
    # Patient IDs from MIMIC dataset
    patient_ids = [93810, 24592, 13278]
    
    client = FallbackModelClient()
    
    for pid in patient_ids:
        try:
            patient = load_patient_data(patient_id=pid, admission_id=pid*10, use_synthetic=False)
            if patient.get('admission'):
                print(f"✓ Patient {pid}: Age {patient['admission']['age']}")
            else:
                print(f"⊘ Patient {pid}: No data in real MIMIC")
        except Exception as e:
            print(f"✗ Patient {pid}: {str(e)[:50]}")
    
    print()


# ============================================================================
# EXAMPLE 7: Multi-step Clinical Workflow
# ============================================================================

async def example_multi_step_workflow():
    """Multi-step clinical workflow."""
    from app.data_loader import load_patient_data
    from app.nim_client import ChiefModelClient, FallbackModelClient
    
    print("[EXAMPLE 7] Multi-Step Workflow")
    print("-" * 70)
    
    # Step 1: Load patient data
    print("Step 1: Loading patient data...")
    patient = load_patient_data(patient_id=93810, admission_id=193810)
    print(f"  ✓ Loaded: Patient {patient['admission']['patient_id']}, Age {patient['admission']['age']}")
    
    # Step 2: Clinical reasoning
    print("\nStep 2: Generating clinical reasoning (Chief Model)...")
    chief = ChiefModelClient()
    reasoning = await chief.reason(
        prompt=f"Patient age {patient['admission']['age']}, vitals abnormal. Concerns?",
        reasoning_budget=4000,
        max_tokens=800
    )
    print(f"  ✓ Generated ({len(reasoning)} chars)")
    
    # Step 3: Documentation
    print("\nStep 3: Generating clinical note (Fallback Model)...")
    fallback = FallbackModelClient()
    note = await fallback.document(
        prompt=f"Write brief note: {reasoning[:100]}",
        max_tokens=400
    )
    print(f"  ✓ Generated note ({len(note)} chars)")
    print()


# ============================================================================
# MAIN: Run Examples
# ============================================================================

async def main():
    """Run all examples."""
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║         IGNISIA OpenAI SDK NIM Integration - Quick Start                  ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝\n")
    
    # Verify setup
    if not os.getenv("NIM_API_KEY_CHIEF") or not os.getenv("NIM_API_KEY_FALLBACK"):
        print("✗ API keys not configured")
        print("  Set: NIM_API_KEY_CHIEF and NIM_API_KEY_FALLBACK")
        return
    
    print("✓ API keys configured\n")
    
    # Run examples
    examples = [
        ("Fast Documentation", example_fast_documentation),
        ("Advanced Reasoning", example_advanced_reasoning),
        ("Patient Workflow", example_patient_workflow),
        ("Custom Parameters", example_custom_parameters),
        ("Error Handling", example_error_handling),
        ("Batch Processing", example_batch_patients),
        ("Multi-Step Workflow", example_multi_step_workflow),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            print(f"\n{'=' * 70}")
            await func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"✗ Example failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 70)
    print("\n✓ Quick start examples completed!")
    print("\nNext steps:")
    print("1. Review NIM_INTEGRATION.md for detailed API documentation")
    print("2. Check app/nim_client.py for implementation details")
    print("3. Customize prompts for your clinical workflows")
    print("4. Run: python demo.py (full clinical pipeline)")


if __name__ == "__main__":
    asyncio.run(main())
