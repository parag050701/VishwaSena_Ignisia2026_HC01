#!/usr/bin/env python3
"""
Demo script testing the IGNISIA pipeline with sample patient data.
Demonstrates end-to-end clinical decision support workflow.
"""

import asyncio
import json
import sys
import time
from typing import Dict, Optional

from app.config import cfg
from app.clients import ollama, nim, OllamaClient, NIMClient
from app.data_loader import load_patient_data
from app.data import GUIDELINES


async def retrieve_relevant_guidelines(query: str, top_k: int = 3) -> list:
    """
    Retrieve relevant clinical guidelines based on query using keyword matching.
    In production, this would use semantic similarity with embeddings.
    """
    query_lower = query.lower()
    scored = []

    for guideline in GUIDELINES:
        score = 0
        keywords = guideline.get("keywords", [])

        # Keyword matching
        for keyword in keywords:
            if keyword.lower() in query_lower:
                score += 1

        # Title/section matching
        if guideline.get("id", "").lower() in query_lower:
            score += 2

        if score > 0:
            scored.append((score, guideline))

    # Sort by score and return top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [g for _, g in scored[:top_k]]


async def demo_patient_analysis(patient_id: int = 1, admission_id: int = 1):
    """
    Demonstrate end-to-end patient analysis pipeline:
    1. Load patient data (vitals, notes, labs)
    2. Extract information from notes using local Ollama model
    3. Retrieve relevant clinical guidelines
    4. Generate clinical reasoning with Chief Model
    5. Generate clinical note with Fallback Model
    """
    print("=" * 80)
    print(f"IGNISIA Clinical Decision Support - Demo (Patient {patient_id})")
    print("=" * 80)

    # Step 1: Load patient data
    print("\n[1/5] Loading patient data...")
    patient_data = load_patient_data(patient_id, admission_id, use_synthetic=True)
    admission = patient_data["admission"]
    vitals = patient_data["vitals"]
    notes = patient_data["notes"]

    print(f"  ✓ Patient ID: {admission.get('patient_id', 'N/A')}")
    print(f"  ✓ Age: {admission.get('age', 'N/A')}, Gender: {admission.get('gender', 'N/A')}")
    print(
        f"  ✓ Vitals: HR={vitals.get('Heart Rate', ('N/A', ''))[0]}, "
        f"SBP={vitals.get('Systolic BP', ('N/A', ''))[0]}, "
        f"Temp={vitals.get('Temperature', ('N/A', ''))[0]}"
    )
    if notes:
        print(f"  ✓ Clinical notes available: {len(notes)} note(s)")

    # Step 2: Extract information from clinical notes
    print("\n[2/5] Extracting information from clinical notes...")
    note_text = notes[0]["text"] if notes else "Patient admitted to ICU."
    extracted_info = None

    # Try Ollama first (local processing)
    if await ollama.is_online():
        try:
            print(f"  Using local model: {cfg.NOTE_MODEL}")
            extraction_prompt = (
                "Extract the chief complaint, severity level, and key findings from this note.\n"
                "Respond with JSON: {chief_complaint, severity, key_findings}\n\n" + note_text
            )
            messages = [
                {"role": "system", "content": "You are a clinical NLP agent. Return ONLY valid JSON."},
                {"role": "user", "content": extraction_prompt},
            ]
            raw = await ollama.chat(cfg.NOTE_MODEL, messages, max_tokens=300)

            # Try to parse JSON
            import re

            m = re.search(r"\{[^}]*\}", raw)
            if m:
                extracted_info = json.loads(m.group())
                print(f"  ✓ Extracted: {extracted_info.get('chief_complaint', 'N/A')}")
            else:
                extracted_info = {"raw": raw[:200]}
        except Exception as e:
            print(f"  ⚠ Ollama extraction failed: {e}")

    # Step 3: Retrieve relevant clinical guidelines
    print("\n[3/5] Retrieving relevant clinical guidelines...")
    query = (
        extracted_info.get("chief_complaint", "")
        if extracted_info
        else admission.get("chief_complaint", "ICU monitoring")
    )
    guidelines = await retrieve_relevant_guidelines(query, top_k=3)
    print(f"  ✓ Retrieved {len(guidelines)} relevant guidelines")
    for i, g in enumerate(guidelines, 1):
        print(f"    {i}. {g['id']}: {g['source']}")
        print(f"       {g['text'][:100]}...")

    # Step 4: Generate clinical reasoning with Chief Model
    print("\n[4/5] Generating clinical reasoning (Chief Model via NIM)...")
    vitals_str = ", ".join([f"{k}={v[0]}{v[1]}" for k, v in vitals.items()])
    guidelines_str = "\n".join([g["text"] for g in guidelines[:2]]) if guidelines else "No specific guidelines"

    reasoning_prompt = f"""You are an expert ICU clinician. Analyze this patient:

PATIENT DATA:
- Age: {admission.get('age', 'N/A')}, Gender: {admission.get('gender', 'N/A')}
- Chief Complaint: {query or 'Not specified'}
- Vital Signs: {vitals_str}
- Notes Summary: {note_text[:200]}

RELEVANT GUIDELINES:
{guidelines_str}

Provide 2-3 sentences of clinical assessment and initial care plan considering the guidelines above."""

    try:
        messages = [{"role": "user", "content": reasoning_prompt}]
        reasoning = await nim.chat(
            cfg.CHIEF_MODEL, messages, cfg.NIM_API_KEY_CHIEF, max_tokens=400
        )
        print(f"  ✓ Clinical Reasoning Generated")
        print(f"    {reasoning[:150]}...")
    except Exception as e:
        print(f"  ⚠ Chief model failed: {e}")
        reasoning = "Unable to generate reasoning. Ensure NIM_API_KEY_CHIEF is set correctly."

    # Step 5: Generate clinical note
    print("\n[5/5] Generating clinical note (Fallback Model via NIM)...")
    note_prompt = f"""Generate a concise clinical note for this patient:

Chief Complaint: {query or 'Not specified'}
Vitals: {vitals_str}
Assessment: {reasoning[:150]}

Write 2-3 sentences in standard medical notation."""

    try:
        messages = [{"role": "user", "content": note_prompt}]
        note = await nim.chat(
            cfg.FALLBACK_MODEL, messages, cfg.NIM_API_KEY_FALLBACK, max_tokens=300
        )
        print(f"  ✓ Clinical Note Generated")
        print(f"    {note[:150]}...")
    except Exception as e:
        print(f"  ⚠ Note generation failed: {e}")
        note = "Unable to generate note."

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Patient: ID={admission.get('patient_id', 'N/A')}, Age={admission.get('age', 'N/A')}")
    print(f"Chief Complaint: {query or 'Not specified'}")
    print(f"\n📋 CLINICAL REASONING:\n{reasoning}")
    print(f"\n📝 CLINICAL NOTE:\n{note}")
    print("=" * 80)


async def test_connections():
    """Test connectivity to local Ollama and remote NIM services."""
    print("[DIAGNOSTIC] Testing service connectivity...\n")

    # Test Ollama
    print("Testing Ollama...")
    ollama_ok = await ollama.is_online()
    if ollama_ok:
        models = await ollama.available_models()
        print(f"✓ Ollama online. Available models: {', '.join(models[:3])}")
    else:
        print("✗ Ollama offline (http://localhost:11434)")

    # Test NIM
    print("\nTesting NIM API...")
    try:
        test_msg = [{"role": "user", "content": "What is clinical decision support?"}]
        response = await nim.chat(
            cfg.FALLBACK_MODEL, test_msg, cfg.NIM_API_KEY_FALLBACK, max_tokens=100
        )
        print(f"✓ NIM accessible. Response: {response[:100]}...")
    except Exception as e:
        print(f"✗ NIM failed: {e}")
        print("  Ensure NIM_API_KEY_CHIEF and NIM_API_KEY_FALLBACK are set")

    print()


async def main():
    """Main entry point."""
    print("IGNISIA Clinical Decision Support System - Demo\n")

    # Test services
    await test_connections()

    # Run demo with real patient data
    print("=" * 80)
    print("Starting patient analysis demo...")
    print("=" * 80)

    try:
        # Use a real patient from MIMIC database
        await demo_patient_analysis(patient_id=93810, admission_id=193810)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
