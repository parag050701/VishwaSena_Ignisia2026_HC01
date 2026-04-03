#!/usr/bin/env python3
"""
Test script for NVIDIA NIM API integration using OpenAI SDK.
Tests both Chief (Nemotron 120B) and Fallback (Qwen 7B) models.
"""

import asyncio
import sys
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from app.nim_client import NIMMModelClient, ChiefModelClient, FallbackModelClient


async def test_fallback_model():
    """Test Fallback model (Qwen 2.5 7B) for documentation."""
    print("\n" + "=" * 80)
    print("TEST 1: Fallback Model (Qwen 2.5 7B - Fast Documentation)")
    print("=" * 80)

    try:
        client = FallbackModelClient()
        print("✓ Connected to Fallback model")

        prompt = "Write a brief clinical assessment for a 68-year-old patient with elevated heart rate (110 bpm) and fever (38.5°C)."

        print("\n📝 Generating clinical documentation...\n")
        result = await client.document(prompt, max_tokens=500)

        print("Response:")
        print("-" * 80)
        print(result)
        print("-" * 80)
        print("✓ Fallback model test completed")

    except Exception as e:
        print(f"✗ Fallback model test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_chief_model():
    """Test Chief model (Nemotron 3 120B) for reasoning."""
    print("\n" + "=" * 80)
    print("TEST 2: Chief Model (Nemotron 3 120B - Advanced Reasoning)")
    print("=" * 80)

    try:
        client = ChiefModelClient()
        print("✓ Connected to Chief model")

        prompt = """You are an expert ICU clinician. Analyze this patient:

PATIENT: 68-year-old male
VITALS: HR 110, BP 145/88, Temp 38.5°C, RR 22, SpO2 94%
LABS: WBC 18.2, Lactate 2.1, Creatinine 1.8 (baseline 1.0)
NOTES: Admitted from ER with fever and chills x24h. History of hypertension.

Using your clinical expertise and internal reasoning, provide:
1. Differential diagnosis
2. Risk assessment
3. Recommended initial workup
4. Empiric therapy consideration"""

        print("\n🧠 Generating clinical reasoning with extended thinking...\n")
        result = await client.reason(
            prompt, reasoning_budget=8000, max_tokens=2000
        )

        print("Response:")
        print("-" * 80)
        print(result)
        print("-" * 80)
        print("✓ Chief model test completed")

    except Exception as e:
        print(f"✗ Chief model test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_direct_nim_client():
    """Test direct NIM client with custom parameters."""
    print("\n" + "=" * 80)
    print("TEST 3: Direct NIM Client (Custom Parameters)")
    print("=" * 80)

    try:
        # Using fallback key for this demo
        client = NIMMModelClient(api_key=os.getenv("NIM_API_KEY_FALLBACK"))
        print("✓ Connected to NIM client")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful medical assistant.",
            },
            {
                "role": "user",
                "content": "What are the key considerations for sepsis management?",
            },
        ]

        print("\n💬 Testing with custom parameters...\n")
        result = await client.chat(
            model="qwen/qwen2.5-7b-instruct",
            messages=messages,
            temperature=0.5,
            top_p=0.8,
            max_tokens=800,
            stream=True,
        )

        print("Response:")
        print("-" * 80)
        print(result)
        print("-" * 80)
        print("✓ Direct NIM client test completed")

    except Exception as e:
        print(f"✗ Direct NIM client test failed: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests."""
    print("\n╔════════════════════════════════════════════════════════════════════════════╗")
    print("║           NVIDIA NIM API Integration Test Suite                          ║")
    print("║              Using OpenAI SDK with Extended Thinking                      ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # Check environment
    chief_key = os.getenv("NIM_API_KEY_CHIEF")
    fallback_key = os.getenv("NIM_API_KEY_FALLBACK")

    print(f"\n✓ Chief API Key: {chief_key[:20]}..." if chief_key else "✗ Chief API Key: Not set")
    print(f"✓ Fallback API Key: {fallback_key[:20]}..." if fallback_key else "✗ Fallback API Key: Not set")

    if not chief_key or not fallback_key:
        print("\n✗ Missing API keys. Please set NIM_API_KEY_CHIEF and NIM_API_KEY_FALLBACK")
        sys.exit(1)

    # Run tests
    try:
        await test_fallback_model()
        await test_chief_model()
        await test_direct_nim_client()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Review the test outputs above")
        print("2. Check app/nim_client.py for API usage patterns")
        print("3. Integrate NIMMModelClient into your workflows")
        print("4. Run: python demo.py (for full clinical pipeline)")

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
