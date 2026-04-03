#!/usr/bin/env python3
"""Quick integration test for NVIDIA NIM APIs."""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from app.nim_client import NIMMModelClient, FallbackModelClient


async def quick_test():
    """Quick test of both models."""
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║         NVIDIA NIM Quick Integration Test                                 ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝\n")

    # Check API keys
    chief_key = os.getenv("NIM_API_KEY_CHIEF", "")
    fallback_key = os.getenv("NIM_API_KEY_FALLBACK", "")

    print(f"✓ Chief key:    {chief_key[:20]}..." if chief_key else "✗ Chief key not set")
    print(f"✓ Fallback key: {fallback_key[:20]}..." if fallback_key else "✗ Fallback key not set")
    print()

    if not chief_key or not fallback_key:
        print("✗ Missing API keys")
        return False

    try:
        # Test 1: Fallback model (fast)
        print("[1/2] Testing Fallback Model (Qwen 7B)...")
        fallback = FallbackModelClient()
        response = await fallback.document(
            "Write a 1-sentence clinical note about a patient with fever.",
            max_tokens=100,
        )
        print(f"✓ Response: {response[:80]}...\n")

        # Test 2: Chief model (reasoning)
        print("[2/2] Testing Chief Model (Nemotron 120B with reasoning)...")
        chief = NIMMModelClient(api_key=chief_key)
        response = await chief.generate_reasoning(
            "What is the primary concern for a patient with WBC 18 and lactate 2.1?",
            reasoning_budget=4000,
            max_tokens=800,
        )
        print(f"✓ Response received ({len(response)} chars)")
        print(f"✓ Preview: {response[:150]}...\n")

        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                    ✓ ALL TESTS PASSED                                     ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print("\nNIM APIs are working correctly!")
        print("Ready to use in: app/nim_client.py")
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(quick_test())
    exit(0 if success else 1)
