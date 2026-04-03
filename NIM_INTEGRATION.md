# NVIDIA NIM Integration Guide

## Overview

IGNISIA now integrates with NVIDIA NIM (NVIDIA Inference Microservices) APIs for advanced clinical reasoning. The system uses the **OpenAI SDK** (OpenAI-compatible API) for clean, maintainable integration.

## What is NIM?

**NVIDIA NIM** provides enterprise-grade LLM inference with:
- Multiple model options
- OpenAI-compatible API
- Streaming support
- Reasoning capabilities (for Chief model)
- Reliable inference at scale

## Architecture

```
Your Code
    ↓
OpenAI SDK Client
    ↓
NIM API Endpoint (https://integrate.api.nvidia.com/v1)
    ↓
Inference Engine
    ├─ Chief Model (Nemotron 3 120B) ── Extended thinking/reasoning
    └─ Fallback Model (Qwen 2.5 7B) ─── Fast documentation
```

## Setup (Quick Start)

### 1. Get API Keys

Visit: https://build.nvidia.com/

1. Sign up or log in
2. Navigate to API Keys
3. Generate key for each model:
   - Chief Model: Nemotron 3 120B
   - Fallback Model: Qwen 2.5 7B Instruct

### 2. Configure Environment

Create `.env` file in project root:
```bash
NIM_API_KEY_CHIEF=nvapi-your-chief-key
NIM_API_KEY_FALLBACK=nvapi-your-fallback-key
```

Or set as environment variables:
```bash
export NIM_API_KEY_CHIEF="nvapi-xxx"
export NIM_API_KEY_FALLBACK="nvapi-yyy"
```

### 3. Test Connection

```bash
conda activate hc01
python test_nim_quick.py
```

Output should show both models responding successfully.

## Usage

### Basic Usage

```python
import asyncio
from app.nim_client import ChiefModelClient, FallbackModelClient

async def main():
    # Clinical reasoning (with extended thinking)
    chief = ChiefModelClient()
    response = await chief.reason(
        prompt="Patient with fever and elevated lactate. What are the concerns?",
        reasoning_budget=8000,
        max_tokens=2000
    )
    print(response)
    
    # Documentation generation
    fallback = FallbackModelClient()
    response = await fallback.document(
        prompt="Generate a clinical note about...",
        max_tokens=1024
    )
    print(response)

asyncio.run(main())
```

### Advanced: Direct Client

```python
from app.nim_client import NIMMModelClient

client = NIMMModelClient(api_key="nvapi-your-key")

# Custom parameters
response = await client.chat(
    model="qwen/qwen2.5-7b-instruct",
    messages=[
        {"role": "system", "content": "You are a clinical assistant"},
        {"role": "user", "content": "What is sepsis?"}
    ],
    temperature=0.3,
    top_p=0.95,
    max_tokens=1024,
    stream=True
)
```

## API Reference

### ChiefModelClient

**Purpose**: Advanced clinical reasoning with internal thinking

```python
chief = ChiefModelClient(api_key="optional")

# Generate reasoning with extended thinking
response = await chief.reason(
    prompt="Clinical question or scenario",
    reasoning_budget=8000,      # Tokens for internal reasoning (max 16384)
    max_tokens=2000            # Tokens for response
)

# Returns: "[CLINICAL_REASONING]\n{thinking}\n\n[RESPONSE]\n{answer}"
```

### FallbackModelClient

**Purpose**: Fast documentation and general queries

```python
fallback = FallbackModelClient(api_key="optional")

# Generate documentation
response = await fallback.document(
    prompt="Clinical documentation prompt",
    max_tokens=1024
)

# Returns: Documentation text
```

### NIMMModelClient

**Purpose**: Direct access to NIM APIs with full control

```python
client = NIMMModelClient(
    api_key="nvapi-your-key",
    base_url="https://integrate.api.nvidia.com/v1",  # Optional
    timeout=90.0  # Optional
)

# Chat with any model
response = await client.chat(
    model="nvidia/nemotron-3-super-120b-a12b",
    messages=[{"role": "user", "content": "query"}],
    temperature=0.3,
    top_p=0.95,
    max_tokens=1024,
    stream=True,  # Optional
    **kwargs  # Pass extra_body for extended thinking
)

# Generate reasoning (chief model)
response = await client.generate_reasoning(
    prompt="Clinical scenario",
    reasoning_budget=8000,
    max_tokens=2000
)

# Generate documentation (fallback model)
response = await client.generate_documentation(
    prompt="Documentation prompt",
    max_tokens=1024
)
```

## Models Available

### Chief Model (Advanced Reasoning)
- **Name**: `nvidia/nemotron-3-super-120b-a12b`
- **Capabilities**: 
  - Extended thinking (internal reasoning visible)
  - Complex clinical analysis
  - Multi-step reasoning
- **Cost**: Higher (120B parameters)
- **Latency**: ~15-30 seconds
- **Best for**: Complex cases, clinical reasoning, differential diagnosis

### Fallback Model (Fast Documentation)
- **Name**: `qwen/qwen2.5-7b-instruct`
- **Capabilities**:
  - Fast inference
  - Documentation generation
  - General queries
- **Cost**: Lower (7B parameters)
- **Latency**: ~10-20 seconds
- **Best for**: Notes, summaries, routine documentation

## Features

### Extended Thinking (Chief Model)

The Chief model includes "extended thinking" - internal reasoning that's visible:

```python
response = await chief_client.reason(prompt)
# Output format:
# [CLINICAL_REASONING]
# <internal thinking process>
# 
# [RESPONSE]
# <final answer>
```

### Streaming

All models support streaming for reduced latency:

```python
response = await client.chat(
    model="...",
    messages=[...],
    stream=True  # Enables streaming
)
```

### Error Handling

```python
try:
    response = await client.chat(...)
except Exception as e:
    print(f"API Error: {e}")
    # Implement fallback logic
```

## Environment Variables

The system uses python-dotenv for configuration:

```python
# In .env file:
NIM_API_KEY_CHIEF=nvapi-...              # Chief model key
NIM_API_KEY_FALLBACK=nvapi-...           # Fallback model key
OLLAMA_HOST=http://localhost:11434       # Local Ollama (optional)
HC01_DATA_DIR=.                          # MIMIC data directory
DEBUG=false                              # Enable debugging
```

## Testing

### Quick Test (Both Models)
```bash
python test_nim_quick.py
```

### Comprehensive Test Suite
```bash
python test_nim.py
```

Tests:
1. Fallback model - Documentation generation
2. Chief model - Reasoning with extended thinking
3. Direct client - Custom parameters

## Performance

| Model | Task | Latency | Cost |
|-------|------|---------|------|
| Nemotron 120B | Complex reasoning | 15-30s | Higher |
| Qwen 7B | Fast documentation | 10-20s | Lower |

**Tip**: Use Chief model for complex cases, Fallback for routine documentation.

## Limitations

- Rate limiting on NIM API (check NVIDIA documentation for limits)
- Streaming responses may have slight latency overhead
- Extended thinking token budget is shared with response tokens (max 16384 total)

## Troubleshooting

### "API Key not provided" error
```
Solution: Set environment variable or pass api_key to client
export NIM_API_KEY_CHIEF="nvapi-..."
export NIM_API_KEY_FALLBACK="nvapi-..."
```

### "401 Unauthorized"
```
Solution: Check API key is correct and hasn't expired
Visit https://build.nvidia.com/ to verify/regenerate keys
```

### "Connection timeout"
```
Solution: Increase timeout or check network connectivity
client = NIMMModelClient(api_key="...", timeout=180.0)
```

### "Rate limit exceeded"
```
Solution: Implement backoff retry logic or reduce request frequency
Check NIM API limits at https://build.nvidia.com/
```

## Integration with IGNISIA

The NIM client is integrated into the clinical pipeline:

```python
from app.data_loader import load_patient_data
from app.nim_client import ChiefModelClient

# 1. Load patient data
patient = load_patient_data(patient_id=93810, admission_id=193810)

# 2. Generate clinical reasoning
chief = ChiefModelClient()
response = await chief.reason(
    prompt=f"""Analyze this patient:
    Age: {patient['admission']['age']}
    Vitals: {patient['vitals']}
    Notes: {patient['notes']}
    """
)

# 3. Use in your workflow
print(response)
```

## Best Practices

1. **Use environment variables** for API keys (never hardcode)
2. **Implement retry logic** for production use
3. **Use Fallback model** for routine tasks (cost savings)
4. **Use Chief model** only for complex reasoning (worth the cost)
5. **Stream responses** for better UX
6. **Handle errors gracefully** with fallback options
7. **Monitor API usage** for cost control

## Next Steps

1. Set up `.env` file with your API keys
2. Run `python test_nim_quick.py` to verify
3. Integrate into your clinical workflows
4. Review `app/nim_client.py` for API details
5. Customize prompts for your use cases

## Resources

- **NIM Build Platform**: https://build.nvidia.com/
- **NIM Documentation**: https://docs.nvidia.com/ai-enterprise/nim/
- **OpenAI SDK**: https://github.com/openai/openai-python
- **IGNISIA Demo**: `python demo.py`

## Support

- Check test outputs: `python test_nim_quick.py`
- Review code in `app/nim_client.py`
- Examine usage in `demo.py`
- Verify API keys at https://build.nvidia.com/

---

**System Status**: ✓ NIM APIs configured and tested
**Ready for**: Clinical decision support workflows
