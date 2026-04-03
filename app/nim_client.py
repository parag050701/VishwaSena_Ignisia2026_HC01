"""
OpenAI SDK-based NIM client for NVIDIA inference models.
Uses OpenAI's API format with NIM endpoints.
"""

from typing import Optional, List, Dict, AsyncIterator
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class NIMMModelClient:
    """Client for NVIDIA NIM models using OpenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: float = 90.0,
    ):
        """
        Initialize NIM client.
        
        Args:
            api_key: NIM API key (defaults to env variable)
            base_url: NIM API endpoint
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("NIM_API_KEY_FALLBACK")
        self.base_url = base_url
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "NIM API key not provided. "
                "Set NIM_API_KEY_FALLBACK environment variable or pass api_key parameter."
            )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            model: Model name (e.g., "qwen/qwen2.5-7b-instruct")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens in response
            stream: Whether to stream response
            **kwargs: Additional parameters (e.g., reasoning_budget for Chief model)

        Returns:
            Full response text
        """
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

        if stream:
            # Collect streamed chunks
            result = ""
            reasoning = ""
            async for chunk in response:
                if not chunk.choices:
                    continue

                # Extract reasoning (for Chief model)
                chunk_reasoning = getattr(
                    chunk.choices[0].delta, "reasoning_content", None
                )
                if chunk_reasoning:
                    reasoning += chunk_reasoning

                # Extract content
                if chunk.choices[0].delta.content is not None:
                    result += chunk.choices[0].delta.content

            # Return reasoning + content for chief model (has thinking enabled)
            if reasoning:
                return f"[REASONING]\n{reasoning}\n\n[RESPONSE]\n{result}"
            return result
        else:
            # Non-streamed response
            return response.choices[0].message.content or ""

    async def generate_reasoning(
        self,
        prompt: str,
        reasoning_budget: int = 8000,
        max_tokens: int = 2000,
    ) -> str:
        """
        Generate clinical reasoning using Chief model with extended thinking.

        Args:
            prompt: Clinical question/scenario
            reasoning_budget: Tokens for internal reasoning (max 16384)
            max_tokens: Tokens for final response

        Returns:
            Response with reasoning and final answer
        """
        messages = [{"role": "user", "content": prompt}]

        # Use Chief model with reasoning enabled
        response = await self.client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=messages,
            temperature=0.3,
            top_p=0.95,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": min(reasoning_budget, 16384),
            },
            stream=True,
        )

        result = ""
        reasoning = ""
        async for chunk in response:
            if not chunk.choices:
                continue

            chunk_reasoning = getattr(
                chunk.choices[0].delta, "reasoning_content", None
            )
            if chunk_reasoning:
                reasoning += chunk_reasoning

            if chunk.choices[0].delta.content is not None:
                result += chunk.choices[0].delta.content

        return f"[CLINICAL_REASONING]\n{reasoning}\n\n[RESPONSE]\n{result}"

    async def generate_documentation(
        self,
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate clinical documentation using Fallback model.

        Args:
            prompt: Documentation prompt
            max_tokens: Maximum response tokens

        Returns:
            Generated documentation
        """
        messages = [{"role": "user", "content": prompt}]

        response = await self.client.chat.completions.create(
            model="qwen/qwen2.5-7b-instruct",
            messages=messages,
            temperature=0.2,
            top_p=0.7,
            max_tokens=max_tokens,
            stream=True,
        )

        result = ""
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                result += chunk.choices[0].delta.content

        return result


class ChiefModelClient:
    """Convenience wrapper for Chief model (Nemotron 120B)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NIM_API_KEY_CHIEF")
        if not self.api_key:
            raise ValueError(
                "NIM_API_KEY_CHIEF not set. "
                "Set environment variable or pass api_key parameter."
            )
        self.client = NIMMModelClient(api_key=self.api_key)

    async def reason(
        self,
        prompt: str,
        reasoning_budget: int = 8000,
        max_tokens: int = 2000,
    ) -> str:
        """Generate clinical reasoning with extended thinking."""
        return await self.client.generate_reasoning(
            prompt=prompt,
            reasoning_budget=reasoning_budget,
            max_tokens=max_tokens,
        )


class FallbackModelClient:
    """Convenience wrapper for Fallback model (Qwen 7B)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NIM_API_KEY_FALLBACK")
        if not self.api_key:
            raise ValueError(
                "NIM_API_KEY_FALLBACK not set. "
                "Set environment variable or pass api_key parameter."
            )
        self.client = NIMMModelClient(api_key=self.api_key)

    async def document(
        self,
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Generate clinical documentation."""
        return await self.client.generate_documentation(
            prompt=prompt,
            max_tokens=max_tokens,
        )


# Convenience singleton instances
async def get_chief_client() -> ChiefModelClient:
    """Get Chief model client."""
    return ChiefModelClient()


async def get_fallback_client() -> FallbackModelClient:
    """Get Fallback model client."""
    return FallbackModelClient()
