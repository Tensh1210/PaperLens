"""
LLM service for PaperLens.

Provides chat completion via multiple providers (Cerebras, Groq, OpenAI) using
litellm for a unified interface. Includes provider fallback on rate limits
and streaming support.
"""

import json
import os
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any

import structlog
from litellm import acompletion, completion
from litellm.exceptions import RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

logger = structlog.get_logger()


class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded on all providers."""

    pass


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    model: str
    api_key_env: str
    tpm_limit: int  # tokens per minute (for reference/logging)


# Provider chain ordered by preference (higher rate limits first)
PROVIDER_CHAIN = [
    ProviderConfig("cerebras", "cerebras/llama-3.3-70b", "CEREBRAS_API_KEY", 60000),
    ProviderConfig("groq", "groq/llama-3.3-70b-versatile", "GROQ_API_KEY", 12000),
]


class LLMService:
    """
    Service for LLM interactions with multi-provider fallback.

    Uses litellm for unified API access. On rate limit errors, automatically
    falls back to the next provider in the chain.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the LLM service.

        Args:
            model: Model to use. Defaults to config value.
            temperature: Temperature for generation. Defaults to config value.
            api_key: API key for backward compatibility (sets primary provider key).
        """
        self.model = model or settings.llm_full_model
        self.temperature = temperature if temperature is not None else settings.agent_temperature
        self._providers = self._build_provider_chain()
        self._setup_env(api_key)

        logger.info(
            "LLM service initialized",
            model=self.model,
            temperature=self.temperature,
            providers=[p.name for p in self._providers],
        )

    def _build_provider_chain(self) -> list[ProviderConfig]:
        """Build the provider fallback chain based on available API keys."""
        chain = []
        for provider in PROVIDER_CHAIN:
            key = self._get_provider_key(provider.name)
            if key:
                chain.append(provider)
        return chain

    @staticmethod
    def _get_provider_key(provider_name: str) -> str:
        """Get the API key for a provider from settings."""
        key_map = {
            "cerebras": settings.cerebras_api_key,
            "groq": settings.groq_api_key,
            "openai": settings.openai_api_key,
        }
        return key_map.get(provider_name, "")

    def _setup_env(self, api_key: str | None = None) -> None:
        """Set API key env vars for litellm."""
        key_map = {
            "GROQ_API_KEY": settings.groq_api_key,
            "CEREBRAS_API_KEY": settings.cerebras_api_key,
            "OPENAI_API_KEY": settings.openai_api_key,
        }
        # Backward compat: if explicit api_key passed, treat as primary provider key
        if api_key:
            if settings.llm_provider == "groq":
                key_map["GROQ_API_KEY"] = api_key
            elif settings.llm_provider == "cerebras":
                key_map["CEREBRAS_API_KEY"] = api_key
            elif settings.llm_provider == "openai":
                key_map["OPENAI_API_KEY"] = api_key
        for env_var, value in key_map.items():
            if value:
                os.environ[env_var] = value

    def _call_with_fallback(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int,
        stop: list[str] | None,
        **kwargs: Any,
    ) -> str:
        """Call LLM with provider fallback on rate limit errors."""
        temp = temperature if temperature is not None else self.temperature
        errors: list[tuple[str, str]] = []

        # Try the primary model first
        models_to_try = [self.model] + [
            p.model for p in self._providers if p.model != self.model
        ]

        for model in models_to_try:
            try:
                response = completion(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    stop=stop,
                    **kwargs,
                )
                content = response.choices[0].message.content
                logger.debug(
                    "Chat completion generated",
                    model=model,
                    input_messages=len(messages),
                    output_length=len(content) if content else 0,
                )
                return content or ""
            except RateLimitError as e:
                logger.warning("Rate limit hit, trying next provider", model=model)
                errors.append((model, str(e)))
                continue
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    logger.warning("Rate limit hit, trying next provider", model=model)
                    errors.append((model, str(e)))
                    continue
                logger.error("Chat completion failed", model=model, error=str(e))
                raise LLMError(f"Chat completion failed ({model}): {e}") from e

        raise LLMRateLimitError(f"All providers rate limited: {errors}")

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=30),
        reraise=True,
    )
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a chat completion with provider fallback.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Maximum tokens to generate.
            stop: Stop sequences.
            **kwargs: Additional arguments for litellm.

        Returns:
            Generated text response.

        Raises:
            LLMError: If generation fails on all providers.
        """
        return self._call_with_fallback(messages, temperature, max_tokens, stop, **kwargs)

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def achat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a chat completion asynchronously with provider fallback.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Maximum tokens to generate.
            stop: Stop sequences.
            **kwargs: Additional arguments for litellm.

        Returns:
            Generated text response.

        Raises:
            LLMError: If generation fails.
        """
        temp = temperature if temperature is not None else self.temperature
        errors: list[tuple[str, str]] = []

        models_to_try = [self.model] + [
            p.model for p in self._providers if p.model != self.model
        ]

        for model in models_to_try:
            try:
                response = await acompletion(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    stop=stop,
                    **kwargs,
                )
                content = response.choices[0].message.content
                logger.debug(
                    "Async chat completion generated",
                    model=model,
                    input_messages=len(messages),
                    output_length=len(content) if content else 0,
                )
                return content or ""
            except (RateLimitError, Exception) as e:
                if isinstance(e, RateLimitError) or "rate_limit" in str(e).lower():
                    logger.warning("Rate limit hit (async), trying next", model=model)
                    errors.append((model, str(e)))
                    continue
                logger.error("Async chat completion failed", error=str(e))
                raise LLMError(f"Chat completion failed ({model}): {e}") from e

        raise LLMRateLimitError(f"All providers rate limited: {errors}")

    def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Generate a streaming chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Maximum tokens to generate.
            stop: Stop sequences.
            **kwargs: Additional arguments for litellm.

        Yields:
            Text chunks as they are generated.

        Raises:
            LLMError: If generation fails.
        """
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                stop=stop,
                stream=True,
                **kwargs,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error("Streaming chat completion failed", error=str(e))
            if "rate_limit" in str(e).lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
            raise LLMError(f"Streaming chat completion failed: {e}") from e

    async def achat_completion_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Generate an async streaming chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Maximum tokens to generate.
            stop: Stop sequences.
            **kwargs: Additional arguments for litellm.

        Yields:
            Text chunks as they are generated.

        Raises:
            LLMError: If generation fails.
        """
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                stop=stop,
                stream=True,
                **kwargs,
            )
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error("Async streaming chat completion failed", error=str(e))
            if "rate_limit" in str(e).lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
            raise LLMError(f"Streaming chat completion failed: {e}") from e

    def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a completion with tool/function calling.

        Args:
            messages: List of message dicts.
            tools: List of tool definitions in OpenAI format.
            temperature: Override default temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments for litellm.

        Returns:
            Dict with 'content' and optionally 'tool_calls'.

        Raises:
            LLMError: If generation fails.
        """
        try:
            response = completion(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            message = response.choices[0].message
            result: dict[str, Any] = {
                "content": message.content or "",
                "tool_calls": None,
            }

            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                    for tc in message.tool_calls
                ]

            logger.debug(
                "Tool completion generated",
                model=self.model,
                has_tool_calls=result["tool_calls"] is not None,
            )
            return result
        except Exception as e:
            logger.error("Tool completion failed", error=str(e))
            if "rate_limit" in str(e).lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
            raise LLMError(f"Tool completion failed: {e}") from e

    async def agenerate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a completion with tool/function calling asynchronously.

        Args:
            messages: List of message dicts.
            tools: List of tool definitions in OpenAI format.
            temperature: Override default temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments for litellm.

        Returns:
            Dict with 'content' and optionally 'tool_calls'.

        Raises:
            LLMError: If generation fails.
        """
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            message = response.choices[0].message
            result: dict[str, Any] = {
                "content": message.content or "",
                "tool_calls": None,
            }

            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                    for tc in message.tool_calls
                ]

            logger.debug(
                "Async tool completion generated",
                model=self.model,
                has_tool_calls=result["tool_calls"] is not None,
            )
            return result
        except Exception as e:
            logger.error("Async tool completion failed", error=str(e))
            if "rate_limit" in str(e).lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
            raise LLMError(f"Tool completion failed: {e}") from e


# Singleton instance
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


if __name__ == "__main__":
    # Quick test
    service = LLMService()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is BERT in one sentence?"},
    ]

    print(f"Model: {service.model}")
    print(f"Temperature: {service.temperature}")
    print(f"Providers: {[p.name for p in service._providers]}")
    print("\nGenerating response...")

    response = service.chat_completion(messages)
    print(f"\nResponse: {response}")
