"""
LLM service for PaperLens.

Provides chat completion via Groq using litellm for unified interface.
Includes streaming support and retry logic.
"""

import json
from collections.abc import AsyncGenerator, Generator
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
    """Rate limit exceeded."""

    pass


class LLMService:
    """
    Service for LLM interactions via Groq.

    Uses litellm for unified API access and includes retry logic
    for handling transient failures.
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
            api_key: API key. Defaults to config value.
        """
        self.model = model or settings.llm_full_model
        self.temperature = temperature if temperature is not None else settings.agent_temperature
        self._api_key = api_key or settings.groq_api_key

        # Set environment variable for litellm
        if self._api_key:
            import os
            os.environ["GROQ_API_KEY"] = self._api_key

        logger.info(
            "LLM service initialized",
            model=self.model,
            temperature=self.temperature,
        )

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, RateLimitError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60),
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
        Generate a chat completion.

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
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs,
            )
            content = response.choices[0].message.content
            logger.debug(
                "Chat completion generated",
                model=self.model,
                input_messages=len(messages),
                output_length=len(content) if content else 0,
            )
            return content or ""
        except RateLimitError:
            logger.warning("Rate limit hit, will retry with backoff...")
            raise  # Let tenacity handle retry
        except Exception as e:
            logger.error("Chat completion failed", error=str(e))
            raise LLMError(f"Chat completion failed: {e}") from e

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
        Generate a chat completion asynchronously.

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
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs,
            )
            content = response.choices[0].message.content
            logger.debug(
                "Async chat completion generated",
                model=self.model,
                input_messages=len(messages),
                output_length=len(content) if content else 0,
            )
            return content or ""
        except Exception as e:
            logger.error("Async chat completion failed", error=str(e))
            if "rate_limit" in str(e).lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
            raise LLMError(f"Chat completion failed: {e}") from e

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
    print("\nGenerating response...")

    response = service.chat_completion(messages)
    print(f"\nResponse: {response}")
