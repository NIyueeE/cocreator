"""OpenAI-compatible VLM provider wrapper.

Async wrapper for OpenAI-compatible API endpoints with retry + concurrency control.
"""

import asyncio
import base64
from typing import Any, Optional

from openai import AsyncOpenAI

from ..schemas import RateLimitConfig, VLMConfig


class OpenAICompatibleProvider:
    """
    Async VLM provider for OpenAI-compatible APIs.

    Only requires base_url and api_key - supports any OpenAI-compatible backend
    including SiliconFlow, Ollama, Azure OpenAI, etc.
    """

    def __init__(self, vlm_config: VLMConfig, rate_limit_config: RateLimitConfig):
        self.vlm_config = vlm_config
        self.rate_limit_config = rate_limit_config
        self._client = AsyncOpenAI(
            base_url=vlm_config.base_url,
            api_key=vlm_config.api_key,
        )
        self._semaphore = asyncio.Semaphore(rate_limit_config.concurrency)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> str:
        """
        Send a chat request to the VLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Optional {"type": "json_object"} for JSON responses
            **kwargs: Additional arguments to pass to chat completions API

        Returns:
            The VLM's response text
        """
        async with self._semaphore:
            try:
                create_kwargs: dict[str, Any] = {
                    "model": self.vlm_config.model,
                    "messages": messages,  # type: ignore[arg-type]
                    **kwargs
                }
                if response_format is not None:
                    create_kwargs["response_format"] = response_format  # type: ignore[assignment]
                response = await self._client.chat.completions.create(**create_kwargs)  # type: ignore[call-overload]
                return response.choices[0].message.content or ""
            except Exception:
                # Retry logic handled by caller
                raise

    async def chat_with_images(
        self,
        image_paths: list[str],
        messages: list[dict[str, Any]],
        response_format: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Send a chat request with images to the VLM.

        Args:
            image_paths: List of paths to image files
            messages: List of message dicts
            response_format: Optional {"type": "json_object"} for JSON responses
            **kwargs: Additional arguments to pass to chat completions API

        Returns:
            The VLM's response text
        """
        # Build image content blocks
        image_contents = []
        for image_path in image_paths:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            })

        # Create user message with images
        user_message = {
            "role": "user",
            "content": image_contents
        }

        messages_with_images = messages + [user_message]

        return await self.chat(messages_with_images, response_format, **kwargs)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()


async def create_provider(config: VLMConfig, rate_limit: RateLimitConfig) -> OpenAICompatibleProvider:
    """
    Factory function to create a provider with retry logic.
    """
    provider = OpenAICompatibleProvider(config, rate_limit)

    # Test connection
    try:
        await provider.chat([{"role": "user", "content": "test"}])
    except Exception as e:
        await provider.close()
        raise ValueError(f"Failed to connect to VLM at {config.base_url}: {e}")

    return provider


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0
):
    """
    Decorator for retrying async functions with exponential backoff.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                        delay *= backoff_factor

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected state in retry_with_backoff")

        return wrapper
    return decorator


__all__ = ["OpenAICompatibleProvider", "create_provider", "retry_with_backoff"]