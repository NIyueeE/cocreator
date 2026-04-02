"""OpenAI-compatible VLM provider wrapper.

Async wrapper for OpenAI-compatible API endpoints with retry + concurrency control.
"""

import asyncio
import base64
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI

from ..schemas import RateLimitConfig, VLMConfig


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0
):
    """Decorator for retrying async functions with exponential backoff."""
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
            timeout=httpx.Timeout(vlm_config.timeout),
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
                    "messages": messages,
                    **kwargs
                }
                if response_format is not None:
                    create_kwargs["response_format"] = response_format
                response = await self._client.chat.completions.create(**create_kwargs)
                return response.choices[0].message.content or ""
            except Exception:
                raise

    async def chat_with_images(
        self,
        image_paths: list[str],
        messages: list[dict[str, Any]],
        response_format: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> str:
        image_contents = []
        for image_path in image_paths:
            def read_image(path: str) -> bytes:
                with open(path, "rb") as f:
                    return f.read()

            image_bytes = await asyncio.to_thread(read_image, image_path)
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })

        if messages and isinstance(messages[-1].get("content"), str):
            text_content = messages[-1]["content"]
            content_array = [*image_contents, {"type": "text", "text": text_content}]
            messages_with_images = messages[:-1] + [
                {"role": "user", "content": content_array}
            ]
        else:
            messages_with_images = messages + [
                {"role": "user", "content": image_contents}
            ]

        return await self.chat(messages_with_images, response_format, **kwargs)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()


async def create_provider(config: VLMConfig, rate_limit: RateLimitConfig) -> OpenAICompatibleProvider:
    """Factory function to create a provider with retry logic."""
    provider = OpenAICompatibleProvider(config, rate_limit)

    # Test connection
    try:
        await provider.chat([{"role": "user", "content": "test"}])
    except Exception as e:
        await provider.close()
        raise ValueError(f"Failed to connect to VLM at {config.base_url}: {e}")

    return provider


__all__ = ["OpenAICompatibleProvider", "create_provider", "retry_with_backoff"]
