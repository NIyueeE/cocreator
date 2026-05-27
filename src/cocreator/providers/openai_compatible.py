"""OpenAI-compatible VLM provider wrapper.

Async wrapper for OpenAI-compatible API endpoints with retry + concurrency control.
"""

import asyncio
import base64
import io
from pathlib import Path
from typing import Any, Optional, Union

import httpx
from openai import AsyncOpenAI
from PIL import Image

from ..schemas import RateLimitConfig, VLMConfig


def read_raw(path: Union[str, Path]) -> bytes:
    """Open image, resize to 960×540, JPEG quality 85 — matches API preprocessing."""
    with open(path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        img_resized = img.resize((960, 540), Image.BICUBIC)
        buf = io.BytesIO()
        img_resized.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def retry_with_backoff(
    max_attempts: int = 3, backoff_factor: float = 2.0, initial_delay: float = 1.0
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
        base_url = vlm_config.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(vlm_config.timeout),
            transport=httpx.AsyncHTTPTransport(
                # disable connection pooling to rule out pool-related hangs
                limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
            ),
        )
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=vlm_config.api_key,
            http_client=http_client,
        )

    @retry_with_backoff(max_attempts=3, backoff_factor=2.0, initial_delay=1.0)
    async def chat(
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[dict[str, Any]] = None,
        **kwargs: Any,
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
        create_kwargs: dict[str, Any] = {
            "model": self.vlm_config.model,
            "messages": messages,
            **kwargs,
        }
        if self.vlm_config.enable_thinking:
            create_kwargs["extra_body"] = {"enable_thinking": True}
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        response = await self._client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content or ""

    async def chat_with_images(
        self,
        image_paths: list[str],
        messages: list[dict[str, Any]],
        response_format: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        async def _load(path: str) -> dict:
            image_bytes = await asyncio.to_thread(read_raw, path)
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}

        image_contents = await asyncio.gather(*[_load(p) for p in image_paths])

        if messages and isinstance(messages[-1].get("content"), str) and messages[-1]["content"].strip():
            text_content = messages[-1]["content"]
            content_array = [*image_contents, {"type": "text", "text": text_content}]
            messages_with_images = messages[:-1] + [
                {"role": "user", "content": content_array}
            ]
        else:
            messages_with_images = messages + [
                {"role": "user", "content": image_contents}
            ]

        result = await self.chat(messages_with_images, response_format, **kwargs)
        return result

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()
