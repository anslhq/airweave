"""OpenAI-compatible LLM implementation.

Uses the official OpenAI Python SDK with configurable base_url, making it
work with any OpenAI-compatible endpoint (local LLMs, vLLM, Ollama, etc.).

Key differences from other providers:
- Uses standard OpenAI response_format for structured JSON output
- No provider-specific reasoning/thinking parameters
- base_url configured explicitly via OPENAI_COMPAT_BASE_URL
"""

import json
import time
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from airweave.adapters.llm.base import BaseLLM
from airweave.adapters.llm.exceptions import LLMTransientError
from airweave.adapters.llm.registry import LLMModelSpec
from airweave.adapters.llm.tool_response import LLMResponse, LLMToolCall
from airweave.core.config import settings

T = TypeVar("T", bound=BaseModel)


class OpenAICompatLLM(BaseLLM):
    """OpenAI-compatible LLM provider.

    Works with any endpoint that speaks the OpenAI chat completions API:
    local LLMs, vLLM, Ollama, LM Studio, etc.
    """

    def __init__(
        self,
        model_spec: LLMModelSpec,
        max_retries: int | None = None,
    ) -> None:
        """Initialize the OpenAI-compatible client with explicit proxy settings."""
        super().__init__(model_spec, max_retries=max_retries)

        api_key = settings.OPENAI_COMPAT_API_KEY
        if not api_key:
            raise ValueError(
                "OPENAI_COMPAT_API_KEY not configured. Set it in your environment or .env file."
            )

        base_url = settings.OPENAI_COMPAT_BASE_URL
        if not base_url:
            raise ValueError(
                "OPENAI_COMPAT_BASE_URL is required for OpenAI-compatible provider. "
                "Set it to your endpoint URL (e.g. http://localhost:8317/v1)."
            )

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.DEFAULT_TIMEOUT,
        )

        self._logger.debug(
            f"[OpenAICompatLLM] Initialized with model={model_spec.api_model_name}, "
            f"base_url={base_url}, context_window={model_spec.context_window}"
        )

    def _prepare_schema(self, schema_json: dict[str, Any]) -> dict[str, Any]:
        return self._normalize_strict_schema(schema_json)

    async def _call_api(
        self,
        prompt: str,
        schema: type[T],
        schema_json: dict[str, Any],
        system_prompt: str,
        thinking: bool = False,
    ) -> T:
        api_start = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self._model_spec.api_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            response_format={"type": "json_object"},
            max_tokens=self._model_spec.max_output_tokens,
        )
        api_time = time.monotonic() - api_start

        choice = response.choices[0]
        content = choice.message.content
        # Some endpoints return reasoning in a separate field and leave content empty
        if not content and hasattr(choice.message, "reasoning_content"):
            content = getattr(choice.message, "reasoning_content", None)
        self._logger.info(
            f"[OpenAICompatLLM] Response: finish={choice.finish_reason}, "
            f"content_len={len(content) if content else 0}, "
            f"content={repr(content[:300]) if content else 'NONE/EMPTY'}"
        )
        if not content or not content.strip():
            raise LLMTransientError(
                "OpenAI-compatible endpoint returned empty response content",
                provider=self._name,
            )

        if response.usage:
            self._logger.debug(
                f"[OpenAICompatLLM] API call completed in {api_time:.2f}s, "
                f"tokens: prompt={response.usage.prompt_tokens}, "
                f"completion={response.usage.completion_tokens}"
            )

        # Strip markdown code fences that some models wrap around JSON
        cleaned = content.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1:]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()

        return self._parse_json_response(cleaned, schema)

    async def _call_api_chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """OpenAI-compatible tool calling."""
        converted = self._prepare_messages_for_api(messages)
        api_messages = [{"role": "system", "content": system_prompt}, *converted]

        kwargs: dict[str, Any] = {
            "model": self._model_spec.api_model_name,
            "messages": api_messages,
            "tools": tools,
            "tool_choice": "required",
            "temperature": 0.6,
            "max_tokens": max_tokens or self._model_spec.max_output_tokens,
        }

        api_start = time.monotonic()
        response = await self._client.chat.completions.create(**kwargs)
        api_time = time.monotonic() - api_start

        choice = response.choices[0]
        message = choice.message

        text = message.content if message.content else None

        tool_calls: list[LLMToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                arguments = tc.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                tool_calls.append(
                    LLMToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        prompt_tokens = 0
        completion_tokens = 0
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            self._logger.debug(
                f"[OpenAICompatLLM] Tool call completed in {api_time:.2f}s, "
                f"tokens: prompt={prompt_tokens}, completion={completion_tokens}"
            )

        return LLMResponse(
            text=text,
            thinking=None,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def close(self) -> None:
        """Close the OpenAI async client."""
        if self._client:
            await self._client.close()
            self._logger.debug("[OpenAICompatLLM] Client closed")
