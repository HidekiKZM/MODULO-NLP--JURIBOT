"""Bounded Google GenAI adapter; no SDK globals, streaming or automatic tools."""
import asyncio
from dataclasses import dataclass
import logging

import httpx
from google import genai
from google.genai import errors, types

from backend.services.chat_errors import ChatFailure

logger = logging.getLogger(__name__)
TRANSIENT_CODES = {408, 500, 502, 503, 504}
BLOCKED_REASONS = {
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
    "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "MODEL_ARMOR",
}


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    usage: dict


def enum_value(value):
    return getattr(value, "value", value)


def validate_response(response):
    feedback = getattr(response, "prompt_feedback", None)
    blocked = enum_value(getattr(feedback, "block_reason", None))
    if blocked and blocked != "BLOCK_REASON_UNSPECIFIED":
        raise ChatFailure("gemini_blocked")
    candidates = getattr(response, "candidates", None) or []
    if len(candidates) != 1:
        raise ChatFailure("gemini_invalid_response")
    candidate = candidates[0]
    reason = enum_value(getattr(candidate, "finish_reason", None))
    ratings = getattr(candidate, "safety_ratings", None) or []
    if reason in BLOCKED_REASONS or any(getattr(rating, "blocked", False) for rating in ratings):
        raise ChatFailure("gemini_blocked")
    if reason == "MAX_TOKENS":
        raise ChatFailure("gemini_truncated")
    if reason != "STOP":
        raise ChatFailure("gemini_invalid_response")
    content = getattr(candidate, "content", None)
    chunks = []
    for part in getattr(content, "parts", None) or []:
        if getattr(part, "thought", False):
            continue
        if any(getattr(part, name, None) is not None for name in (
            "function_call", "function_response", "inline_data", "file_data",
            "executable_code", "code_execution_result",
        )):
            raise ChatFailure("gemini_invalid_response")
        text = getattr(part, "text", None)
        if not isinstance(text, str):
            raise ChatFailure("gemini_invalid_response")
        chunks.append(text)
    text = "".join(chunks).strip()
    if not text:
        raise ChatFailure("gemini_invalid_response")
    metadata = getattr(response, "usage_metadata", None)
    usage = {}
    for field in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count"):
        value = getattr(metadata, field, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            usage[field] = value
    return GeneratedAnswer(text, usage)


class GeminiGenerator:
    def __init__(self, settings, *, client=None):
        key, self.model = settings.require_gemini()
        self.settings = settings
        self.client = client if client is not None else genai.Client(
            api_key=key, enterprise=False,
            http_options=types.HttpOptions(
                timeout=max(1, int(settings.GEMINI_TIMEOUT_SECONDS * 1000)),
                # Retries have one owner; prevent multiplication of SDK/app attempts.
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    async def generate(self, question, *, system_instruction):
        try:
            return await asyncio.wait_for(
                self._generate_with_retries(question, system_instruction),
                timeout=self.settings.GEMINI_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as error:
            raise ChatFailure("gemini_timeout") from error

    async def _generate_with_retries(self, question, system_instruction):
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=self.settings.GEMINI_MAX_OUTPUT_TOKENS,
            candidate_count=1,
            response_mime_type="text/plain",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        for attempt in range(1, self.settings.GEMINI_MAX_ATTEMPTS + 1):
            failure = None
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model, contents=question, config=config,
                )
                return validate_response(response)
            except ChatFailure:
                raise
            except errors.APIError as error:
                if error.code == 429:
                    raise ChatFailure("gemini_quota_exceeded") from error
                if error.code in {400, 401, 403, 404}:
                    raise ChatFailure("gemini_configuration") from error
                if error.code not in TRANSIENT_CODES:
                    raise ChatFailure("gemini_invalid_response") from error
                failure = ChatFailure("gemini_timeout" if error.code in {408, 504} else "gemini_unavailable")
            except (httpx.TimeoutException, asyncio.TimeoutError):
                failure = ChatFailure("gemini_timeout")
            except httpx.TransportError:
                failure = ChatFailure("gemini_unavailable")
            except Exception as error:
                # Do not expose/log SDK exception messages: they can contain requests.
                logger.warning("Gemini adapter failed (%s)", type(error).__name__)
                raise ChatFailure("gemini_invalid_response") from error
            if attempt >= self.settings.GEMINI_MAX_ATTEMPTS:
                raise failure
            logger.info("Retrying Gemini transient failure (%s), attempt %s", failure.code, attempt + 1)
            await asyncio.sleep(self.settings.GEMINI_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)))
        raise ChatFailure("gemini_unavailable")

    async def aclose(self):
        try:
            await self.client.aio.aclose()
        finally:
            self.client.close()
