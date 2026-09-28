"""
Lab 11 — Helper Utilities
"""
import asyncio

from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


# Transient provider failures worth retrying (Gemini 503/429 spikes, OpenRouter
# 429s, socket timeouts). Anything else is a real bug and should surface fast.
_RETRYABLE_MARKERS = (
    "503",
    "429",
    "500",
    "502",
    "504",
    "unavailable",
    "resource_exhausted",
    "overloaded",
    "high demand",
    "rate limit",
    "rate_limit",
    "timeout",
    "timed out",
    "connection",
)


def _is_retryable(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


async def _with_retry(call, *, attempts: int = 4, base_delay: float = 2.0):
    """Run an async call, retrying transient provider errors with backoff."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 — decide by marker, then re-raise
            last = exc
            if attempt == attempts - 1 or not _is_retryable(exc):
                raise
            delay = base_delay * (2 ** attempt)
            print(
                f"  [retry {attempt + 1}/{attempts - 1}] "
                f"{type(exc).__name__}: {str(exc)[:90]} — waiting {delay:.0f}s"
            )
            await asyncio.sleep(delay)
    if last:
        raise last


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    Transient provider errors (503/429/timeouts) are retried with backoff.
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await _with_retry(lambda: runner.chat(agent, user_message))
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    async def _run() -> str:
        final_response = ""
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=content
        ):
            if hasattr(event, "content") and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_response += part.text
        return final_response

    final_response = await _with_retry(_run)
    return final_response, session
