"""
llm_client.py
-------------
Shared helper for making LLM calls through OpenRouter.
"""

from os import getenv
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv()

# Initialize the client lazily so deterministic paths can run without an API key.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Create (once) and return the OpenRouter-backed OpenAI client.

    Raises a clear RuntimeError only when an LLM call is actually attempted
    without a configured API key, rather than at import time.
    """
    global _client
    if _client is None:
        api_key = getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found. "
                "Create a .env file containing:\n"
                "OPENROUTER_API_KEY=your_key_here"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 600,
    model: str = "deepseek/deepseek-v4-flash",#openai/gpt-oss-120b:free",
) -> str:
    """
    Send a chat completion request to OpenRouter.

    Returns
    -------
    str
        The assistant's response text.
    """

    response = _get_client().chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"reasoning": {"enabled": False}},
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    msg = response.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        # Fail loudly so node fallbacks record WHY the LLM was skipped
        # (e.g. finish_reason='length' = token budget exhausted) instead
        # of a cryptic AttributeError from None.strip().
        raise RuntimeError(
            f"LLM returned empty content "
            f"(finish_reason={response.choices[0].finish_reason!r}, "
            f"model={model})"
        )
    return content
