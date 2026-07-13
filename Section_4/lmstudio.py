import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"


def _base_url(base_url: str | None = None) -> str:
    return (base_url or os.getenv("LMSTUDIO_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _api_key(api_key: str | None = None) -> str:
    return api_key or os.getenv("LMSTUDIO_API_KEY") or DEFAULT_API_KEY


def _model(model: str | None = None) -> str:
    resolved = model or os.getenv("LMSTUDIO_MODEL")
    if not resolved:
        raise ValueError(
            "LM Studio model is required. Set LMSTUDIO_MODEL in .env or pass model=."
        )
    return resolved


def get_lmstudio_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAI:
    """OpenAI client pointed at the local LM Studio server."""
    return OpenAI(base_url=_base_url(base_url), api_key=_api_key(api_key))


def get_lmstudio_llm(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAI:
    """Alias for get_lmstudio_client (notebook-friendly name)."""
    return get_lmstudio_client(base_url=base_url, api_key=api_key)


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    client: OpenAI | None = None,
    temperature: float | None = None,
    **kwargs,
):
    """Run a chat completion against LM Studio."""
    client = client or get_lmstudio_client()
    request: dict = {
        "model": _model(model),
        "messages": messages,
        **kwargs,
    }
    if temperature is not None:
        request["temperature"] = temperature
    return client.chat.completions.create(**request)


if __name__ == "__main__":
    response = chat([{"role": "user", "content": "Say: LM Studio is connected."}])
    print(response.choices[0].message.content)
