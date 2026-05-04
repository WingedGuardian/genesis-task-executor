"""LLM provider implementations."""

from __future__ import annotations

__all__ = ["get_provider"]


def get_provider(
    name: str = "openai",
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
):
    """Get an LLM provider by name.

    Args:
        name: Provider name ("openai" or "anthropic")
        api_key: API key (falls back to env var)
        model: Model name (provider-specific default if None)
        base_url: Optional base URL for OpenAI-compatible APIs
    """
    if name == "openai":
        from genesis_task_executor.providers.openai import OpenAIProvider
        return OpenAIProvider(api_key=api_key, model=model, base_url=base_url)
    elif name == "anthropic":
        from genesis_task_executor.providers.anthropic import AnthropicProvider
        return AnthropicProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {name!r}. Use 'openai' or 'anthropic'.")
