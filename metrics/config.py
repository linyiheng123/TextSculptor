"""Environment-based model configuration."""

import os
import random
from typing import Any


def _split_keys(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_api_config() -> dict[str, Any]:
    """Build an Azure OpenAI config from environment variables."""
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    model_name = os.environ.get("AZURE_OPENAI_MODEL")
    keys = (
        _split_keys(os.environ.get("AZURE_OPENAI_API_KEYS"))
        or _split_keys(os.environ.get("AZURE_OPENAI_API_KEY"))
    )

    missing = []
    if not azure_endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not api_version:
        missing.append("AZURE_OPENAI_API_VERSION")
    if not model_name:
        missing.append("AZURE_OPENAI_MODEL")
    if not keys:
        missing.append("AZURE_OPENAI_API_KEYS or AZURE_OPENAI_API_KEY")
    if missing:
        raise ValueError("Missing model configuration environment variables: " + ", ".join(missing))

    return {
        "provider": "azure_openai",
        "azure_endpoint": azure_endpoint,
        "api_version": api_version,
        "model_name": model_name,
        "keys": keys,
    }


def get_random_api_key() -> str:
    """Return one configured API key."""
    return random.choice(get_api_config()["keys"])


def check_api_configured() -> bool:
    """Return whether all required environment variables are present."""
    try:
        get_api_config()
        return True
    except ValueError:
        return False
