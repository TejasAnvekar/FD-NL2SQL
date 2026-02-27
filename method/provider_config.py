#!/usr/bin/env python3
"""Provider presets for OpenAI-compatible endpoints."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    api_base: str
    api_key_env: str
    default_model: str
    requires_api_key: bool = True


PROVIDER_CHOICES = [
    "local_vllm",
    "openai",
    "gemini",
    "openrouter",
    "together",
    "custom",
]


_PROVIDER_PRESETS: Dict[str, ProviderPreset] = {
    "local_vllm": ProviderPreset(
        name="local_vllm",
        api_base="http://127.0.0.1:8000/v1",
        api_key_env="VLLM_API_KEY",
        default_model="qwen3-30b-local",
        requires_api_key=False,
    ),
    "openai": ProviderPreset(
        name="openai",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        requires_api_key=True,
    ),
    "gemini": ProviderPreset(
        name="gemini",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
        requires_api_key=True,
    ),
    "openrouter": ProviderPreset(
        name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="openai/gpt-4o-mini",
        requires_api_key=True,
    ),
    "together": ProviderPreset(
        name="together",
        api_base="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        default_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        requires_api_key=True,
    ),
    "custom": ProviderPreset(
        name="custom",
        api_base="",
        api_key_env="",
        default_model="",
        requires_api_key=False,
    ),
}


_ALIASES = {
    "vllm": "local_vllm",
    "local": "local_vllm",
    "google": "gemini",
    "router": "openrouter",
}


def normalize_provider(name: str) -> str:
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _PROVIDER_PRESETS:
        raise ValueError(
            f"Unknown provider '{name}'. Valid choices: {', '.join(PROVIDER_CHOICES)}"
        )
    return key


def _first_env(names: list[str]) -> Tuple[Optional[str], Optional[str]]:
    for n in names:
        if not n:
            continue
        v = os.getenv(n)
        if v and v.strip():
            return v.strip(), n
    return None, None


def resolve_openai_compat(
    *,
    provider: str,
    api_base: str,
    api_key: str,
    model_name: str,
    api_key_env: str = "",
    local_default_api_base: str = "",
    local_default_model_name: str = "",
) -> Tuple[str, str, str, Dict[str, Any]]:
    """Resolve api_base/api_key/model_name from provider presets + env."""
    pname = normalize_provider(provider)
    preset = _PROVIDER_PRESETS[pname]

    resolved_base = (api_base or "").strip()
    resolved_key = (api_key or "").strip()
    resolved_model = (model_name or "").strip()

    base_is_local_default = bool(local_default_api_base) and resolved_base == local_default_api_base
    model_is_local_default = bool(local_default_model_name) and resolved_model == local_default_model_name

    # If caller selected a cloud provider but left local defaults, swap to provider preset.
    if pname not in {"local_vllm", "custom"}:
        if not resolved_base or base_is_local_default:
            resolved_base = preset.api_base
        if (not resolved_model or model_is_local_default) and preset.default_model:
            resolved_model = preset.default_model
    else:
        if not resolved_base and preset.api_base:
            resolved_base = preset.api_base
        if not resolved_model and preset.default_model:
            resolved_model = preset.default_model

    env_candidates: list[str] = []
    if (api_key_env or "").strip():
        env_candidates.append(api_key_env.strip())
    if preset.api_key_env and preset.api_key_env not in env_candidates:
        env_candidates.append(preset.api_key_env)
    env_key, env_used = _first_env(env_candidates)

    key_from_env = False
    if (not resolved_key) or resolved_key == "dummy":
        if env_key:
            resolved_key = env_key
            key_from_env = True

    if not resolved_key:
        resolved_key = "dummy"

    if pname not in {"local_vllm", "custom"} and preset.requires_api_key and resolved_key == "dummy":
        env_hint = ", ".join(env_candidates) if env_candidates else "<no env configured>"
        raise ValueError(
            f"Provider '{pname}' requires an API key. Set --api-key or export one of: {env_hint}"
        )

    meta = {
        "provider": pname,
        "api_base": resolved_base,
        "model_name": resolved_model,
        "api_key_env_used": env_used,
        "api_key_from_env": key_from_env,
    }
    return resolved_base, resolved_key, resolved_model, meta
