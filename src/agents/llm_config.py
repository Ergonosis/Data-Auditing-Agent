"""Shared LLM configuration for all CrewAI agents"""

import os
from crewai import LLM
from src.utils.logging import get_logger

logger = get_logger(__name__)

def get_agent_llm(temperature: float = 0.1) -> LLM:
    """
    Get configured LLM for agent orchestration.

    Uses OpenRouter with Claude Sonnet for reliable tool calling and JSON structuring.
    Separate from tools' llm_client.py to allow independent model selection.

    Args:
        temperature: Sampling temperature (0.1 for deterministic tool calling)

    Returns:
        Configured CrewAI LLM instance
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    # Use explicit provider="openai" to force CrewAI's OpenAI-compat SDK path.
    # CrewAI 1.9.3+ removed the "openrouter/" prefix fallback to LiteLLM — using
    # provider="openai" bypasses model-name validation and routes through the
    # OpenAI SDK (already installed) to OpenRouter's API which is OpenAI-compatible.
    llm = LLM(
        model="anthropic/claude-sonnet-4.5",
        provider="openai",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=4096  # Sufficient for complex JSON structures
    )

    logger.info(f"Initialized agent LLM: anthropic/claude-sonnet-4.5 via OpenRouter (openai-compat path, temp={temperature})")
    return llm


# Singleton instance for reuse across agents
_agent_llm_instance = None

def get_shared_agent_llm() -> LLM:
    """Get or create shared LLM instance for all agents"""
    global _agent_llm_instance
    if _agent_llm_instance is None:
        _agent_llm_instance = get_agent_llm()
    return _agent_llm_instance
