"""OpenRouter LLM client with automatic cost tracking and model selection."""

from openai import OpenAI
import os
import time
from typing import Optional
from src.utils.metrics import llm_tokens_counter, llm_cost_counter, llm_api_latency
from src.utils.errors import LLMError
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy-initialize OpenRouter client
_client = None

def get_client():
    """Get or create the OpenRouter client (lazy initialization)"""
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    return _client

# Pricing per 1M tokens (input tokens, simplified)
MODEL_PRICING = {
    "anthropic/claude-haiku-4.5": 0.80 / 1_000_000,
    "anthropic/claude-sonnet-4.5": 3.0 / 1_000_000,
    "anthropic/claude-sonnet-4": 3.0 / 1_000_000,
    "openai/gpt-4o-mini": 0.15 / 1_000_000,  # kept for cost calc fallback
}


def call_llm(
    prompt: str,
    model: Optional[str] = None,
    agent_name: str = "unknown",
    max_retries: int = 3
) -> str:
    """
    Call LLM with automatic cost tracking and retries.

    Args:
        prompt: User prompt
        model: Model name (defaults to GPT-4o-mini)
        agent_name: Name of agent calling LLM (for metrics)
        max_retries: Max retry attempts

    Returns:
        LLM response text

    Raises:
        LLMError: If API call fails after retries
    """
    model = model or os.getenv("DEFAULT_LLM_MODEL", "anthropic/claude-haiku-4.5")

    for attempt in range(max_retries):
        try:
            start_time = time.time()

            response = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )

            # Track metrics
            latency = time.time() - start_time
            tokens = response.usage.total_tokens
            cost = calculate_cost(tokens, model)

            llm_tokens_counter.labels(model_name=model, agent_name=agent_name).inc(tokens)
            llm_cost_counter.labels(model_name=model).inc(cost)
            llm_api_latency.labels(model_name=model).observe(latency)

            logger.info(
                f"LLM call successful",
                model=model,
                tokens=tokens,
                cost=cost,
                latency=latency,
                agent=agent_name
            )

            return response.choices[0].message.content

        except Exception as e:
            # Check if it's a rate limit error
            if "rate_limit" in str(e).lower() or "429" in str(e):
                logger.warning(f"Rate limit hit, retrying... (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise LLMError(f"Rate limit exceeded after {max_retries} attempts: {e}")
            else:
                logger.error(f"LLM API error: {e}", attempt=attempt)
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise LLMError(f"LLM API call failed after {max_retries} attempts: {e}")


def calculate_cost(tokens: int, model: str) -> float:
    """
    Calculate cost based on token usage and model pricing.

    Args:
        tokens: Number of tokens used
        model: Model name

    Returns:
        Cost in USD
    """
    price_per_token = MODEL_PRICING.get(model, 0.15 / 1_000_000)
    return tokens * price_per_token


def batch_call_llm(prompts: list[str], model: Optional[str] = None) -> list[str]:
    """
    Batch LLM calls for efficiency (processes sequentially but with shared setup).

    Args:
        prompts: List of prompts
        model: Model name

    Returns:
        List of responses
    """
    return [call_llm(prompt, model) for prompt in prompts]
