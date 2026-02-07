"""Retry logic with exponential backoff"""

import time
from typing import Callable, Any
from src.utils.logging import get_logger
from src.utils.errors import AuditSystemError

logger = get_logger(__name__)


def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 5,
    base_delay: int = 30,
    max_delay: int = 480,
    *args,
    **kwargs
) -> Any:
    """
    Retry function with exponential backoff

    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds (30s)
        max_delay: Max delay cap (480s = 8 min)
        *args, **kwargs: Arguments to pass to func

    Returns:
        Function result

    Raises:
        AuditSystemError: If all retries exhausted
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} retry attempts exhausted")
                raise AuditSystemError(f"Failed after {max_retries} attempts: {e}")

            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            time.sleep(delay)
