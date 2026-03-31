"""Retry decorator for async functions.

Only retries RetryableError and subclasses.
PermanentError passes through immediately.
RateLimitError uses the error's wait_seconds instead of backoff.
"""
import asyncio
import functools
import logging
from typing import Callable

from shared.api.errors import RetryableError, RateLimitError, PermanentError

logger = logging.getLogger(__name__)

DEFAULT_BACKOFF = [1, 2, 5]


def retry(
    max_attempts: int = 3,
    backoff: list[float] | None = None,
    retry_on: type = RetryableError,
):
    """Decorator that retries async functions on transient errors.

    Args:
        max_attempts: Total attempts (including first call)
        backoff: Wait seconds between retries [1st, 2nd, 3rd, ...]
        retry_on: Exception type to retry on (and subclasses)

    Usage:
        @retry(max_attempts=3, backoff=[1, 2, 5])
        async def get_ticker(self, symbol):
            ...
    """
    backoff = backoff or DEFAULT_BACKOFF

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except PermanentError:
                    raise  # never retry
                except RateLimitError as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            "%s rate limited — waiting %.1fs (attempt %d/%d)",
                            func.__qualname__, e.wait_seconds, attempt + 1, max_attempts,
                        )
                        await asyncio.sleep(e.wait_seconds)
                    else:
                        raise
                except retry_on as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        wait = backoff[min(attempt, len(backoff) - 1)]
                        logger.warning(
                            "%s failed — retrying in %.1fs (attempt %d/%d): %s",
                            func.__qualname__, wait, attempt + 1, max_attempts, e,
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise

            raise last_error

        return wrapper
    return decorator
