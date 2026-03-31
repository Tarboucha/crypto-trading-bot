"""Tests for retry decorator — backoff, error types, max attempts."""
import pytest
from shared.api.retry import retry
from shared.api.errors import RetryableError, PermanentError, RateLimitError


@pytest.mark.asyncio
class TestRetryBehavior:

    async def test_succeeds_first_try(self):
        call_count = 0

        @retry(max_attempts=3, backoff=[0, 0, 0])
        async def func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await func()
        assert result == "ok"
        assert call_count == 1

    async def test_retries_on_retryable(self):
        call_count = 0

        @retry(max_attempts=3, backoff=[0, 0, 0])
        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("fail")
            return "ok"

        result = await func()
        assert result == "ok"
        assert call_count == 3

    async def test_raises_after_max_attempts(self):
        call_count = 0

        @retry(max_attempts=3, backoff=[0, 0, 0])
        async def func():
            nonlocal call_count
            call_count += 1
            raise RetryableError("always fail")

        with pytest.raises(RetryableError):
            await func()
        assert call_count == 3

    async def test_permanent_error_no_retry(self):
        call_count = 0

        @retry(max_attempts=3, backoff=[0, 0, 0])
        async def func():
            nonlocal call_count
            call_count += 1
            raise PermanentError("permanent")

        with pytest.raises(PermanentError):
            await func()
        assert call_count == 1  # no retry

    async def test_rate_limit_uses_wait_seconds(self):
        call_count = 0

        @retry(max_attempts=2, backoff=[0])
        async def func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("rate limited", wait_seconds=0)
            return "ok"

        result = await func()
        assert result == "ok"
        assert call_count == 2

    async def test_non_matching_error_passes_through(self):
        @retry(max_attempts=3, backoff=[0, 0, 0])
        async def func():
            raise ValueError("not a bot error")

        with pytest.raises(ValueError):
            await func()

    async def test_preserves_return_value(self):
        @retry(max_attempts=1)
        async def func():
            return {"key": "value", "num": 42}

        result = await func()
        assert result == {"key": "value", "num": 42}
