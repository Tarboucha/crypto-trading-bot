"""Error classification for the entire bot.

Retry decorator only retries RetryableError and subclasses.
PermanentError and subclasses are never retried.
"""


class BotError(Exception):
    """Base for all bot errors."""


# --- Operational (config/setup — bot should stop) ---

class OperationalError(BotError):
    """Requires manual fix. Bot should stop."""


class ConfigError(OperationalError):
    """Invalid configuration."""


# --- Exchange errors ---

class ExchangeError(BotError):
    """Base for exchange-related errors."""


class RetryableError(ExchangeError):
    """Transient — safe to retry (network timeout, server error)."""


class RateLimitError(RetryableError):
    """Rate limited by exchange — retry with longer backoff."""
    def __init__(self, message: str = "Rate limited", wait_seconds: float = 10.0):
        super().__init__(message)
        self.wait_seconds = wait_seconds


class ExchangeDownError(RetryableError):
    """Exchange is completely unreachable."""


class PermanentError(ExchangeError):
    """Logic error — retrying won't help."""


class AuthError(PermanentError):
    """Wrong API key/secret."""


class InsufficientFundsError(PermanentError):
    """Not enough balance to place order."""


# --- Order-specific errors ---

class OrderError(ExchangeError):
    """Base for order errors."""


class OrderRejectedError(PermanentError):
    """Exchange rejected the order."""
    def __init__(self, reason: str, pair: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.pair = pair


class OrderUnknownStateError(OrderError):
    """Order sent but response lost — state is UNKNOWN."""
    def __init__(self, pair: str = "", order_id: str = ""):
        super().__init__(f"Unknown order state for {pair} (order_id={order_id})")
        self.pair = pair
        self.order_id = order_id


class OrderPartialFillError(OrderError):
    """Order partially filled."""
    def __init__(self, pair: str, order_id: str, filled_size: float, requested_size: float):
        super().__init__(f"Partial fill {pair}: {filled_size}/{requested_size}")
        self.pair = pair
        self.order_id = order_id
        self.filled_size = filled_size
        self.requested_size = requested_size


class OrderNotFoundError(RetryableError):
    """Order not found on exchange — may appear later, retry."""


# --- Strategy errors ---

class StrategyError(BotError):
    """Strategy code raised an error — skip, don't crash."""


# --- Pricing errors ---

class PricingError(BotError):
    """Could not determine price for a pair."""
