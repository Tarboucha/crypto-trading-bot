"""System events — engine lifecycle, errors, reconciliation."""
from dataclasses import dataclass, field

from shared.events.base import Event


# --- Engine Lifecycle ---

@dataclass(frozen=True)
class EngineStarted(Event):
    """Bot engine started."""
    mode: str = ""
    strategy: str = ""
    pairs: tuple = ()
    session_id: int = 0


@dataclass(frozen=True)
class EngineStopped(Event):
    """Bot engine stopped."""
    reason: str = ""
    session_id: int = 0
    total_trades: int = 0
    pnl_pct: float = 0.0
    pnl_abs: float = 0.0


@dataclass(frozen=True)
class TickStart(Event):
    """Engine begins processing a new tick."""
    tick_number: int = 0


@dataclass(frozen=True)
class TickComplete(Event):
    """Engine finished processing a tick."""
    tick_number: int = 0
    pairs_processed: int = 0
    elapsed_s: float = 0.0


# --- Reconciliation ---

@dataclass(frozen=True)
class ReconcileMismatch(Event):
    """State mismatch detected between bot and exchange."""
    pair: str = ""
    mismatch_type: str = ""   # "phantom" | "untracked" | "size_diff"
    details: str = ""


# --- Errors ---

@dataclass(frozen=True)
class ExchangeError(Event):
    """Exchange API error."""
    pair: str = ""
    error: str = ""
    retryable: bool = False


@dataclass(frozen=True)
class StrategyError(Event):
    """Strategy code raised an error."""
    pair: str = ""
    strategy: str = ""
    error: str = ""


@dataclass(frozen=True)
class OrderError(Event):
    """Order-specific error."""
    pair: str = ""
    order_id: str = ""
    error: str = ""
    error_type: str = ""      # "rejected" | "unknown_state" | "partial_fill"
