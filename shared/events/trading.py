"""Trading events — signals, orders, positions."""
from dataclasses import dataclass

from shared.events.base import Event


# --- Signal Events ---

@dataclass(frozen=True)
class SignalEntry(Event):
    """Strategy detected an entry opportunity."""
    pair: str = ""
    direction: str = ""       # "long" | "short"
    confidence: float = 0.0
    reason: str = ""
    strategy: str = ""
    timeframe: str = ""


@dataclass(frozen=True)
class SignalExit(Event):
    """Strategy detected an exit condition."""
    pair: str = ""
    direction: str = ""
    confidence: float = 0.0
    reason: str = ""
    strategy: str = ""


@dataclass(frozen=True)
class SignalRejected(Event):
    """Risk manager rejected a signal."""
    pair: str = ""
    direction: str = ""
    reason: str = ""


# --- Order Events ---

@dataclass(frozen=True)
class OrderSubmitted(Event):
    """Order sent to exchange."""
    pair: str = ""
    side: str = ""            # "buy" | "sell"
    order_type: str = ""      # "market" | "limit"
    size: float = 0.0
    price: float = 0.0
    order_id: str = ""


@dataclass(frozen=True)
class OrderFilled(Event):
    """Order confirmed filled."""
    pair: str = ""
    side: str = ""
    size: float = 0.0
    price: float = 0.0
    fee: float = 0.0
    order_id: str = ""


@dataclass(frozen=True)
class OrderPartiallyFilled(Event):
    """Order partially filled."""
    pair: str = ""
    side: str = ""
    filled_size: float = 0.0
    requested_size: float = 0.0
    price: float = 0.0
    order_id: str = ""


@dataclass(frozen=True)
class OrderCancelled(Event):
    """Order cancelled or timed out."""
    pair: str = ""
    order_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class OrderRejected(Event):
    """Exchange rejected the order."""
    pair: str = ""
    order_id: str = ""
    reason: str = ""


# --- Position Events ---

@dataclass(frozen=True)
class PositionOpened(Event):
    """New position opened."""
    pair: str = ""
    direction: str = ""
    entry_price: float = 0.0
    size: float = 0.0
    stoploss: float | None = None
    take_profit: float | None = None
    strategy: str = ""


@dataclass(frozen=True)
class PositionClosed(Event):
    """Position closed with final P&L."""
    pair: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    profit_pct: float = 0.0
    profit_abs: float = 0.0
    funding_pnl: float = 0.0
    entry_time: int = 0       # ms since epoch
    reason: str = ""
    strategy: str = ""


@dataclass(frozen=True)
class PositionUpdated(Event):
    """Position state changed (size, funding, etc.)."""
    pair: str = ""
    direction: str = ""
    size: float = 0.0
    cumulative_funding: float = 0.0


@dataclass(frozen=True)
class StoplossHit(Event):
    """Stoploss triggered."""
    pair: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    profit_pct: float = 0.0
    profit_abs: float = 0.0


@dataclass(frozen=True)
class TakeprofitHit(Event):
    """Takeprofit triggered."""
    pair: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    profit_pct: float = 0.0
    profit_abs: float = 0.0


# --- Funding Events ---

@dataclass(frozen=True)
class FundingApplied(Event):
    """Funding rate applied to a position."""
    pair: str = ""
    rate: float = 0.0
    amount: float = 0.0       # positive = received, negative = paid
    cumulative: float = 0.0
