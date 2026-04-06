from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Ticker:
    symbol: str
    last: float
    bid: float
    ask: float
    volume: float


@dataclass
class Balance:
    total: float
    free: float
    used: float


@dataclass
class Position:
    symbol: str
    side: str  # "long" | "short"
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: float


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str  # "buy" | "sell"
    order_type: str  # "market" | "limit"
    amount: float
    price: float | None
    status: str  # "pending" | "open" | "filled" | "partially_filled" | "cancelled" | "rejected"
    filled_size: float = 0.0
    fill_price: float = 0.0


class BaseExchange(ABC):
    """Interface that all exchange adapters must implement."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the exchange and validate credentials."""

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price info for a symbol."""

    @abstractmethod
    async def get_candles(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> pd.DataFrame:
        """Fetch OHLCV candles. Returns DataFrame with columns:
        timestamp, open, high, low, close, volume"""

    @abstractmethod
    async def get_balance(self) -> Balance:
        """Get account balance."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get open positions."""

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
    ) -> OrderResult:
        """Place an order."""

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order. Returns True if successful."""

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> OrderResult:
        """Get the current status of an order."""

    async def get_max_leverage(self, symbol: str) -> float:
        """Get maximum leverage allowed for this pair on the exchange."""
        raise NotImplementedError

    async def set_leverage(self, symbol: str, leverage: float) -> None:
        """Set leverage for a pair on the exchange (required before order on some exchanges)."""
        raise NotImplementedError

    async def get_maintenance_margin_rate(self, symbol: str) -> float:
        """Get maintenance margin rate for liquidation calculation."""
        raise NotImplementedError

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> float:
        """Get the current funding rate for a symbol."""

    @abstractmethod
    async def get_funding_history(self, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
        """Get historical funding rates. Returns list of {timestamp, rate}."""
