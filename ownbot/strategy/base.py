"""Base strategy interface. All strategies inherit from this."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Signal:
    pair: str
    direction: str          # "long" | "short"
    action: str             # "enter" | "exit"
    confidence: float       # 0.0 to 1.0
    reason: str             # human-readable explanation
    timestamp: int = 0
    strategy: str = ""
    timeframe: str = ""


class BaseStrategy(ABC):
    """Interface that all strategies must implement.

    Strategies are mode-agnostic: they receive data and return signals.
    They never call the exchange or DB directly.

    Data convention:
        data dict keys follow this pattern:
        - "{timeframe}" for the current pair: "5m", "1h"
        - "{pair}_{timeframe}" for extra pairs: "BTC_5m", "ETH_1h"
    """

    name: str = "base"
    timeframes: list[str] = field(default_factory=lambda: ["5m"])
    startup_candle_count: int = 100

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def required_pairs(self) -> list[str]:
        """Declare additional pairs this strategy needs besides the one being evaluated.

        Override this if your strategy uses cross-pair data (e.g., BTC correlation).
        The engine will fetch candles for these pairs and include them in the data dict
        as "{pair}_{timeframe}" keys.

        Returns:
            List of pair names, e.g. ["BTC"] or ["BTC", "ETH"]
        """
        return []

    @abstractmethod
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to the candle DataFrame.

        Args:
            df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            Same DataFrame with indicator columns added.
        """

    @abstractmethod
    def should_enter(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        """Check if we should open a position.

        Args:
            pair: The trading pair (e.g. "ETH")
            data: Dict of timeframe -> DataFrame with indicators already applied.
                  e.g. {"5m": df_5m, "1h": df_1h}

        Returns:
            Signal if entry condition met, None otherwise.
        """

    @abstractmethod
    def should_exit(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        """Check if we should close a position.

        Args:
            pair: The trading pair
            data: Dict of timeframe -> DataFrame with indicators already applied.

        Returns:
            Signal if exit condition met, None otherwise.
        """

    def evaluate(
        self, pair: str, data: dict[str, pd.DataFrame], has_position: bool = False
    ) -> Signal | None:
        """Run indicators and check for entry/exit signals.

        Args:
            pair: Trading pair
            data: Dict of timeframe -> DataFrame
            has_position: Whether we currently have an open position for this pair.
                          If True, check exit. If False, check entry.
        """
        # Apply indicators to current pair's timeframes only (not extra pairs)
        for tf in self.timeframes:
            if tf in data:
                data[tf] = self.indicators(data[tf])

        if has_position:
            signal = self.should_exit(pair, data)
        else:
            signal = self.should_enter(pair, data)

        if signal:
            signal.strategy = self.name
            return signal

        return None
