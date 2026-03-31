"""RSI Mean Reversion strategy — buy oversold, sell overbought with Bollinger Band confirmation."""
import pandas as pd

from ownbot.strategy.base import BaseStrategy, Signal
from ownbot.strategy.indicators import rsi, bbands


class RSIMeanReversionStrategy(BaseStrategy):
    name = "rsi_mean_reversion"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.timeframes = [self.params.get("timeframe", "1m")]
        self.rsi_period = self.params.get("rsi_period", 14)
        self.rsi_oversold = self.params.get("rsi_oversold", 25)
        self.rsi_overbought = self.params.get("rsi_overbought", 75)
        self.rsi_exit = self.params.get("rsi_exit", 50)
        self.bb_period = self.params.get("bb_period", 20)
        self.bb_std = self.params.get("bb_std", 2.0)
        self.direction = self.params.get("direction", "both")

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = rsi(df, period=self.rsi_period)
        df = bbands(df, period=self.bb_period, std=self.bb_std)
        return df

    def should_enter(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]

        if len(df) < self.bb_period + 1:
            return None

        curr = df.iloc[-1]
        rsi_col = f"rsi_{self.rsi_period}"
        rsi_val = curr[rsi_col]
        close = curr["close"]

        # Skip if indicators not ready (NaN)
        if pd.isna(rsi_val) or pd.isna(curr.get("bb_lower")):
            return None

        # Entry LONG: RSI oversold AND price near/below lower Bollinger Band
        if (
            rsi_val < self.rsi_oversold
            and close <= curr["bb_lower"] * 1.005  # within 0.5% of lower band
            and self.direction in ("long", "both")
        ):
            confidence = min(1.0, (self.rsi_oversold - rsi_val) / 20)
            return Signal(
                pair=pair,
                direction="long",
                action="enter",
                confidence=confidence,
                reason=f"RSI oversold ({rsi_val:.1f}), price near BB lower ({curr['bb_lower']:.2f})",
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        # Entry SHORT: RSI overbought AND price near/above upper Bollinger Band
        if (
            rsi_val > self.rsi_overbought
            and close >= curr["bb_upper"] * 0.995  # within 0.5% of upper band
            and self.direction in ("short", "both")
        ):
            confidence = min(1.0, (rsi_val - self.rsi_overbought) / 20)
            return Signal(
                pair=pair,
                direction="short",
                action="enter",
                confidence=confidence,
                reason=f"RSI overbought ({rsi_val:.1f}), price near BB upper ({curr['bb_upper']:.2f})",
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        return None

    def should_exit(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]

        if len(df) < self.bb_period + 1:
            return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        rsi_col = f"rsi_{self.rsi_period}"
        rsi_val = curr[rsi_col]
        prev_rsi = prev[rsi_col]

        if pd.isna(rsi_val) or pd.isna(prev_rsi):
            return None

        # Exit LONG: RSI crosses above midline (mean reverted)
        if prev_rsi < self.rsi_exit and rsi_val >= self.rsi_exit:
            return Signal(
                pair=pair,
                direction="long",
                action="exit",
                confidence=1.0,
                reason=f"RSI crossed above {self.rsi_exit} ({rsi_val:.1f}) — mean reverted",
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        # Exit SHORT: RSI crosses below midline (mean reverted)
        if prev_rsi > self.rsi_exit and rsi_val <= self.rsi_exit:
            return Signal(
                pair=pair,
                direction="short",
                action="exit",
                confidence=1.0,
                reason=f"RSI crossed below {self.rsi_exit} ({rsi_val:.1f}) — mean reverted",
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        return None
