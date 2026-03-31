"""Trend Follow strategy — SMA crossover with RSI filter."""
import pandas as pd

from ownbot.strategy.base import BaseStrategy, Signal
from ownbot.strategy.indicators import sma, rsi


class TrendFollowStrategy(BaseStrategy):
    name = "trend_follow"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.timeframes = [self.params.get("timeframe", "5m")]
        self.fast_ma = self.params.get("fast_ma", 5)
        self.slow_ma = self.params.get("slow_ma", 20)
        self.rsi_period = self.params.get("rsi_period", 14)
        self.rsi_oversold = self.params.get("rsi_oversold", 30)
        self.rsi_overbought = self.params.get("rsi_overbought", 70)
        self.direction = self.params.get("direction", "both")

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = sma(df, period=self.fast_ma)
        df = sma(df, period=self.slow_ma)
        df = rsi(df, period=self.rsi_period)
        return df

    def should_enter(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]

        if len(df) < self.slow_ma + 1:
            return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        fast_col = f"sma_{self.fast_ma}"
        slow_col = f"sma_{self.slow_ma}"
        rsi_col = f"rsi_{self.rsi_period}"

        # Check for crossover: fast was below slow, now above
        cross_up = prev[fast_col] <= prev[slow_col] and curr[fast_col] > curr[slow_col]
        # Check for crossunder: fast was above slow, now below
        cross_down = prev[fast_col] >= prev[slow_col] and curr[fast_col] < curr[slow_col]

        rsi_val = curr[rsi_col]

        # Entry LONG: fast crosses above slow AND RSI not overbought
        if cross_up and rsi_val < self.rsi_overbought and self.direction in ("long", "both"):
            return Signal(
                pair=pair,
                direction="long",
                action="enter",
                confidence=min(1.0, (self.rsi_overbought - rsi_val) / 40),
                reason=f"SMA{self.fast_ma} crossed above SMA{self.slow_ma}, RSI={rsi_val:.1f}",
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        # Entry SHORT: fast crosses below slow AND RSI not oversold
        if cross_down and rsi_val > self.rsi_oversold and self.direction in ("short", "both"):
            return Signal(
                pair=pair,
                direction="short",
                action="enter",
                confidence=min(1.0, (rsi_val - self.rsi_oversold) / 40),
                reason=f"SMA{self.fast_ma} crossed below SMA{self.slow_ma}, RSI={rsi_val:.1f}",
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        return None

    def should_exit(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]

        if len(df) < self.slow_ma + 1:
            return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        fast_col = f"sma_{self.fast_ma}"
        slow_col = f"sma_{self.slow_ma}"
        rsi_col = f"rsi_{self.rsi_period}"

        rsi_val = curr[rsi_col]

        # Exit LONG: fast crosses below slow OR RSI overbought
        cross_down = prev[fast_col] >= prev[slow_col] and curr[fast_col] < curr[slow_col]
        if cross_down or rsi_val > self.rsi_overbought:
            reason = f"SMA{self.fast_ma} crossed below SMA{self.slow_ma}" if cross_down else f"RSI overbought ({rsi_val:.1f})"
            return Signal(
                pair=pair,
                direction="long",
                action="exit",
                confidence=1.0,
                reason=reason,
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        # Exit SHORT: fast crosses above slow OR RSI oversold
        cross_up = prev[fast_col] <= prev[slow_col] and curr[fast_col] > curr[slow_col]
        if cross_up or rsi_val < self.rsi_oversold:
            reason = f"SMA{self.fast_ma} crossed above SMA{self.slow_ma}" if cross_up else f"RSI oversold ({rsi_val:.1f})"
            return Signal(
                pair=pair,
                direction="short",
                action="exit",
                confidence=1.0,
                reason=reason,
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        return None
