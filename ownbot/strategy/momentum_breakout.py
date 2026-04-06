"""Momentum Breakout strategy — trend following with ATR risk management.

Core idea: trade in the direction of the longer-term trend,
enter on momentum breakouts confirmed by volume, manage risk with ATR.

All computed on a single timeframe (5m). Uses longer EMA periods
to approximate higher-timeframe trend (EMA252 on 5m ≈ EMA21 on 1h).

Signals:
  ENTRY:
    1. Trend filter: fast EMA > slow EMA (longer-term direction)
    2. Momentum: price breaks above/below Donchian channel
    3. Volume confirmation: current volume > N x average
    4. Volatility filter: ATR expanding (current ATR > its SMA)
    5. RSI not extreme (avoid chasing)

  EXIT:
    1. Chandelier stop: trailing stop at N * ATR from highest high
    2. Trend reversal: fast EMA crosses back below slow EMA
    3. RSI extreme

  SIZING:
    ATR-based leverage: low vol → more leverage, high vol → less

References: Turtle Trading, Keltner Channel, Elder's Triple Screen
"""
import pandas as pd
import numpy as np

from ownbot.strategy.base import BaseStrategy, Signal
from ownbot.strategy.indicators import ema, rsi, atr, volume_sma


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian Channel — highest high and lowest low over N periods."""
    df[f"dc_upper_{period}"] = df["high"].rolling(period).max()
    df[f"dc_lower_{period}"] = df["low"].rolling(period).min()
    return df


class MomentumBreakoutStrategy(BaseStrategy):
    name = "momentum_breakout"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        tf = self.params.get("timeframe", "5m")
        self.timeframes = [tf]

        # Trend filter (long EMAs on 5m ≈ shorter EMAs on higher TF)
        # EMA252 on 5m ≈ EMA21 on 1h, EMA660 ≈ EMA55 on 1h
        self.trend_fast_ema = self.params.get("trend_fast_ema", 252)
        self.trend_slow_ema = self.params.get("trend_slow_ema", 660)

        # Entry: Donchian breakout
        self.dc_period = self.params.get("dc_period", 20)

        # ATR
        self.atr_period = self.params.get("atr_period", 14)
        self.atr_sma_period = self.params.get("atr_sma_period", 50)

        # Volume
        self.vol_sma_period = self.params.get("vol_sma_period", 20)
        self.vol_mult = self.params.get("vol_mult", 1.3)

        # RSI
        self.rsi_period = self.params.get("rsi_period", 14)
        self.rsi_entry_max = self.params.get("rsi_entry_max", 70)
        self.rsi_entry_min = self.params.get("rsi_entry_min", 30)

        # Exit: Chandelier
        self.chandelier_mult = self.params.get("chandelier_mult", 3.0)
        self.chandelier_period = self.params.get("chandelier_period", 22)

        # Exit: RSI
        self.rsi_exit_long = self.params.get("rsi_exit_long", 80)
        self.rsi_exit_short = self.params.get("rsi_exit_short", 20)

        # Direction
        self.direction = self.params.get("direction", "both")

        # Cooldown: don't re-enter within N candles of last exit
        self.cooldown_candles = self.params.get("cooldown_candles", 12)
        self._last_exit_ts = 0

        # Need enough candles for the slow EMA
        self.startup_candle_count = self.trend_slow_ema + 50

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ema(df, period=self.trend_fast_ema)
        df = ema(df, period=self.trend_slow_ema)
        df = rsi(df, period=self.rsi_period)
        df = atr(df, period=self.atr_period)
        df = donchian(df, period=self.dc_period)
        df = volume_sma(df, period=self.vol_sma_period)

        # ATR SMA for volatility expansion detection
        atr_col = f"atr_{self.atr_period}"
        df[f"atr_sma_{self.atr_sma_period}"] = df[atr_col].rolling(self.atr_sma_period).mean()

        # Chandelier exit levels
        df["chandelier_long"] = (
            df["high"].rolling(self.chandelier_period).max()
            - self.chandelier_mult * df[atr_col]
        )
        df["chandelier_short"] = (
            df["low"].rolling(self.chandelier_period).min()
            + self.chandelier_mult * df[atr_col]
        )
        return df

    def leverage(self, pair: str, direction: str, data: dict[str, pd.DataFrame]) -> float:
        """ATR-based leverage: lower vol → more leverage, higher vol → less."""
        df = data[self.timeframes[0]]
        curr = df.iloc[-1]
        atr_val = curr.get(f"atr_{self.atr_period}")
        close = curr["close"]

        if pd.isna(atr_val) or close == 0 or atr_val == 0:
            return 1.0

        atr_pct = atr_val / close
        # Target: risk ~1% per chandelier stop distance
        lev = 0.01 / (atr_pct * self.chandelier_mult)
        return max(1.0, min(round(lev, 1), 5.0))

    def should_enter(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]

        if len(df) < self.startup_candle_count:
            return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # Cooldown check
        if self._last_exit_ts > 0:
            candle_ms = 5 * 60 * 1000  # 5 min
            if curr["timestamp"] - self._last_exit_ts < self.cooldown_candles * candle_ms:
                return None

        # Column names
        fast_col = f"ema_{self.trend_fast_ema}"
        slow_col = f"ema_{self.trend_slow_ema}"
        dc_upper = f"dc_upper_{self.dc_period}"
        dc_lower = f"dc_lower_{self.dc_period}"
        atr_col = f"atr_{self.atr_period}"
        atr_sma_col = f"atr_sma_{self.atr_sma_period}"
        rsi_col = f"rsi_{self.rsi_period}"
        vol_sma_col = f"vol_sma_{self.vol_sma_period}"

        # Check NaN
        needed = [fast_col, slow_col, dc_upper, dc_lower, atr_col, atr_sma_col, rsi_col, vol_sma_col]
        if any(pd.isna(curr.get(c)) for c in needed):
            return None
        if any(pd.isna(prev.get(c)) for c in [dc_upper, dc_lower]):
            return None

        # Trend direction
        trend_long = curr[fast_col] > curr[slow_col]
        trend_short = curr[fast_col] < curr[slow_col]

        rsi_val = curr[rsi_col]
        vol_ok = curr["volume"] > self.vol_mult * curr[vol_sma_col]
        atr_expanding = curr[atr_col] > curr[atr_sma_col]

        # LONG: trend up + breakout above Donchian high + confirmations
        if (trend_long
                and self.direction in ("long", "both")
                and curr["close"] > prev[dc_upper]
                and rsi_val < self.rsi_entry_max
                and vol_ok
                and atr_expanding):

            breakout_dist = (curr["close"] - prev[dc_upper]) / curr[atr_col]
            confidence = min(1.0, 0.5 + breakout_dist * 0.3)

            return Signal(
                pair=pair,
                direction="long",
                action="enter",
                confidence=confidence,
                reason=(f"Breakout DC{self.dc_period} {prev[dc_upper]:.0f}, "
                        f"RSI={rsi_val:.0f}, vol={curr['volume']/curr[vol_sma_col]:.1f}x"),
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        # SHORT: trend down + breakout below Donchian low + confirmations
        if (trend_short
                and self.direction in ("short", "both")
                and curr["close"] < prev[dc_lower]
                and rsi_val > self.rsi_entry_min
                and vol_ok
                and atr_expanding):

            breakout_dist = (prev[dc_lower] - curr["close"]) / curr[atr_col]
            confidence = min(1.0, 0.5 + breakout_dist * 0.3)

            return Signal(
                pair=pair,
                direction="short",
                action="enter",
                confidence=confidence,
                reason=(f"Breakdown DC{self.dc_period} {prev[dc_lower]:.0f}, "
                        f"RSI={rsi_val:.0f}, vol={curr['volume']/curr[vol_sma_col]:.1f}x"),
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        return None

    def should_exit(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]

        if len(df) < self.chandelier_period + 2:
            return None

        curr = df.iloc[-1]
        rsi_col = f"rsi_{self.rsi_period}"
        rsi_val = curr.get(rsi_col, 50)
        if pd.isna(rsi_val):
            rsi_val = 50

        def _exit(direction, reason):
            self._last_exit_ts = int(curr["timestamp"])
            return Signal(
                pair=pair, direction=direction, action="exit",
                confidence=1.0, reason=reason,
                timestamp=int(curr["timestamp"]), timeframe=self.timeframes[0],
            )

        # Chandelier exit for longs
        chand_long = curr.get("chandelier_long")
        if not pd.isna(chand_long) and curr["close"] < chand_long:
            return _exit("long", f"Chandelier stop {chand_long:.0f}")

        # Chandelier exit for shorts
        chand_short = curr.get("chandelier_short")
        if not pd.isna(chand_short) and curr["close"] > chand_short:
            return _exit("short", f"Chandelier stop {chand_short:.0f}")

        # RSI extreme exits
        if rsi_val > self.rsi_exit_long:
            return _exit("long", f"RSI overbought {rsi_val:.0f}")
        if rsi_val < self.rsi_exit_short:
            return _exit("short", f"RSI oversold {rsi_val:.0f}")

        # Trend reversal exit
        fast_col = f"ema_{self.trend_fast_ema}"
        slow_col = f"ema_{self.trend_slow_ema}"
        if not pd.isna(curr.get(fast_col)) and not pd.isna(curr.get(slow_col)):
            if curr[fast_col] < curr[slow_col]:
                return _exit("long", "Trend reversed (EMA cross down)")
            if curr[fast_col] > curr[slow_col]:
                return _exit("short", "Trend reversed (EMA cross up)")

        return None
