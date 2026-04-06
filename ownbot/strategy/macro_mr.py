"""Macro Momentum + Micro Mean Reversion strategy.

Designed for crisis/bear markets (2025-2026 ETH).

Core logic:
  1. MACRO BIAS (48h momentum): determines long/short/flat
     - 48h return < -threshold → short bias
     - 48h return > +threshold → long bias
     - Otherwise → flat (no trades)

  2. MICRO ENTRY (30min mean reversion): times the entry
     - In short bias: enter short when price bounces UP over 30min
       (fade the bounce = enter at better price)
     - In long bias: enter long when price dips DOWN over 30min
     - Requires the bounce/dip to exceed a minimum size (ATR-scaled)

  3. EXIT:
     - ATR trailing stop (ride the trend)
     - Take profit at N × ATR
     - Exit if macro bias flips

  4. FILTERS:
     - Minimum time between trades (cooldown)
     - ATR-based position sizing
"""
import pandas as pd
import numpy as np

from ownbot.strategy.base import BaseStrategy, Signal
from ownbot.strategy.indicators import ema, rsi, atr


class MacroMRStrategy(BaseStrategy):
    name = "macro_mr"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        tf = self.params.get("timeframe", "5m")
        self.timeframes = [tf]

        # Macro momentum
        self.macro_window = self.params.get("macro_window", 576)  # 48h in 5m candles
        self.macro_threshold = self.params.get("macro_threshold", 0.02)  # 2% min move

        # Micro mean reversion (entry timing)
        self.mr_window = self.params.get("mr_window", 6)  # 30min in 5m candles
        self.mr_min_atr = self.params.get("mr_min_atr", 0.5)  # bounce must be > 0.5 × ATR

        # ATR for stops and sizing
        self.atr_period = self.params.get("atr_period", 48)  # 4h ATR

        # Exit
        self.trailing_atr_mult = self.params.get("trailing_atr_mult", 3.5)
        self.trailing_period = self.params.get("trailing_period", 96)  # 8h lookback
        self.tp_atr_mult = self.params.get("tp_atr_mult", 8.0)  # TP at 8 × ATR

        # Cooldown
        self.cooldown_candles = self.params.get("cooldown_candles", 24)  # 2h cooldown
        self._last_exit_ts = 0

        # Startup
        self.startup_candle_count = self.macro_window + 50

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = atr(df, period=self.atr_period)

        # Macro momentum: 48h return
        df["macro_ret"] = df["close"].pct_change(self.macro_window)

        # Micro mean reversion: 30min return
        df["mr_ret"] = df["close"].pct_change(self.mr_window)

        # Trailing stop levels
        atr_col = f"atr_{self.atr_period}"
        df["trail_long"] = (
            df["high"].rolling(self.trailing_period).max()
            - self.trailing_atr_mult * df[atr_col]
        )
        df["trail_short"] = (
            df["low"].rolling(self.trailing_period).min()
            + self.trailing_atr_mult * df[atr_col]
        )

        return df

    def leverage(self, pair: str, direction: str, data: dict[str, pd.DataFrame]) -> float:
        df = data[self.timeframes[0]]
        curr = df.iloc[-1]
        atr_val = curr.get(f"atr_{self.atr_period}")
        close = curr["close"]

        if pd.isna(atr_val) or close == 0 or atr_val == 0:
            return 1.0

        atr_pct = atr_val / close
        lev = 0.01 / (atr_pct * self.trailing_atr_mult)
        return max(1.0, min(round(lev, 1), 5.0))

    def should_enter(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]

        if len(df) < self.startup_candle_count:
            return None

        curr = df.iloc[-1]
        atr_col = f"atr_{self.atr_period}"

        # Check NaN
        if any(pd.isna(curr.get(c)) for c in ["macro_ret", "mr_ret", atr_col]):
            return None

        # Cooldown
        if self._last_exit_ts > 0:
            candle_ms = 5 * 60 * 1000
            if curr["timestamp"] - self._last_exit_ts < self.cooldown_candles * candle_ms:
                return None

        macro_ret = curr["macro_ret"]
        mr_ret = curr["mr_ret"]
        atr_val = curr[atr_col]
        close = curr["close"]

        # Min bounce/dip size in ATR terms
        mr_size = abs(mr_ret * close) / atr_val if atr_val > 0 else 0

        # SHORT: macro is bearish + price bounced up (mean reversion entry)
        if macro_ret < -self.macro_threshold and mr_ret > 0 and mr_size > self.mr_min_atr:
            confidence = min(1.0, abs(macro_ret) / (self.macro_threshold * 3))
            return Signal(
                pair=pair,
                direction="short",
                action="enter",
                confidence=confidence,
                reason=(f"Macro {macro_ret*100:+.1f}% (48h), "
                        f"bounce +{mr_ret*100:.2f}% (30m), "
                        f"size={mr_size:.1f}ATR"),
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        # LONG: macro is bullish + price dipped down (mean reversion entry)
        if macro_ret > self.macro_threshold and mr_ret < 0 and mr_size > self.mr_min_atr:
            confidence = min(1.0, abs(macro_ret) / (self.macro_threshold * 3))
            return Signal(
                pair=pair,
                direction="long",
                action="enter",
                confidence=confidence,
                reason=(f"Macro {macro_ret*100:+.1f}% (48h), "
                        f"dip {mr_ret*100:.2f}% (30m), "
                        f"size={mr_size:.1f}ATR"),
                timestamp=int(curr["timestamp"]),
                timeframe=self.timeframes[0],
            )

        return None

    def should_exit(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]

        if len(df) < self.trailing_period + 2:
            return None

        curr = df.iloc[-1]

        def _exit(direction, reason):
            self._last_exit_ts = int(curr["timestamp"])
            return Signal(
                pair=pair, direction=direction, action="exit",
                confidence=1.0, reason=reason,
                timestamp=int(curr["timestamp"]), timeframe=self.timeframes[0],
            )

        # Trailing stop
        trail_long = curr.get("trail_long")
        if not pd.isna(trail_long) and curr["close"] < trail_long:
            return _exit("long", f"Trailing stop {trail_long:.0f}")

        trail_short = curr.get("trail_short")
        if not pd.isna(trail_short) and curr["close"] > trail_short:
            return _exit("short", f"Trailing stop {trail_short:.0f}")

        # Macro bias flip — exit if trend reversed
        macro_ret = curr.get("macro_ret")
        if not pd.isna(macro_ret):
            if macro_ret > 0:
                return _exit("short", f"Macro flipped bullish ({macro_ret*100:+.1f}%)")
            if macro_ret < 0:
                return _exit("long", f"Macro flipped bearish ({macro_ret*100:+.1f}%)")

        return None
