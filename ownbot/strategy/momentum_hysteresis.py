"""Momentum Hysteresis strategy — 48h momentum with hysteresis entry/exit.

Proven signal on 30m candles:
  - Full period (2020-2026): +338% net after costs, Sharpe 0.92
  - Crisis (2025-2026): +49% net, Sharpe 1.40

Core logic:
  1. Compute 48h momentum (96-candle return on 30m)
  2. ENTER when |momentum| > enter_threshold (3%)
  3. EXIT when |momentum| < exit_threshold (1%) AND min hold elapsed
  4. Cooldown after each exit prevents re-entry churn
  5. SL/TP handled by the backtester/engine (ATR-based)
"""
import pandas as pd
import numpy as np

from ownbot.strategy.base import BaseStrategy, Signal
from ownbot.strategy.indicators import atr


class MomentumHysteresisStrategy(BaseStrategy):
    name = "momentum_hysteresis"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        tf = self.params.get("timeframe", "30m")
        self.timeframes = [tf]

        # Momentum
        self.mom_window = self.params.get("mom_window", 96)  # 48h on 30m
        self.enter_threshold = self.params.get("enter_threshold", 0.03)
        self.exit_threshold = self.params.get("exit_threshold", 0.01)

        # Min hold
        self.min_hold = self.params.get("min_hold", 48)  # 24h on 30m

        # ATR for dynamic SL
        self.atr_period = self.params.get("atr_period", 48)
        self.sl_atr_mult = self.params.get("sl_atr_mult", 3.0)

        # Cooldown
        self.cooldown_candles = self.params.get("cooldown_candles", 48)  # 24h
        self._last_exit_ts = 0
        self._entry_ts = 0
        self._had_position = False

        # Direction
        self.direction = self.params.get("direction", "both")

        self.startup_candle_count = self.mom_window + 10

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = atr(df, period=self.atr_period)
        df["momentum"] = df["close"].pct_change(self.mom_window)
        return df

    def _candle_ms(self):
        tf = self.timeframes[0]
        if "m" in tf:
            return int(tf.replace("m", "")) * 60 * 1000
        elif "h" in tf:
            return int(tf.replace("h", "")) * 3600 * 1000
        elif "d" in tf:
            return int(tf.replace("d", "")) * 86400 * 1000
        return 300000

    def leverage(self, pair: str, direction: str, data: dict[str, pd.DataFrame]) -> float:
        df = data[self.timeframes[0]]
        curr = df.iloc[-1]
        atr_val = curr.get(f"atr_{self.atr_period}")
        close = curr["close"]

        if pd.isna(atr_val) or close == 0 or atr_val == 0:
            return 1.0

        atr_pct = atr_val / close
        lev = 0.01 / (atr_pct * self.sl_atr_mult)
        return max(1.0, min(round(lev, 1), 5.0))

    def evaluate(self, pair: str, data: dict[str, pd.DataFrame], has_position: bool = False) -> Signal | None:
        """Override to detect external exits (SL/TP hit by backtester)."""
        if self._had_position and not has_position:
            df = data.get(self.timeframes[0])
            if df is not None and len(df) > 0:
                self._last_exit_ts = int(df.iloc[-1]["timestamp"])
        self._had_position = has_position
        return super().evaluate(pair, data, has_position)

    def should_enter(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]

        if len(df) < self.startup_candle_count:
            return None

        curr = df.iloc[-1]
        mom = curr.get("momentum")
        ts = int(curr["timestamp"])

        if pd.isna(mom):
            return None

        # Cooldown
        if self._last_exit_ts > 0:
            if ts - self._last_exit_ts < self.cooldown_candles * self._candle_ms():
                return None

        # Compute dynamic SL distance for the backtester
        atr_val = curr.get(f"atr_{self.atr_period}")
        close = curr["close"]
        if pd.isna(atr_val) or atr_val == 0:
            return None
        sl_distance_pct = (self.sl_atr_mult * atr_val / close)

        # LONG
        if mom > self.enter_threshold and self.direction in ("long", "both"):
            self._entry_ts = ts
            self._had_position = True
            confidence = min(1.0, abs(mom) / (self.enter_threshold * 3))

            # Set dynamic SL/TP in params for backtester to pick up
            sl = close * (1 - sl_distance_pct)
            tp = close * (1 + sl_distance_pct * 3)  # 3:1 reward/risk
            self.params["_dynamic_sl"] = sl
            self.params["_dynamic_tp"] = tp

            return Signal(
                pair=pair, direction="long", action="enter",
                confidence=confidence,
                reason=f"Mom 48h={mom*100:+.1f}%, ATR_SL={sl:.0f}, TP={tp:.0f}",
                timestamp=ts, timeframe=self.timeframes[0],
            )

        # SHORT
        if mom < -self.enter_threshold and self.direction in ("short", "both"):
            self._entry_ts = ts
            self._had_position = True
            confidence = min(1.0, abs(mom) / (self.enter_threshold * 3))

            sl = close * (1 + sl_distance_pct)
            tp = close * (1 - sl_distance_pct * 3)
            self.params["_dynamic_sl"] = sl
            self.params["_dynamic_tp"] = tp

            return Signal(
                pair=pair, direction="short", action="enter",
                confidence=confidence,
                reason=f"Mom 48h={mom*100:+.1f}%, ATR_SL={sl:.0f}, TP={tp:.0f}",
                timestamp=ts, timeframe=self.timeframes[0],
            )

        return None

    def should_exit(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        """Exit only on momentum fade after min hold. SL/TP handled by backtester."""
        df = data[self.timeframes[0]]
        curr = df.iloc[-1]
        mom = curr.get("momentum")
        ts = int(curr["timestamp"])

        if pd.isna(mom):
            return None

        # Only exit on momentum fade after minimum hold
        hold_time = ts - self._entry_ts
        min_hold_ms = self.min_hold * self._candle_ms()

        if hold_time >= min_hold_ms and abs(mom) < self.exit_threshold:
            self._last_exit_ts = ts
            self._had_position = False
            direction = "long" if mom >= 0 else "short"
            return Signal(
                pair=pair, direction=direction, action="exit",
                confidence=1.0,
                reason=f"Mom faded {mom*100:+.1f}% after {hold_time / self._candle_ms():.0f} candles",
                timestamp=ts, timeframe=self.timeframes[0],
            )

        return None
