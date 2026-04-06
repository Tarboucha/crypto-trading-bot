"""Build the 27-feature observation for the v3 conviction RL agent (live).

Mirrors the feature engineering in mllab/rl/trading_env_v3.py.
Candle-derived features (returns, volatility) are computed from the DataFrame
the engine already provides (512 candles). Only MC history needs a rolling
buffer since the engine doesn't track it across ticks.

Feature groups:
  11 MC features       — from KronosMCSampler
   9 enriched features — returns, volatility, MC context from engine's candle DF
   5 momentum features — MC prediction deltas and streaks
   2 signal state      — previous action, running mean |return|
  --
  27 total
"""
from collections import deque

import numpy as np
import pandas as pd

MC_FEATURE_COLS = [
    "p_long", "p_short",
    "mu_return", "sigma_return",
    "mu_opt_long", "mu_opt_short",
    "worst_mae_long", "worst_mae_short",
    "p_sl_long", "p_sl_short",
    "avg_agreement",
]


class FeatureBuilder:
    """Builds 27-dim observations for the LSTM agent.

    Candle features are computed from the DataFrame the engine provides each tick.
    MC features require a small rolling buffer for streak/delta/trend tracking.
    """

    def __init__(self):
        # MC history (for streak, delta, agreement trend)
        self._mu_history: deque[float] = deque(maxlen=48)
        self._sigma_history: deque[float] = deque(maxlen=48)
        self._agreement_history: deque[float] = deque(maxlen=6)

        # Running mean of |return| for signal state
        self._abs_return_sum: float = 0.0
        self._abs_return_count: int = 0

        # Previous action (signal state)
        self.prev_action: float = 0.0

        # MC streak tracking
        self._streak: int = 0
        self._last_mu_sign: float = 0.0

    @property
    def is_warm(self) -> bool:
        """True once we have enough MC history for momentum features."""
        return len(self._mu_history) >= 2

    def update_mc(self, mc_features: dict[str, float]):
        """Call each tick with the latest MC features."""
        mu = mc_features["mu_return"]
        sigma = mc_features["sigma_return"]
        agreement = mc_features["avg_agreement"]

        # Streak
        mu_sign = np.sign(mu)
        if mu_sign == self._last_mu_sign and mu != 0:
            self._streak += 1
        else:
            self._streak = 0
        self._last_mu_sign = mu_sign

        self._mu_history.append(mu)
        self._sigma_history.append(sigma)
        self._agreement_history.append(agreement)

    def build(self, mc_features: dict[str, float], df: pd.DataFrame) -> np.ndarray:
        """Build 27-feature observation.

        Args:
            mc_features: 11 MC features from KronosMCSampler
            df: OHLCV DataFrame from engine (512+ candles)
        """
        closes = df["close"].values
        n = len(closes)

        # ── 11 MC features ──
        mc = np.array([mc_features[col] for col in MC_FEATURE_COLS], dtype=np.float32)

        # ── 9 enriched features (from engine's candle DF) ──

        # 1-step return
        ret_1 = (closes[-1] - closes[-2]) / (closes[-2] + 1e-10) if n >= 2 else 0.0

        # Update running mean |return|
        if n >= 2:
            self._abs_return_sum += abs(ret_1)
            self._abs_return_count += 1

        # 6-step return (~30min at subsample=6)
        ret_6 = (closes[-1] - closes[-7]) / (closes[-7] + 1e-10) if n >= 7 else 0.0

        # 24-step return (~2h)
        ret_24 = (closes[-1] - closes[-25]) / (closes[-25] + 1e-10) if n >= 25 else 0.0

        # Rolling volatility from candle returns
        if n >= 13:
            recent_rets_12 = np.diff(closes[-13:]) / (closes[-13:-1] + 1e-10)
            vol_12 = float(np.std(recent_rets_12))
        else:
            vol_12 = 0.0

        if n >= 49:
            recent_rets_48 = np.diff(closes[-49:]) / (closes[-49:-1] + 1e-10)
            vol_48 = float(np.std(recent_rets_48))
        else:
            vol_48 = vol_12

        # Vol ratio
        vol_ratio = vol_12 / vol_48 if vol_48 > 1e-10 else 0.0

        # MC streak (normalized, capped at 20)
        mc_streak = min(self._streak / 20.0, 1.0)

        # Agreement trend (rolling mean of last 6)
        agreement_trend = float(np.mean(list(self._agreement_history))) if self._agreement_history else 0.0

        mu = mc_features["mu_return"]

        enriched = np.array([
            ret_1 * 100,
            ret_6 * 100,
            ret_24 * 100,
            vol_12 * 100,
            vol_48 * 100,
            vol_ratio,
            mc_streak,
            agreement_trend,
            mu * 1000,
        ], dtype=np.float32)

        # ── 5 momentum features ──
        mu_list = list(self._mu_history)
        sigma_list = list(self._sigma_history)

        delta_mu = mu_list[-1] - mu_list[-2] if len(mu_list) >= 2 else 0.0
        delta_sigma = sigma_list[-1] - sigma_list[-2] if len(sigma_list) >= 2 else 0.0

        momentum = np.array([
            delta_mu * 10000,
            delta_sigma * 10000,
            delta_mu * np.sign(mu) * 10000,       # aligned delta
            mc_streak * np.sign(mu),                # signed streak
            mu * mc_features["avg_agreement"] * 1000,  # mu * agreement
        ], dtype=np.float32)

        # ── 2 signal state features ──
        mean_abs_ret = (self._abs_return_sum / self._abs_return_count
                        if self._abs_return_count > 0 else 0.0)

        signal = np.array([
            self.prev_action,
            mean_abs_ret * 1000,
        ], dtype=np.float32)

        return np.concatenate([mc, enriched, momentum, signal]).astype(np.float32)

    def reset(self):
        """Reset all state (on strategy restart)."""
        self._mu_history.clear()
        self._sigma_history.clear()
        self._agreement_history.clear()
        self._abs_return_sum = 0.0
        self._abs_return_count = 0
        self.prev_action = 0.0
        self._streak = 0
        self._last_mu_sign = 0.0
