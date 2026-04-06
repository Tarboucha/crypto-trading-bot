"""RL Conviction strategy — LSTM agent outputting conviction signal.

The trained RecurrentPPO agent receives 27 features (11 MC + 9 enriched +
5 momentum + 2 signal state) and outputs conviction a ∈ [-1, +1].

sign(a)  = direction (long/short)
|a|      = conviction (how confident)
|a| < ε  = flat (no opinion)

Leverage is a fixed config parameter. Position size is scaled by conviction
and Kronos uncertainty. Entry/exit use hysteresis thresholds.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ownbot.strategy.base import BaseStrategy, Signal
from ownbot.strategy.feature_builder import FeatureBuilder
from ownbot.strategy.signal_interpreter import SignalInterpreter

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _prepare_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 'amount' column Kronos expects (volume × close)."""
    out = df.copy()
    if "amount" not in out.columns:
        out["amount"] = out["volume"] * out["close"]
    return out


class RLConvictionStrategy(BaseStrategy):
    """RL conviction agent with Kronos MC features and LSTM memory.

    The agent outputs conviction ∈ [-1, +1].
    Entry when |conviction| > entry_threshold (default 0.6).
    Exit when |conviction| < exit_threshold (default 0.3) or direction flips.
    Leverage is fixed. Size scales with conviction + uncertainty.
    """

    name = "rl_conviction"
    startup_candle_count = 512

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.timeframes = [self.params.get("timeframe", "5m")]

        # Model paths
        self._model_path = self.params.get(
            "model_path",
            str(PROJECT_ROOT / "data/ml/rl/models/conviction_lstm128x1_mlp64_linear/best_model/best_model.zip"),
        )
        tokenizer_path = self.params.get(
            "tokenizer_path",
            str(PROJECT_ROOT / "data/ml/kronos/finetuned/eth_5m/tokenizer/best_model"),
        )
        predictor_path = self.params.get(
            "predictor_path",
            str(PROJECT_ROOT / "data/ml/kronos/finetuned/eth_5m/predictor/best_model"),
        )
        device = self.params.get("device", "cuda")

        # Execution params
        self._leverage = self.params.get("leverage", 3.0)
        self._hard_sl_pct = self.params.get("hard_sl_pct", 3.0) / 100
        self._hard_tp_pct = self.params.get("hard_tp_pct", 6.0) / 100
        self._use_dynamic_sl_tp = self.params.get("use_dynamic_sl_tp", True)
        self._sl_margin = self.params.get("sl_margin", 0.1)
        self._tp_margin = self.params.get("tp_margin", 0.01)

        # Signal interpreter
        self._interpreter = SignalInterpreter(
            entry_threshold=self.params.get("entry_threshold", 0.6),
            exit_threshold=self.params.get("exit_threshold", 0.3),
            epsilon=self.params.get("epsilon", 0.05),
            conviction_size_min=self.params.get("conviction_size_min", 0.5),
            conviction_size_max=self.params.get("conviction_size_max", 1.0),
            unc_scale_low=self.params.get("unc_scale_low", 0.002),
            unc_scale_high=self.params.get("unc_scale_high", 0.01),
            warmup_ticks=self.params.get("warmup_ticks", 10),
            warmup_discount=self.params.get("warmup_discount", 0.5),
        )

        # Feature builder
        self._features = FeatureBuilder()

        # Per-pair state
        self._position_direction: dict[str, str] = {}
        self._tick_count: int = 0

        # MC features cache (for dynamic SL/TP)
        self._last_mc_features: dict[str, float] | None = None
        # Last conviction (for logging)
        self._last_conviction: float = 0.0
        # Last size scalar (for engine to read)
        self._last_size_scalar: float = 1.0

        # LSTM hidden state (persists across ticks)
        self._lstm_states = None
        self._episode_start = np.ones((1,), dtype=bool)

        # Components
        from ownbot.strategy.rl_kronos_strategy import KronosMCSampler
        self._mc_sampler = KronosMCSampler(
            tokenizer_path=tokenizer_path,
            predictor_path=predictor_path,
            device=device,
            mc_samples=self.params.get("mc_samples", 5),
            temperature=self.params.get("mc_temperature", 1.0),
            top_p=self.params.get("mc_top_p", 0.9),
        )
        self._model = self._load_model()

        logger.info(
            "RLConvictionStrategy: leverage=%.1f entry=%.2f exit=%.2f model=%s",
            self._leverage, self._interpreter.entry_threshold,
            self._interpreter.exit_threshold, self._model_path,
        )

    def _load_model(self):
        """Load trained RecurrentPPO model."""
        try:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(self._model_path, device=self.params.get("device", "cuda"))
            logger.info("RecurrentPPO model loaded from %s", self._model_path)
            return model
        except Exception as e:
            logger.error("Failed to load RecurrentPPO model: %s", e)
            return None

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _run_inference(self, pair: str, df: pd.DataFrame) -> float | None:
        """Full pipeline: candles → MC → features → LSTM → conviction.

        Returns conviction ∈ [-1, +1] or None on failure.
        """
        if self._model is None or not self._mc_sampler.loaded:
            return None

        if len(df) < 100:
            return None

        current_close = float(df.iloc[-1]["close"])

        # MC sampling
        candles = _prepare_candles(df)
        mc_features = self._mc_sampler.sample(candles)
        if mc_features is None:
            return None

        self._last_mc_features = mc_features
        self._features.update_mc(mc_features)

        # Need enough MC history for momentum features
        if not self._features.is_warm:
            self._tick_count += 1
            return None

        # Build observation from MC features + engine's candle DataFrame
        obs = self._features.build(mc_features, df)

        # LSTM inference with persistent hidden state
        action, self._lstm_states = self._model.predict(
            obs,
            state=self._lstm_states,
            episode_start=self._episode_start,
            deterministic=True,
        )
        self._episode_start = np.zeros((1,), dtype=bool)

        raw_conviction = float(np.clip(action[0], -1.0, 1.0))
        conviction = self._interpreter.effective_conviction(raw_conviction, self._tick_count)

        # Update feature builder state
        self._features.prev_action = raw_conviction
        self._last_conviction = conviction
        self._tick_count += 1

        # Compute size scalar for the engine
        sigma = mc_features.get("sigma_return", 0.0)
        self._last_size_scalar = self._interpreter.position_size_scalar(conviction, sigma)

        logger.debug(
            "[%s] conviction=%.2f (raw=%.2f) size_scalar=%.2f mc_mu=%.4f sigma=%.4f",
            pair, conviction, raw_conviction, self._last_size_scalar,
            mc_features["mu_return"], sigma,
        )

        return conviction

    def _compute_dynamic_sl_tp(
        self, direction: str, current_close: float
    ) -> tuple[float, float]:
        """Derive SL/TP from MC features, clamped by hard limits."""
        mc = self._last_mc_features
        if mc is None:
            return self._hard_sl_tp(direction, current_close)

        if direction == "long":
            dynamic_sl = current_close * (1 + mc["worst_mae_long"]) * (1 - self._sl_margin)
            dynamic_tp = current_close * (1 + mc["mu_opt_long"]) * (1 - self._tp_margin)
            hard_sl = current_close * (1 - self._hard_sl_pct)
            hard_tp = current_close * (1 + self._hard_tp_pct)
            sl = max(dynamic_sl, hard_sl)
            tp = min(dynamic_tp, hard_tp) if dynamic_tp > current_close else hard_tp
        else:
            dynamic_sl = current_close * (1 + mc["worst_mae_short"]) * (1 + self._sl_margin)
            dynamic_tp = current_close * (1 - mc["mu_opt_short"]) * (1 - self._tp_margin)
            hard_sl = current_close * (1 + self._hard_sl_pct)
            hard_tp = current_close * (1 - self._hard_tp_pct)
            sl = min(dynamic_sl, hard_sl)
            tp = max(dynamic_tp, hard_tp) if dynamic_tp < current_close else hard_tp

        return sl, tp

    def _hard_sl_tp(self, direction: str, current_close: float) -> tuple[float, float]:
        if direction == "long":
            return (current_close * (1 - self._hard_sl_pct),
                    current_close * (1 + self._hard_tp_pct))
        else:
            return (current_close * (1 + self._hard_sl_pct),
                    current_close * (1 - self._hard_tp_pct))

    def leverage(self, pair: str, direction: str, data: dict[str, pd.DataFrame]) -> float:
        """Fixed leverage from config."""
        return self._leverage

    def should_enter(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]
        conviction = self._run_inference(pair, df)
        if conviction is None:
            return None

        should, direction = self._interpreter.should_enter(conviction)
        if not should:
            return None

        current_close = float(df.iloc[-1]["close"])

        # Dynamic SL/TP
        if self._use_dynamic_sl_tp:
            sl, tp = self._compute_dynamic_sl_tp(direction, current_close)
        else:
            sl, tp = self._hard_sl_tp(direction, current_close)
        self.params["_dynamic_sl"] = sl
        self.params["_dynamic_tp"] = tp

        # Store position state
        self._position_direction[pair] = direction

        logger.info(
            "[%s] ENTRY %s: conviction=%+.2f size_scalar=%.2f lev=%.1fx sl=%.2f tp=%.2f",
            pair, direction.upper(), conviction, self._last_size_scalar,
            self._leverage, sl, tp,
        )

        return Signal(
            action="enter",
            direction=direction,
            pair=pair,
            confidence=abs(conviction),
            reason=f"RL conviction={conviction:+.2f} (size={self._last_size_scalar:.2f})",
        )

    def should_exit(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]
        conviction = self._run_inference(pair, df)
        if conviction is None:
            return None

        direction = self._position_direction.get(pair, "long")
        should, reason = self._interpreter.should_exit(conviction, direction)
        if not should:
            return None

        # Clean up state
        self._position_direction.pop(pair, None)
        self._features.prev_action = 0.0

        logger.info("[%s] EXIT %s: %s", pair, direction.upper(), reason)

        return Signal(
            action="exit",
            direction=direction,
            pair=pair,
            confidence=1.0,
            reason=f"RL exit: {reason}",
        )

    def reset_lstm(self):
        """Reset LSTM hidden state (call on restart or long gap)."""
        self._lstm_states = None
        self._episode_start = np.ones((1,), dtype=bool)
        self._tick_count = 0
        self._features.reset()
        logger.info("LSTM state reset — warmup period active for %d ticks",
                     self._interpreter.warmup_ticks)
