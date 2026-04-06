"""RL Kronos strategy — SAC agent with Kronos MC features.

The trained SAC agent receives 16 features (11 MC + 5 position state)
and outputs a continuous signed exposure in [-3, +3].
|exposure| > dead_zone → entry/hold with leverage = |exposure|.
|exposure| ≤ dead_zone → stay flat or exit.
Leverage is locked at entry — no mid-position rebalancing.
"""
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ownbot.strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# MC feature names (must match training env / precompute_mc_features.py)
MC_FEATURE_COLS = [
    "p_long", "p_short",
    "mu_return", "sigma_return",
    "mu_opt_long", "mu_opt_short",
    "worst_mae_long", "worst_mae_short",
    "p_sl_long", "p_sl_short",
    "avg_agreement",
]

# Cost/funding constants (match training env)
CANDLES_PER_8H = 96


def _prepare_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 'amount' column Kronos expects (volume × close)."""
    out = df.copy()
    if "amount" not in out.columns:
        out["amount"] = out["volume"] * out["close"]
    return out


class KronosMCSampler:
    """Loads Kronos model, runs MC sampling, extracts features."""

    def __init__(self, tokenizer_path: str, predictor_path: str,
                 device: str = "cuda", mc_samples: int = 5,
                 temperature: float = 1.0, top_p: float = 0.9):
        self.mc_samples = mc_samples
        self.temperature = temperature
        self.top_p = top_p
        self.loaded = False

        kronos_path = str(PROJECT_ROOT / "third_party" / "kronos")
        if kronos_path not in sys.path:
            sys.path.insert(0, kronos_path)

        try:
            from model import Kronos, KronosTokenizer, KronosPredictor

            dev = torch.device(device if torch.cuda.is_available() else "cpu")
            tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(dev).eval()
            model = Kronos.from_pretrained(predictor_path).to(dev).eval()
            self.predictor = KronosPredictor(model, tokenizer, max_context=512)
            self.loaded = True
            logger.info("Kronos MC sampler loaded (device=%s, samples=%d)", dev, mc_samples)
        except Exception as e:
            logger.error("Failed to load Kronos: %s", e)

    def sample(self, df: pd.DataFrame) -> dict[str, float] | None:
        """Run MC sampling on candle DataFrame, return 11 MC features.

        Args:
            df: OHLCV DataFrame with 'amount' column (use _prepare_candles).
        """
        if not self.loaded:
            return None

        try:
            if "timestamp" in df.columns:
                x_timestamps = pd.to_datetime(df["timestamp"], unit="ms").reset_index(drop=True)
            else:
                x_timestamps = pd.Series(pd.date_range(
                    end=pd.Timestamp.now(), periods=len(df), freq="5min",
                ))

            if len(x_timestamps) >= 2:
                freq = x_timestamps.iloc[-1] - x_timestamps.iloc[-2]
            else:
                freq = pd.Timedelta(minutes=5)

            y_timestamps = pd.Series(pd.date_range(
                start=x_timestamps.iloc[-1] + freq, periods=10, freq=freq,
            ))

            x_df = df[["open", "close", "high", "low", "volume", "amount"]]

            # Sample N paths
            paths = []
            for _ in range(self.mc_samples):
                pred_df = self.predictor.predict(
                    df=x_df, x_timestamp=x_timestamps, y_timestamp=y_timestamps,
                    pred_len=10, T=self.temperature, top_p=self.top_p,
                    sample_count=1, verbose=False,
                )
                paths.append(pred_df)

            return self._analyze_paths(paths, float(df["close"].iloc[-1]))
        except Exception as e:
            logger.warning("Kronos MC sampling failed: %s", e)
            return None

    def _analyze_paths(self, paths: list, current_close: float) -> dict[str, float]:
        """Extract 11 MC features from sampled paths."""
        final_returns = []
        optimal_long_returns = []
        optimal_short_returns = []
        mae_longs = []
        mae_shorts = []

        for path_df in paths:
            closes = path_df["close"].values
            highs = path_df["high"].values
            lows = path_df["low"].values

            final_ret = (closes[-1] - current_close) / current_close
            opt_long = (max(closes) - current_close) / current_close
            opt_short = (current_close - min(closes)) / current_close
            mae_long = (min(lows) - current_close) / current_close
            mae_short = (max(highs) - current_close) / current_close

            final_returns.append(final_ret)
            optimal_long_returns.append(opt_long)
            optimal_short_returns.append(opt_short)
            mae_longs.append(mae_long)
            mae_shorts.append(mae_short)

        final_returns = np.array(final_returns)

        # Agreement: matches precompute_mc_features.py exactly
        # abs(mean(sign(final_returns))) — how much paths agree on direction
        avg_agreement = float(abs(np.mean(np.sign(final_returns))))

        return {
            "p_long": float(np.mean(final_returns > 0)),
            "p_short": float(np.mean(final_returns < 0)),
            "mu_return": float(np.mean(final_returns)),
            "sigma_return": float(np.std(final_returns)),
            "mu_opt_long": float(np.mean(optimal_long_returns)),
            "mu_opt_short": float(np.mean(optimal_short_returns)),
            "worst_mae_long": float(min(mae_longs)),
            "worst_mae_short": float(max(mae_shorts)),
            "p_sl_long": float(np.mean(np.array(mae_longs) < -0.02)),
            "p_sl_short": float(np.mean(np.array(mae_shorts) > 0.02)),
            "avg_agreement": avg_agreement,
        }


class RLKronosStrategy(BaseStrategy):
    """RL agent (SAC) with Kronos MC features.

    The agent outputs exposure ∈ [-3, +3].
    |exposure| > dead_zone → entry with leverage = |exposure|.
    |exposure| ≤ dead_zone → flat / exit.
    Leverage locked at entry — no mid-position rebalancing.
    """
    name = "rl_kronos"
    startup_candle_count = 512

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.timeframes = [self.params.get("timeframe", "5m")]

        # Paths
        self.model_path = self.params.get(
            "model_path",
            str(PROJECT_ROOT / "data/ml/rl/models/sac_trading/best_model/best_model.zip"),
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

        # MC params
        mc_samples = self.params.get("mc_samples", 5)
        temperature = self.params.get("temperature", 1.0)
        top_p = self.params.get("top_p", 0.9)

        # Risk params
        self.dead_zone = self.params.get("dead_zone", 1.0)
        self.max_leverage = self.params.get("max_leverage", 3.0)
        self.hard_stoploss = self.params.get("stoploss", -2.0) / 100
        self.hard_takeprofit = self.params.get("take_profit", 3.0) / 100
        self.use_dynamic_sl_tp = self.params.get("use_dynamic_sl_tp", True)
        self.sl_margin = self.params.get("sl_margin", 0.01)
        self.tp_margin = self.params.get("tp_margin", 0.01)

        # State
        self._last_target_exposure: float = 0.0
        self._position_direction: dict[str, str] = {}
        self._entry_price: dict[str, float] = {}
        self._entry_exposure: dict[str, float] = {}
        self._ticks_in_position: dict[str, int] = {}
        self._tick_count: int = 0
        self._funding_counter: int = 0

        # Recent returns buffer for volatility (last 13 values to match training env window)
        self._recent_returns: deque[float] = deque(maxlen=13)
        self._last_close: dict[str, float] = {}

        # MC features cache (last tick's features for dynamic SL/TP)
        self._last_mc_features: dict[str, float] | None = None

        # Components
        self.mc_sampler = KronosMCSampler(
            tokenizer_path=tokenizer_path, predictor_path=predictor_path,
            device=device, mc_samples=mc_samples,
            temperature=temperature, top_p=top_p,
        )
        self.model = self._load_rl_model()

        logger.info(
            "RLKronosStrategy: dead_zone=%.1f max_leverage=%.1f mc_samples=%d device=%s",
            self.dead_zone, self.max_leverage, mc_samples, device,
        )

    def _load_rl_model(self):
        """Load trained SAC model."""
        try:
            from stable_baselines3 import SAC
            model = SAC.load(self.model_path, device=self.params.get("device", "cuda"))
            logger.info("SAC model loaded from %s", self.model_path)
            return model
        except Exception as e:
            logger.error("Failed to load SAC model: %s", e)
            return None

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _build_observation(self, pair: str, mc_features: dict[str, float],
                           current_close: float) -> np.ndarray:
        """Build 16-feature observation matching training env."""
        # 11 MC features
        mc = np.array([mc_features[col] for col in MC_FEATURE_COLS], dtype=np.float32)

        # Position state (must match training env exactly)
        # In training, _exposure is the agent's latest action (updated every step),
        # not locked at entry. We use _last_target_exposure to match this.
        exposure = self._last_target_exposure if pair in self._position_direction else 0.0
        unrealized_pnl = 0.0
        time_in_pos = 0.0

        if pair in self._position_direction:
            entry_price = self._entry_price.get(pair, current_close)
            if entry_price > 0:
                price_return = (current_close - entry_price) / entry_price
                unrealized_pnl = exposure * price_return
            ticks = self._ticks_in_position.get(pair, 0)
            time_in_pos = min(ticks / 288, 1.0)  # normalized same as training env

        # Funding countdown
        funding_countdown = (CANDLES_PER_8H - self._funding_counter % CANDLES_PER_8H) / CANDLES_PER_8H

        # Recent volatility
        recent_vol = float(np.std(list(self._recent_returns))) if len(self._recent_returns) > 1 else 0.0

        position_features = np.array([
            exposure / self.max_leverage,
            unrealized_pnl,
            time_in_pos,
            funding_countdown,
            recent_vol * 100,
        ], dtype=np.float32)

        return np.concatenate([mc, position_features])

    def _run_inference(self, pair: str, df: pd.DataFrame) -> float | None:
        """Run full pipeline: candles → MC sampling → RL inference → exposure.

        The engine provides df with startup_candle_count (512) rows.
        """
        if self.model is None or not self.mc_sampler.loaded:
            return None

        if len(df) < 100:
            return None

        # Update recent returns for volatility
        current_close = float(df.iloc[-1]["close"])
        if pair in self._last_close and self._last_close[pair] > 0:
            ret = (current_close - self._last_close[pair]) / self._last_close[pair]
            self._recent_returns.append(ret)
        self._last_close[pair] = current_close

        # MC sampling — engine already provides 512 candles
        candles = _prepare_candles(df)
        mc_features = self.mc_sampler.sample(candles)
        if mc_features is None:
            return None

        self._last_mc_features = mc_features

        # Build observation and run SAC
        obs = self._build_observation(pair, mc_features, current_close)
        action, _ = self.model.predict(obs, deterministic=True)
        target_exposure = float(np.clip(action[0], -self.max_leverage, self.max_leverage))

        self._tick_count += 1
        self._funding_counter += 1

        logger.debug(
            "[%s] RL: exposure=%.2f mc_mu=%.4f mc_sigma=%.4f p_long=%.0f%%",
            pair, target_exposure, mc_features["mu_return"],
            mc_features["sigma_return"], mc_features["p_long"] * 100,
        )

        return target_exposure

    def _compute_dynamic_sl_tp(self, direction: str, current_close: float) -> tuple[float, float]:
        """Derive SL/TP from MC features, clamped by hard limits."""
        mc = self._last_mc_features
        if mc is None:
            return self._hard_sl_tp(direction, current_close)

        if direction == "long":
            # SL below entry: worst drawdown with margin pushing it further down
            dynamic_sl = current_close * (1 + mc["worst_mae_long"]) * (1 - self.sl_margin)
            # TP above entry: best predicted upside, slightly reduced by margin
            dynamic_tp = current_close * (1 + mc["mu_opt_long"]) * (1 - self.tp_margin)
            hard_sl = current_close * (1 + self.hard_stoploss)
            hard_tp = current_close * (1 + self.hard_takeprofit)
            sl = max(dynamic_sl, hard_sl)  # more protective = higher for long
            # TP must be above entry for long — if MC predicts down, use hard TP
            if dynamic_tp <= current_close:
                tp = hard_tp
            else:
                tp = min(dynamic_tp, hard_tp)
        else:
            # SL above entry: worst spike with margin pushing it further up
            dynamic_sl = current_close * (1 + mc["worst_mae_short"]) * (1 + self.sl_margin)
            # TP below entry: best predicted downside, slightly reduced by margin
            dynamic_tp = current_close * (1 - mc["mu_opt_short"]) * (1 - self.tp_margin)
            hard_sl = current_close * (1 - self.hard_stoploss)
            hard_tp = current_close * (1 - self.hard_takeprofit)
            sl = min(dynamic_sl, hard_sl)  # more protective = lower for short
            # TP must be below entry for short — if MC predicts up, use hard TP
            if dynamic_tp >= current_close:
                tp = hard_tp
            else:
                tp = max(dynamic_tp, hard_tp)

        return sl, tp

    def _hard_sl_tp(self, direction: str, current_close: float) -> tuple[float, float]:
        if direction == "long":
            return (current_close * (1 + self.hard_stoploss),
                    current_close * (1 + self.hard_takeprofit))
        else:
            return (current_close * (1 - self.hard_stoploss),
                    current_close * (1 - self.hard_takeprofit))

    def leverage(self, pair: str, direction: str, data: dict[str, pd.DataFrame]) -> float:
        """Return leverage = |exposure|. Only called when entering a position,
        so |exposure| > dead_zone is guaranteed."""
        return abs(self._last_target_exposure)

    def should_enter(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]
        exposure = self._run_inference(pair, df)
        if exposure is None:
            return None

        self._last_target_exposure = exposure
        abs_exp = abs(exposure)

        # Dead zone — not enough conviction
        if abs_exp <= self.dead_zone:
            return None

        direction = "long" if exposure > 0 else "short"
        current_close = float(df.iloc[-1]["close"])

        # Set dynamic SL/TP for the engine
        if self.use_dynamic_sl_tp:
            sl, tp = self._compute_dynamic_sl_tp(direction, current_close)
        else:
            sl, tp = self._hard_sl_tp(direction, current_close)
        self.params["_dynamic_sl"] = sl
        self.params["_dynamic_tp"] = tp

        # Track position state
        self._position_direction[pair] = direction
        self._entry_price[pair] = current_close
        self._entry_exposure[pair] = exposure
        self._ticks_in_position[pair] = 0

        confidence = min(abs_exp / self.max_leverage, 1.0)
        logger.info(
            "[%s] ENTRY %s: exposure=%.2f leverage=%.1fx sl=%.2f tp=%.2f",
            pair, direction.upper(), exposure, abs_exp, sl, tp,
        )

        return Signal(
            action="enter", direction=direction, pair=pair,
            confidence=confidence,
            reason=f"RL: exposure={exposure:+.2f} leverage={abs_exp:.1f}x",
        )

    def should_exit(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]

        exposure = self._run_inference(pair, df)
        if exposure is None:
            return None

        direction = self._position_direction.get(pair, "long")
        abs_exp = abs(exposure)

        # Exit if conviction dropped below dead zone
        if abs_exp <= self.dead_zone:
            return self._exit(pair, direction, f"conviction dropped: exposure={exposure:+.2f}")

        # Exit if direction flipped
        new_direction = "long" if exposure > 0 else "short"
        if new_direction != direction:
            return self._exit(pair, direction, f"direction flipped to {new_direction}: exposure={exposure:+.2f}")

        # Hold — leverage locked at entry, magnitude change ignored
        # Increment time in position AFTER obs (matches training env: increment at end of step)
        self._ticks_in_position[pair] = self._ticks_in_position.get(pair, 0) + 1
        return None

    def _exit(self, pair: str, direction: str, reason: str) -> Signal:
        self._position_direction.pop(pair, None)
        self._entry_price.pop(pair, None)
        self._entry_exposure.pop(pair, None)
        self._ticks_in_position.pop(pair, None)
        self._last_target_exposure = 0.0  # match training env: exposure=0 when flat
        logger.info("[%s] EXIT %s: %s", pair, direction.upper(), reason)
        return Signal(
            action="exit", direction=direction, pair=pair,
            confidence=1.0,
            reason=f"RL exit: {reason}",
        )
