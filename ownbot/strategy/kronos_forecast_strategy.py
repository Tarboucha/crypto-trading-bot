"""Kronos forecasting strategy — full path-aware decision framework.

Predicts next N candles, analyzes the entire predicted path, enters only
if profitable AND survivable. Continuously reassesses open positions.
"""
import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ownbot.strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


# --- Path Analysis ---

@dataclass
class PathAnalysis:
    """Metrics computed from the predicted candle path."""
    final_return: float = 0.0         # return at last predicted candle
    optimal_return: float = 0.0       # return at optimal exit candle
    optimal_exit_candle: int = 0      # candle index with best return
    path_low: float = 0.0            # min of all predicted lows
    path_high: float = 0.0           # max of all predicted highs
    max_drawdown: float = 0.0        # worst dip from current price
    max_upside: float = 0.0          # best peak from current price
    risk_reward: float = 0.0         # max_upside / |max_drawdown|
    trend_consistency: float = 0.0   # fraction of candles trending in same direction
    path_volatility: float = 0.0     # avg range / current price
    peak_candle: int = 0             # candle with highest high
    trough_candle: int = 0           # candle with lowest low
    direction: str = ""              # "long" or "short"
    predicted_closes: list = None    # all predicted close prices
    pred_candle_count: int = 0       # number of predicted candles


def analyze_path(current_close: float, pred_df: pd.DataFrame) -> PathAnalysis | None:
    """Compute all path metrics from predicted candles."""
    if pred_df is None or pred_df.empty or current_close == 0:
        return None

    closes = pred_df["close"].values.astype(float)
    highs = pred_df["high"].values.astype(float)
    lows = pred_df["low"].values.astype(float)

    path_high = float(np.max(highs))
    path_low = float(np.min(lows))
    peak_candle = int(np.argmax(highs))
    trough_candle = int(np.argmin(lows))

    final_return = (closes[-1] - current_close) / current_close
    max_drawdown = (path_low - current_close) / current_close
    max_upside = (path_high - current_close) / current_close

    # Optimal exit for long = candle with highest close
    best_long_idx = int(np.argmax(closes))
    best_long_return = (closes[best_long_idx] - current_close) / current_close

    # Optimal exit for short = candle with lowest close
    best_short_idx = int(np.argmin(closes))
    best_short_return = (current_close - closes[best_short_idx]) / current_close

    # Direction based on which side has better optimal return
    if best_long_return >= best_short_return:
        direction = "long"
        optimal_exit_candle = best_long_idx
        optimal_return = best_long_return
    else:
        direction = "short"
        optimal_exit_candle = best_short_idx
        optimal_return = best_short_return

    # Risk/reward
    if direction == "long":
        risk = abs(max_drawdown) if max_drawdown < 0 else 0.0001
        reward = max_upside if max_upside > 0 else 0.0
    else:
        risk = max_upside if max_upside > 0 else 0.0001
        reward = abs(max_drawdown) if max_drawdown < 0 else 0.0

    risk_reward = reward / risk if risk > 0 else 0.0

    # Trend consistency: fraction of candle closes that move in the predicted direction
    diffs = np.diff(closes)
    if direction == "long":
        trend_consistency = float(np.mean(diffs > 0))
    else:
        trend_consistency = float(np.mean(diffs < 0))

    # Path volatility
    ranges = highs - lows
    path_volatility = float(np.mean(ranges) / current_close)

    return PathAnalysis(
        final_return=final_return,
        optimal_return=optimal_return,
        optimal_exit_candle=optimal_exit_candle,
        path_low=path_low,
        path_high=path_high,
        max_drawdown=max_drawdown,
        max_upside=max_upside,
        risk_reward=risk_reward,
        trend_consistency=trend_consistency,
        path_volatility=path_volatility,
        peak_candle=peak_candle,
        trough_candle=trough_candle,
        direction=direction,
        predicted_closes=closes.tolist(),
        pred_candle_count=len(closes),
    )


# --- Candle Buffer ---

class CandleBuffer:
    """Rolling FIFO buffer of candles per pair."""

    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._buffers: dict[str, deque] = {}

    def init_from_df(self, pair: str, df: pd.DataFrame) -> None:
        self._buffers[pair] = deque(maxlen=self.max_size)
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for _, row in df[cols].tail(self.max_size).iterrows():
            d = row.to_dict()
            d["amount"] = d["volume"] * d["close"]
            self._buffers[pair].append(d)

    def append(self, pair: str, candle: dict) -> None:
        if pair not in self._buffers:
            self._buffers[pair] = deque(maxlen=self.max_size)
        if "amount" not in candle:
            candle["amount"] = candle.get("volume", 0) * candle.get("close", 0)
        self._buffers[pair].append(candle)

    def get_df(self, pair: str) -> pd.DataFrame:
        return pd.DataFrame(list(self._buffers[pair]))

    def size(self, pair: str) -> int:
        return len(self._buffers.get(pair, []))

    def is_ready(self, pair: str, min_candles: int = 50) -> bool:
        return self.size(pair) >= min_candles


# --- Kronos Predictor ---

class KronosForecastPredictor:
    """Loads Kronos model and tokenizer, predicts next candles."""

    def __init__(self, model_path: str, tokenizer_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.loaded = False

        kronos_path = str(Path(__file__).parent.parent.parent / "third_party" / "kronos")
        if kronos_path not in sys.path:
            sys.path.insert(0, kronos_path)

        try:
            from model import Kronos, KronosTokenizer, KronosPredictor

            logger.info("Loading Kronos tokenizer from %s...", tokenizer_path)
            tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)

            logger.info("Loading Kronos model from %s...", model_path)
            model = Kronos.from_pretrained(model_path)
            model = model.to(self.device)

            self.predictor = KronosPredictor(model, tokenizer, max_context=512)
            self.loaded = True
            logger.info("Kronos forecast predictor loaded (device=%s)", self.device)
        except Exception as e:
            logger.error("Failed to load Kronos: %s", e)

    def forecast(self, df: pd.DataFrame, pred_len: int = 10,
                 temperature: float = 1.0, top_p: float = 0.9) -> pd.DataFrame | None:
        if not self.loaded:
            return None

        try:
            if "timestamp" in df.columns:
                x_timestamp = pd.to_datetime(df["timestamp"], unit="ms").reset_index(drop=True)
            else:
                x_timestamp = pd.Series(pd.date_range(
                    end=pd.Timestamp.now(), periods=len(df), freq="5min",
                ))

            if len(x_timestamp) >= 2:
                freq = x_timestamp.iloc[-1] - x_timestamp.iloc[-2]
            else:
                freq = pd.Timedelta(minutes=5)

            y_timestamp = pd.Series(pd.date_range(
                start=x_timestamp.iloc[-1] + freq,
                periods=pred_len,
                freq=freq,
            ))

            pred_df = self.predictor.predict(
                df=df[["open", "high", "low", "close", "volume", "amount"]],
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=1,
                verbose=False,
            )
            return pred_df
        except Exception as e:
            logger.warning("Kronos forecast failed: %s", e)
            return None


# --- Strategy ---

class KronosForecastStrategy(BaseStrategy):
    """Path-aware forecasting strategy using Kronos.

    Entry: predicted path must be profitable, survivable, and have good risk/reward.
    Hold: continuously reassess — exit when predicted peak passed or path reverses.
    Exit: dynamic, based on optimal exit candle from latest prediction.
    """
    name = "kronos_forecast"
    startup_candle_count = 512

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.timeframes = [self.params.get("timeframe", "5m")]

        # Model
        project_root = Path(__file__).parent.parent.parent
        self.model_path = self.params.get(
            "pretrained_model",
            str(project_root / "data" / "ml" / "pretrained" / "Kronos-base"),
        )
        self.tokenizer_path = self.params.get(
            "pretrained_tokenizer",
            str(project_root / "data" / "ml" / "pretrained" / "Kronos-Tokenizer-base"),
        )
        self.device = self.params.get("device", "cuda")

        # Forecasting
        self.max_context = self.params.get("max_context", 512)
        self.pred_len = self.params.get("pred_len", 10)
        self.temperature = self.params.get("temperature", 1.0)

        # Entry conditions
        self.entry_threshold = self.params.get("entry_threshold", 0.003)
        self.min_risk_reward = self.params.get("min_risk_reward", 1.5)
        self.min_trend_consistency = self.params.get("min_trend_consistency", 0.0)

        # Exit conditions
        self.exit_threshold = self.params.get("exit_threshold", 0.0)
        self.exit_on_peak_passed = self.params.get("exit_on_peak_passed", True)
        self.reversal_candles = self.params.get("reversal_candles", 3)
        self.min_hold_rr = self.params.get("min_hold_rr", 0.5)
        self.max_hold_candles = self.params.get("max_hold_candles", 60)

        # Dynamic SL/TP
        self.use_dynamic_sl_tp = self.params.get("use_dynamic_sl_tp", True)
        self.sl_margin = self.params.get("sl_margin", 0.01)
        self.tp_margin = self.params.get("tp_margin", 0.01)
        self.hard_stoploss = self.params.get("stoploss", -2.0) / 100
        self.hard_takeprofit = self.params.get("take_profit", 3.0) / 100

        # State
        self._candles_in_position: dict[str, int] = {}
        self._position_direction: dict[str, str] = {}

        # Buffer and predictor
        self.buffer = CandleBuffer(max_size=self.max_context)
        self.predictor = KronosForecastPredictor(
            self.model_path, self.tokenizer_path, self.device,
        )

        logger.info(
            "KronosForecastStrategy: context=%d pred=%d entry=%.2f%% rr=%.1f device=%s",
            self.max_context, self.pred_len, self.entry_threshold * 100,
            self.min_risk_reward, self.device,
        )

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _get_path(self, pair: str, df: pd.DataFrame) -> PathAnalysis | None:
        """Update buffer, run Kronos forecast, analyze the predicted path."""
        if not self.buffer.is_ready(pair):
            self.buffer.init_from_df(pair, df)
            logger.info("[%s] Buffer initialized: %d candles", pair, self.buffer.size(pair))
        else:
            row = df.iloc[-1]
            self.buffer.append(pair, {
                "timestamp": int(row["timestamp"]) if "timestamp" in df.columns else 0,
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

        if not self.buffer.is_ready(pair):
            return None

        buffer_df = self.buffer.get_df(pair)
        pred = self.predictor.forecast(
            buffer_df, pred_len=self.pred_len, temperature=self.temperature,
        )
        if pred is None or pred.empty:
            return None

        current_close = float(df.iloc[-1]["close"])
        path = analyze_path(current_close, pred)

        if path:
            logger.debug(
                "[%s] Path: dir=%s final=%+.2f%% optimal=%+.2f%%@C%d dd=%+.2f%% up=%+.2f%% rr=%.1f trend=%.0f%%",
                pair, path.direction, path.final_return * 100,
                path.optimal_return * 100, path.optimal_exit_candle,
                path.max_drawdown * 100, path.max_upside * 100,
                path.risk_reward, path.trend_consistency * 100,
            )

        return path

    def _compute_dynamic_sl_tp(self, path: PathAnalysis, direction: str,
                                current_close: float) -> tuple[float, float]:
        """Derive SL/TP from predicted path, clamped by hard limits."""
        if direction == "long":
            dynamic_sl = path.path_low * (1 - self.sl_margin)
            dynamic_tp = path.path_high * (1 - self.tp_margin)
            hard_sl = current_close * (1 + self.hard_stoploss)
            hard_tp = current_close * (1 + self.hard_takeprofit)
            sl = max(dynamic_sl, hard_sl)
            tp = min(dynamic_tp, hard_tp)
        else:
            dynamic_sl = path.path_high * (1 + self.sl_margin)
            dynamic_tp = path.path_low * (1 + self.tp_margin)
            hard_sl = current_close * (1 - self.hard_stoploss)
            hard_tp = current_close * (1 - self.hard_takeprofit)
            sl = min(dynamic_sl, hard_sl)
            tp = max(dynamic_tp, hard_tp)

        return sl, tp

    def should_enter(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]
        path = self._get_path(pair, df)
        if path is None:
            return None

        direction = path.direction
        current_close = float(df.iloc[-1]["close"])

        # 1. Directional: meaningful predicted move
        if path.optimal_return < self.entry_threshold:
            return None

        # 2. Survivability: predicted path won't trigger stoploss
        if direction == "long" and path.max_drawdown < self.hard_stoploss:
            logger.debug("[%s] Skip LONG: predicted drawdown %.2f%% would hit SL",
                         pair, path.max_drawdown * 100)
            return None
        if direction == "short" and path.max_upside > abs(self.hard_stoploss):
            logger.debug("[%s] Skip SHORT: predicted upside %.2f%% would hit SL",
                         pair, path.max_upside * 100)
            return None

        # 3. Risk/reward
        if path.risk_reward < self.min_risk_reward:
            logger.debug("[%s] Skip: risk/reward %.1f < %.1f",
                         pair, path.risk_reward, self.min_risk_reward)
            return None

        # 4. Trend consistency (optional)
        if self.min_trend_consistency > 0 and path.trend_consistency < self.min_trend_consistency:
            return None

        # Compute dynamic SL/TP
        if self.use_dynamic_sl_tp:
            sl, tp = self._compute_dynamic_sl_tp(path, direction, current_close)
        else:
            if direction == "long":
                sl = current_close * (1 + self.hard_stoploss)
                tp = current_close * (1 + self.hard_takeprofit)
            else:
                sl = current_close * (1 - self.hard_stoploss)
                tp = current_close * (1 - self.hard_takeprofit)

        # Store position state
        self._candles_in_position[pair] = 0
        self._position_direction[pair] = direction

        # Pass SL/TP via strategy params so engine can use them
        self.params["_dynamic_sl"] = sl
        self.params["_dynamic_tp"] = tp

        logger.info(
            "[%s] ENTRY %s: optimal=%+.2f%%@C%d rr=%.1f trend=%.0f%% SL=%.2f TP=%.2f",
            pair, direction.upper(), path.optimal_return * 100,
            path.optimal_exit_candle, path.risk_reward,
            path.trend_consistency * 100, sl, tp,
        )

        return Signal(
            action="enter", direction=direction, pair=pair,
            confidence=path.optimal_return,
            reason=f"Kronos: {direction} {path.optimal_return:+.2%} optimal@C{path.optimal_exit_candle} rr={path.risk_reward:.1f}",
        )

    def should_exit(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]
        direction = self._position_direction.get(pair, "long")

        self._candles_in_position[pair] = self._candles_in_position.get(pair, 0) + 1
        candles_held = self._candles_in_position[pair]

        # Time-based exit
        if self.max_hold_candles > 0 and candles_held >= self.max_hold_candles:
            return self._exit(pair, direction, candles_held, "max hold reached")

        # Re-predict and reassess
        path = self._get_path(pair, df)
        if path is None:
            return None

        # 1. Predicted return flipped against our position
        if direction == "long" and path.final_return < self.exit_threshold:
            return self._exit(pair, direction, candles_held,
                              f"prediction flipped: {path.final_return:+.2%}")

        if direction == "short" and path.final_return > -self.exit_threshold:
            return self._exit(pair, direction, candles_held,
                              f"prediction flipped: {path.final_return:+.2%}")

        # 2. Predicted peak/trough already passed (optimal exit is NOW or behind us)
        if self.exit_on_peak_passed:
            if direction == "long" and path.peak_candle <= 1:
                return self._exit(pair, direction, candles_held,
                                  f"predicted peak passed (candle {path.peak_candle})")
            if direction == "short" and path.trough_candle <= 1:
                return self._exit(pair, direction, candles_held,
                                  f"predicted trough passed (candle {path.trough_candle})")

        # 3. Imminent reversal: next N candles all reverse
        if self.reversal_candles > 0 and path.predicted_closes:
            next_n = path.predicted_closes[:self.reversal_candles]
            if len(next_n) >= self.reversal_candles:
                diffs = [next_n[i+1] - next_n[i] for i in range(len(next_n) - 1)]
                if direction == "long" and all(d < 0 for d in diffs):
                    return self._exit(pair, direction, candles_held,
                                      f"imminent reversal: next {self.reversal_candles} candles down")
                if direction == "short" and all(d > 0 for d in diffs):
                    return self._exit(pair, direction, candles_held,
                                      f"imminent reversal: next {self.reversal_candles} candles up")

        # 4. Risk/reward deteriorated
        if path.risk_reward < self.min_hold_rr:
            return self._exit(pair, direction, candles_held,
                              f"rr deteriorated: {path.risk_reward:.1f} < {self.min_hold_rr}")

        return None

    def _exit(self, pair: str, direction: str, candles_held: int, reason: str) -> Signal:
        """Create exit signal and clean up state."""
        self._candles_in_position.pop(pair, None)
        self._position_direction.pop(pair, None)

        logger.info("[%s] EXIT %s: %s (held %d candles)", pair, direction.upper(), reason, candles_held)

        return Signal(
            action="exit", direction=direction, pair=pair,
            confidence=1.0,
            reason=f"Kronos exit: {reason} (held {candles_held} candles)",
        )
