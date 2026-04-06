"""Kronos-based trading strategy — inference only.

Loads a trained Kronos classifier and uses it to generate/filter signals.
Supports: standalone, smart exit (re-evaluate every candle), exit modes.
No training code here — that lives in mllab/.
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from ownbot.strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)

# Exit mode presets
EXIT_MODES = {
    "aggressive": 0.55,   # exit when win prob drops below 55%
    "normal": 0.50,       # exit when win prob drops below 50%
    "patient": 0.45,      # tolerate lower confidence before exiting
}


class KronosPredictor:
    """Lightweight Kronos model loader for inference."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.loaded = False
        self.model = None
        self.seq_len = 50

        path = Path(model_path)
        if not path.exists():
            logger.warning("Kronos model not found at %s — predictions disabled.", path)
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model_config = checkpoint["config"]
        self.seq_len = model_config.get("sequence_length", 50)

        # Load Kronos encoder
        import sys
        kronos_path = str(Path(__file__).parent.parent.parent / "third_party" / "kronos")
        if kronos_path not in sys.path:
            sys.path.insert(0, kronos_path)

        from mllab.training.kronos_trainer import load_kronos_encoder, KronosClassifier

        encoder, hidden_dim = load_kronos_encoder(
            model_config["model_name"],
            model_config["tokenizer_name"],
            self.device,
        )

        self.model = KronosClassifier(encoder, hidden_dim, num_classes=2, dropout=0.0)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True
        logger.info("Kronos model loaded from %s (seq_len=%d, device=%s)", path, self.seq_len, self.device)

    @torch.no_grad()
    def predict(self, candles: np.ndarray) -> float:
        """Predict win probability from a candle sequence.

        Args:
            candles: (seq_len, 6) array — OHLCVA normalized

        Returns:
            Win probability (0.0 to 1.0)
        """
        if not self.loaded:
            return 0.5

        x = torch.tensor(candles, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1].cpu())


def _prepare_sequence(df: pd.DataFrame, seq_len: int) -> np.ndarray | None:
    """Extract and normalize the last seq_len candles from a DataFrame."""
    if len(df) < seq_len:
        return None

    cols = ["open", "high", "low", "close", "volume"]
    chunk = df.tail(seq_len)

    seq = np.zeros((seq_len, 6))
    for i, col in enumerate(cols):
        seq[:, i] = chunk[col].values

    # Amount = volume * close
    seq[:, 5] = seq[:, 4] * seq[:, 3]

    # Normalize OHLC relative to first candle's close
    base_close = seq[0, 3]
    if base_close > 0:
        seq[:, :4] = seq[:, :4] / base_close
    # Normalize volume by mean
    vol_mean = seq[:, 4].mean()
    if vol_mean > 0:
        seq[:, 4] = seq[:, 4] / vol_mean
    # Normalize amount by mean
    amt_mean = seq[:, 5].mean()
    if amt_mean > 0:
        seq[:, 5] = seq[:, 5] / amt_mean

    return seq


def _detect_direction(df: pd.DataFrame, lookback: int = 5) -> str:
    """Detect trend direction from recent candles.

    Uses SMA slope over the last `lookback` candles.
    More robust than comparing just 2 candles.
    """
    closes = df["close"].tail(lookback).values
    if len(closes) < 2:
        return "long"

    slope = (closes[-1] - closes[0]) / closes[0]
    return "long" if slope > 0 else "short"


class KronosStrategy(BaseStrategy):
    """Trading strategy powered by Kronos model.

    Supports three modes:
    - Standalone: Kronos generates entry/exit signals on its own
    - Smart exit: re-evaluates every candle, exits when confidence drops
    - Configurable exit sensitivity (aggressive/normal/patient/custom)

    Config params (via config.toml [strategy.params] or --param):
        kronos_model_path:     path to fine-tuned .pt file
        device:                "cuda" or "cpu"
        entry_threshold:       min win probability to enter (default 0.58)
        exit_mode:             "aggressive" | "normal" | "patient" | "custom"
        exit_threshold:        custom exit threshold (only if exit_mode="custom")
        max_hold_candles:      force exit after N candles (0 = disabled)
        direction_lookback:    candles to determine trend direction (default 5)
        stoploss:              stoploss % (default -1.5)
        take_profit:           take profit % (default 1.5)
    """
    name = "kronos"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.timeframes = [self.params.get("timeframe", "5m")]

        # Model
        self.model_path = self.params.get(
            "kronos_model_path",
            str(Path(__file__).parent.parent.parent / "data" / "ml" / "models" / "kronos_signal_filter_final.pt"),
        )
        self.device = self.params.get("device", "cpu")
        self.predictor = KronosPredictor(self.model_path, self.device)
        self.seq_len = self.predictor.seq_len

        # Entry
        self.entry_threshold = self.params.get("entry_threshold", 0.58)

        # Exit
        self.exit_mode = self.params.get("exit_mode", "normal")
        if self.exit_mode == "custom":
            self.exit_threshold = self.params.get("exit_threshold", 0.50)
        else:
            self.exit_threshold = EXIT_MODES.get(self.exit_mode, 0.50)

        # Time-based exit
        self.max_hold_candles = self.params.get("max_hold_candles", 0)
        self._candles_in_position: dict[str, int] = {}  # pair → candles held

        # Direction detection
        self.direction_lookback = self.params.get("direction_lookback", 5)

        logger.info(
            "KronosStrategy: entry=%.0f%% exit=%s(%.0f%%) max_hold=%d device=%s",
            self.entry_threshold * 100, self.exit_mode,
            self.exit_threshold * 100, self.max_hold_candles, self.device,
        )

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Kronos reads raw candles — no indicators needed
        return df

    def should_enter(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]
        seq = _prepare_sequence(df, self.seq_len)
        if seq is None:
            return None

        prob = self.predictor.predict(seq)

        if prob >= self.entry_threshold:
            direction = _detect_direction(df, self.direction_lookback)

            # Reset hold counter on entry
            self._candles_in_position[pair] = 0

            return Signal(
                action="enter",
                direction=direction,
                pair=pair,
                confidence=prob,
                reason=f"Kronos entry ({prob:.0%} win prob)",
                timestamp=0,
            )

        return None

    def should_exit(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        df = data[self.timeframes[0]]

        # Track how long we've been in this position
        self._candles_in_position[pair] = self._candles_in_position.get(pair, 0) + 1
        candles_held = self._candles_in_position[pair]

        # 1. Time-based exit
        if self.max_hold_candles > 0 and candles_held >= self.max_hold_candles:
            direction = _detect_direction(df, self.direction_lookback)
            self._candles_in_position.pop(pair, None)
            return Signal(
                action="exit",
                direction=direction,
                pair=pair,
                confidence=1.0,
                reason=f"Max hold reached ({candles_held} candles)",
                timestamp=0,
            )

        # 2. Smart exit — Kronos re-evaluation
        seq = _prepare_sequence(df, self.seq_len)
        if seq is None:
            return None

        prob = self.predictor.predict(seq)

        if prob < self.exit_threshold:
            direction = _detect_direction(df, self.direction_lookback)
            self._candles_in_position.pop(pair, None)
            return Signal(
                action="exit",
                direction=direction,
                pair=pair,
                confidence=1.0 - prob,
                reason=f"Kronos smart exit ({prob:.0%} win prob, {self.exit_mode} mode, held {candles_held} candles)",
                timestamp=0,
            )

        return None
