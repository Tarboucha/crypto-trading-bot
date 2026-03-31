"""Extract features from candle data at the moment a signal fires."""
import numpy as np
import pandas as pd
import pandas_ta as ta

from shared.pairs import PAIR_ENCODING


class FeatureExtractor:
    """Compute features from candle data for ML model."""

    def extract(
        self,
        pair: str,
        df: pd.DataFrame,
        btc_df: pd.DataFrame | None = None,
        signal_direction: str = "long",
        signal_confidence: float = 0.5,
    ) -> dict:
        """Extract features from the latest candle in the DataFrame.

        Args:
            pair: Trading pair name
            df: OHLCV DataFrame with at least 50 candles
            btc_df: BTC OHLCV DataFrame (same timeframe) for cross-pair features
            signal_direction: "long" or "short"
            signal_confidence: Confidence from the base strategy

        Returns:
            Dict of feature_name → value
        """
        if len(df) < 50:
            return {}

        curr = df.iloc[-1]
        close = float(curr["close"])
        high = float(curr["high"])
        low = float(curr["low"])
        open_ = float(curr["open"])

        features = {}

        # --- Price / Indicator Features ---

        # RSI
        rsi_vals = ta.rsi(df["close"], length=14)
        features["rsi"] = float(rsi_vals.iloc[-1]) if rsi_vals is not None and not pd.isna(rsi_vals.iloc[-1]) else 50.0

        # RSI slope (change over last 3 candles)
        if rsi_vals is not None and len(rsi_vals) >= 4:
            features["rsi_slope"] = float(rsi_vals.iloc[-1] - rsi_vals.iloc[-4])
        else:
            features["rsi_slope"] = 0.0

        # Bollinger Bands position
        bb = ta.bbands(df["close"], length=20, std=2.0)
        if bb is not None and not bb.empty:
            bb_lower = float(bb.iloc[-1, 0])
            bb_upper = float(bb.iloc[-1, 2])
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                features["bb_position"] = (close - bb_lower) / bb_range
                features["bb_width"] = bb_range / close
            else:
                features["bb_position"] = 0.5
                features["bb_width"] = 0.0
        else:
            features["bb_position"] = 0.5
            features["bb_width"] = 0.0

        # SMA distance
        sma20 = ta.sma(df["close"], length=20)
        if sma20 is not None and not pd.isna(sma20.iloc[-1]):
            features["sma_distance"] = (close - float(sma20.iloc[-1])) / float(sma20.iloc[-1])
        else:
            features["sma_distance"] = 0.0

        # ATR normalized
        atr_vals = ta.atr(df["high"], df["low"], df["close"], length=14)
        if atr_vals is not None and not pd.isna(atr_vals.iloc[-1]):
            features["atr_pct"] = float(atr_vals.iloc[-1]) / close
        else:
            features["atr_pct"] = 0.0

        # Volume ratio
        vol_sma = ta.sma(df["volume"], length=20)
        if vol_sma is not None and not pd.isna(vol_sma.iloc[-1]) and float(vol_sma.iloc[-1]) > 0:
            features["volume_ratio"] = float(curr["volume"]) / float(vol_sma.iloc[-1])
        else:
            features["volume_ratio"] = 1.0

        # Volume trend
        vol_sma5 = ta.sma(df["volume"], length=5)
        if vol_sma5 is not None and vol_sma is not None and float(vol_sma.iloc[-1]) > 0:
            features["volume_trend"] = float(vol_sma5.iloc[-1]) / float(vol_sma.iloc[-1])
        else:
            features["volume_trend"] = 1.0

        # Candle shape
        candle_range = high - low
        if candle_range > 0:
            features["candle_body_pct"] = abs(close - open_) / candle_range
            features["upper_wick_pct"] = (high - max(open_, close)) / candle_range
            features["lower_wick_pct"] = (min(open_, close) - low) / candle_range
        else:
            features["candle_body_pct"] = 0.0
            features["upper_wick_pct"] = 0.0
            features["lower_wick_pct"] = 0.0

        # --- Cross-Pair Features (BTC) ---
        if btc_df is not None and len(btc_df) >= 50:
            btc_rsi = ta.rsi(btc_df["close"], length=14)
            features["btc_rsi"] = float(btc_rsi.iloc[-1]) if btc_rsi is not None and not pd.isna(btc_rsi.iloc[-1]) else 50.0

            btc_sma5 = ta.sma(btc_df["close"], length=5)
            btc_sma20 = ta.sma(btc_df["close"], length=20)
            if btc_sma5 is not None and btc_sma20 is not None:
                features["btc_trend"] = 1.0 if float(btc_sma5.iloc[-1]) > float(btc_sma20.iloc[-1]) else 0.0
            else:
                features["btc_trend"] = 0.5

            # Correlation with BTC over last 50 candles
            if len(df) >= 50 and len(btc_df) >= 50:
                corr = df["close"].iloc[-50:].corr(btc_df["close"].iloc[-50:])
                features["btc_correlation"] = float(corr) if not pd.isna(corr) else 0.0
            else:
                features["btc_correlation"] = 0.0

            # BTC return over last 12 candles
            if len(btc_df) >= 13:
                btc_ret = (float(btc_df["close"].iloc[-1]) - float(btc_df["close"].iloc[-13])) / float(btc_df["close"].iloc[-13])
                features["btc_return_12"] = btc_ret
            else:
                features["btc_return_12"] = 0.0
        else:
            features["btc_rsi"] = 50.0
            features["btc_trend"] = 0.5
            features["btc_correlation"] = 0.0
            features["btc_return_12"] = 0.0

        # --- Time Features ---
        timestamp_s = int(curr["timestamp"]) / 1000
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
        features["hour"] = dt.hour
        features["day_of_week"] = dt.weekday()
        features["is_weekend"] = 1.0 if dt.weekday() >= 5 else 0.0

        # --- Market Regime ---
        adx_vals = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_vals is not None and not adx_vals.empty:
            features["adx"] = float(adx_vals.iloc[-1, 0]) if not pd.isna(adx_vals.iloc[-1, 0]) else 25.0
        else:
            features["adx"] = 25.0

        # Realized volatility
        returns = df["close"].pct_change().iloc[-20:]
        features["returns_std"] = float(returns.std()) if len(returns) > 1 else 0.0

        # --- Signal-Specific ---
        features["signal_direction"] = 1.0 if signal_direction == "long" else 0.0
        features["signal_confidence"] = signal_confidence
        features["pair_encoded"] = float(PAIR_ENCODING.get(pair, -1))

        return features

    @staticmethod
    def feature_names() -> list[str]:
        """Return ordered list of feature names."""
        return [
            "rsi", "rsi_slope", "bb_position", "bb_width", "sma_distance",
            "atr_pct", "volume_ratio", "volume_trend",
            "candle_body_pct", "upper_wick_pct", "lower_wick_pct",
            "btc_rsi", "btc_trend", "btc_correlation", "btc_return_12",
            "hour", "day_of_week", "is_weekend",
            "adx", "returns_std",
            "signal_direction", "signal_confidence", "pair_encoded",
        ]
