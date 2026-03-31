"""Composite strategy: RSI Mean Reversion filtered by ML model."""
import pandas as pd

from ownbot.strategy.base import BaseStrategy, Signal
from ownbot.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from ownbot.strategy.ml.feature_extractor import FeatureExtractor
from ownbot.strategy.ml.predictor import MLPredictor


class MLFilteredRSIStrategy(BaseStrategy):
    name = "ml_filtered_rsi"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.timeframes = [self.params.get("timeframe", "5m")]
        self.threshold = self.params.get("ml_threshold", 0.60)
        self.model_path = self.params.get("model_path", None)

        self.rsi = RSIMeanReversionStrategy(params=params)
        self.extractor = FeatureExtractor()
        self.predictor = MLPredictor(model_path=self.model_path)

    def required_pairs(self) -> list[str]:
        return ["BTC"]

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.rsi.indicators(df)

    def should_enter(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        # 1. Get RSI signal
        signal = self.rsi.should_enter(pair, data)
        if not signal:
            return None

        # 2. Extract features — BTC data comes from data dict
        df = data[self.timeframes[0]]
        btc_key = f"BTC_{self.timeframes[0]}"
        btc_df = data.get(btc_key)

        features = self.extractor.extract(
            pair=pair,
            df=df,
            btc_df=btc_df,
            signal_direction=signal.direction,
            signal_confidence=signal.confidence,
        )

        if not features:
            return signal

        # 3. ML filter
        probability = self.predictor.predict(features)
        if probability < self.threshold:
            return None

        signal.confidence = probability
        signal.reason = f"[ML {probability:.0%}] {signal.reason}"
        return signal

    def should_exit(self, pair: str, data: dict[str, pd.DataFrame]) -> Signal | None:
        return self.rsi.should_exit(pair, data)
