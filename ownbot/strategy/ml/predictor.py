"""Load trained model and predict win probability for new signals."""
import logging
from pathlib import Path

import pandas as pd
import xgboost as xgb

from ownbot.strategy.ml.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class MLPredictor:
    """Load a trained XGBoost model and predict win probability."""

    def __init__(self, model_path: str | None = None):
        self.model = xgb.XGBClassifier()
        self.feature_names = FeatureExtractor.feature_names()

        if model_path is None:
            model_path = str(Path(__file__).parent.parent.parent.parent / "data" / "ml" / "models" / "rsi_filter_v1.json")

        if Path(model_path).exists():
            self.model.load_model(model_path)
            logger.info("ML model loaded from %s", model_path)
            self.loaded = True
        else:
            logger.warning("ML model not found at %s — predictions disabled.", model_path)
            self.loaded = False

    def predict(self, features: dict) -> float:
        """Return win probability (0.0 to 1.0).

        Args:
            features: Dict from FeatureExtractor.extract()

        Returns:
            Probability of the trade being a winner
        """
        if not self.loaded:
            return 0.5  # neutral if no model

        # Build DataFrame with correct column order
        row = {col: features.get(col, 0.0) for col in self.feature_names}
        X = pd.DataFrame([row])
        proba = self.model.predict_proba(X)[0][1]
        return float(proba)
