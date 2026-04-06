"""Kronos inference — load trained model, predict win probability.

This module is lightweight: only loads weights and runs forward pass.
All training/fine-tuning happens in mllab/.
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent.parent / "data" / "ml" / "models"


class KronosPredictor:
    """Load a trained Kronos classifier and predict win probability."""

    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.loaded = False
        self.model = None
        self.tokenizer = None

        if model_path is None:
            model_path = str(MODEL_DIR / "kronos_phase1.pt")

        if not Path(model_path).exists():
            logger.warning("Kronos model not found at %s — predictions disabled.", model_path)
            return

        try:
            import sys
            kronos_path = str(Path(__file__).parent.parent.parent.parent / "third_party" / "kronos")
            if kronos_path not in sys.path:
                sys.path.insert(0, kronos_path)
            from model import KronosTokenizer, Kronos

            logger.info("Loading Kronos-Base encoder...")
            self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            encoder = Kronos.from_pretrained("NeoQuasar/Kronos-base")

            # Build classifier with same architecture as training
            self.model = _KronosClassifierInference(encoder, self.tokenizer)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            logger.info("Kronos model loaded from %s (device=%s)", model_path, self.device)
        except Exception as e:
            logger.warning("Failed to load Kronos: %s", e)

    def predict(self, candles_tensor: torch.Tensor) -> float:
        """Return win probability (0.0 to 1.0).

        Args:
            candles_tensor: [seq_len, 6] — OHLCVA candle sequence

        Returns:
            Probability of the trade being a winner
        """
        if not self.loaded:
            return 0.5

        with torch.no_grad():
            x = candles_tensor.unsqueeze(0).to(self.device)  # [1, seq_len, 6]
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].cpu())


class _KronosClassifierInference(nn.Module):
    """Same architecture as mllab/training/kronos_trainer.py KronosClassifier."""

    def __init__(self, encoder, tokenizer, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        encoder_dim = encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, candles):
        tokens = self.tokenizer.encode(candles)
        embeddings = self.encoder(tokens)
        cls_embedding = embeddings[:, -1, :]
        return self.head(cls_embedding)
