"""Interpret the RL conviction signal into actionable trading decisions.

The agent outputs a ∈ [-1, +1]. This module decides:
  - Should we enter? (conviction above entry threshold)
  - Should we exit? (conviction dropped or direction flipped)
  - Should we hold? (in the hysteresis band)
  - How to size the position? (conviction + uncertainty scaling)
"""
import logging

logger = logging.getLogger(__name__)


class SignalInterpreter:
    """Stateless interpreter: conviction → (action, direction, size_scalar)."""

    def __init__(
        self,
        entry_threshold: float = 0.6,
        exit_threshold: float = 0.3,
        epsilon: float = 0.05,
        conviction_size_min: float = 0.5,
        conviction_size_max: float = 1.0,
        unc_scale_low: float = 0.002,
        unc_scale_high: float = 0.01,
        warmup_ticks: int = 10,
        warmup_discount: float = 0.5,
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.epsilon = epsilon
        self.conviction_size_min = conviction_size_min
        self.conviction_size_max = conviction_size_max
        self.unc_scale_low = unc_scale_low
        self.unc_scale_high = unc_scale_high
        self.warmup_ticks = warmup_ticks
        self.warmup_discount = warmup_discount

    def effective_conviction(self, raw_conviction: float, tick_count: int) -> float:
        """Apply dead zone and warmup discount."""
        if abs(raw_conviction) < self.epsilon:
            return 0.0

        conviction = raw_conviction
        # Warmup discount: reduce conviction for first N ticks after LSTM reset
        if tick_count < self.warmup_ticks:
            conviction *= self.warmup_discount

        return conviction

    def should_enter(self, conviction: float) -> tuple[bool, str]:
        """Check if conviction warrants an entry.

        Returns:
            (should_enter, direction) where direction is "long" or "short"
        """
        if abs(conviction) >= self.entry_threshold:
            direction = "long" if conviction > 0 else "short"
            return True, direction
        return False, ""

    def should_exit(
        self, conviction: float, position_direction: str
    ) -> tuple[bool, str]:
        """Check if conviction warrants an exit.

        Returns:
            (should_exit, reason)
        """
        # Conviction dropped below exit threshold
        if abs(conviction) < self.exit_threshold:
            return True, f"conviction dropped to {abs(conviction):.2f} (< {self.exit_threshold})"

        # Direction flipped
        new_direction = "long" if conviction > 0 else "short"
        if new_direction != position_direction:
            return True, f"direction flipped to {new_direction} (conviction={conviction:+.2f})"

        return False, ""

    def position_size_scalar(
        self, conviction: float, sigma_return: float
    ) -> float:
        """Compute position size scalar from conviction and uncertainty.

        Returns a value in [conviction_size_min, conviction_size_max] that
        the engine multiplies with the base margin.
        """
        abs_conv = abs(conviction)

        # Remap conviction from [entry_threshold, 1.0] → [size_min, size_max]
        if abs_conv >= 1.0:
            conv_scalar = self.conviction_size_max
        elif abs_conv <= self.entry_threshold:
            conv_scalar = self.conviction_size_min
        else:
            t = (abs_conv - self.entry_threshold) / (1.0 - self.entry_threshold)
            conv_scalar = self.conviction_size_min + t * (self.conviction_size_max - self.conviction_size_min)

        # Uncertainty scalar: high sigma → reduce size
        if sigma_return <= self.unc_scale_low:
            unc_scalar = 1.0
        elif sigma_return >= self.unc_scale_high:
            unc_scalar = 0.5
        else:
            t = (sigma_return - self.unc_scale_low) / (self.unc_scale_high - self.unc_scale_low)
            unc_scalar = 1.0 - 0.5 * t

        return conv_scalar * unc_scalar
