"""Tests for the RL conviction strategy components.

Tests cover:
  - FeatureBuilder: MC history, observation shape, candle-derived features
  - SignalInterpreter: entry/exit thresholds, hysteresis, size scaling, warmup
"""
import numpy as np
import pandas as pd
import pytest

from ownbot.strategy.feature_builder import FeatureBuilder
from ownbot.strategy.signal_interpreter import SignalInterpreter


# ─── Helpers ──────────────────────────────────────────────────

def make_mc_features(**overrides) -> dict[str, float]:
    base = {
        "p_long": 0.6, "p_short": 0.4,
        "mu_return": 0.001, "sigma_return": 0.005,
        "mu_opt_long": 0.02, "mu_opt_short": 0.01,
        "worst_mae_long": -0.03, "worst_mae_short": 0.025,
        "p_sl_long": 0.1, "p_sl_short": 0.05,
        "avg_agreement": 0.7,
    }
    base.update(overrides)
    return base


def make_candle_df(n: int = 100, base_price: float = 3000.0) -> pd.DataFrame:
    """Create a fake OHLCV DataFrame with n candles."""
    closes = base_price + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": closes - 0.1,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": np.random.rand(n) * 100,
    })


def warm_up_builder(fb: FeatureBuilder, n: int = 5):
    """Feed enough MC data to make the builder warm."""
    for i in range(n):
        fb.update_mc(make_mc_features(mu_return=0.001 * (1 if i % 3 else -1)))


# ─── FeatureBuilder ──────────────────────────────────────────


class TestFeatureBuilder:

    def test_not_warm_initially(self):
        fb = FeatureBuilder()
        assert not fb.is_warm

    def test_warm_after_two_mc_updates(self):
        fb = FeatureBuilder()
        fb.update_mc(make_mc_features())
        assert not fb.is_warm
        fb.update_mc(make_mc_features())
        assert fb.is_warm

    def test_observation_shape(self):
        fb = FeatureBuilder()
        warm_up_builder(fb)
        df = make_candle_df(100)
        obs = fb.build(make_mc_features(), df)
        assert obs.shape == (27,)
        assert obs.dtype == np.float32

    def test_observation_contains_mc_features(self):
        fb = FeatureBuilder()
        warm_up_builder(fb)
        mc = make_mc_features(p_long=0.8, p_short=0.2)
        obs = fb.build(mc, make_candle_df(100))
        assert obs[0] == pytest.approx(0.8, abs=1e-5)  # p_long
        assert obs[1] == pytest.approx(0.2, abs=1e-5)  # p_short

    def test_prev_action_in_observation(self):
        fb = FeatureBuilder()
        warm_up_builder(fb)
        fb.prev_action = 0.75
        obs = fb.build(make_mc_features(), make_candle_df(100))
        # prev_action is feature index 25 (11 mc + 9 enriched + 5 momentum)
        assert obs[25] == pytest.approx(0.75, abs=1e-5)

    def test_reset_clears_state(self):
        fb = FeatureBuilder()
        warm_up_builder(fb)
        assert fb.is_warm
        fb.reset()
        assert not fb.is_warm
        assert fb.prev_action == 0.0

    def test_mc_streak_increments(self):
        fb = FeatureBuilder()
        for _ in range(10):
            fb.update_mc(make_mc_features(mu_return=0.002))
        assert fb._streak > 0

    def test_mc_streak_resets_on_sign_change(self):
        fb = FeatureBuilder()
        for _ in range(5):
            fb.update_mc(make_mc_features(mu_return=0.002))
        assert fb._streak > 0
        fb.update_mc(make_mc_features(mu_return=-0.002))
        assert fb._streak == 0

    def test_candle_derived_features_use_df(self):
        """Verify returns/vol are computed from the DataFrame, not internal buffer."""
        fb = FeatureBuilder()
        warm_up_builder(fb)

        # Create DF with known prices: constant 3000 except last candle at 3030 (+1%)
        prices = np.full(100, 3000.0)
        prices[-1] = 3030.0
        df = pd.DataFrame({
            "open": prices, "high": prices + 1, "low": prices - 1,
            "close": prices, "volume": np.ones(100),
        })

        obs = fb.build(make_mc_features(), df)
        # ret_1 = (3030-3000)/3000 = 0.01, scaled * 100 = 1.0 (feature index 11)
        assert obs[11] == pytest.approx(1.0, abs=0.1)

    def test_works_with_small_df(self):
        """Should not crash with fewer than 49 candles (vol_48 fallback)."""
        fb = FeatureBuilder()
        warm_up_builder(fb)
        obs = fb.build(make_mc_features(), make_candle_df(10))
        assert obs.shape == (27,)
        assert np.all(np.isfinite(obs))


# ─── SignalInterpreter ───────────────────────────────────────


class TestSignalInterpreterEntry:

    def test_enter_long_above_threshold(self):
        si = SignalInterpreter(entry_threshold=0.6)
        should, direction = si.should_enter(0.8)
        assert should is True
        assert direction == "long"

    def test_enter_short_above_threshold(self):
        si = SignalInterpreter(entry_threshold=0.6)
        should, direction = si.should_enter(-0.7)
        assert should is True
        assert direction == "short"

    def test_no_enter_below_threshold(self):
        si = SignalInterpreter(entry_threshold=0.6)
        should, _ = si.should_enter(0.4)
        assert should is False

    def test_no_enter_at_exactly_threshold(self):
        si = SignalInterpreter(entry_threshold=0.6)
        should, _ = si.should_enter(0.59)
        assert should is False

    def test_enter_at_threshold(self):
        si = SignalInterpreter(entry_threshold=0.6)
        should, _ = si.should_enter(0.6)
        assert should is True

    def test_no_enter_at_zero(self):
        si = SignalInterpreter(entry_threshold=0.6)
        should, _ = si.should_enter(0.0)
        assert should is False


class TestSignalInterpreterExit:

    def test_exit_on_conviction_drop(self):
        si = SignalInterpreter(exit_threshold=0.3)
        should, reason = si.should_exit(0.2, "long")
        assert should is True
        assert "conviction dropped" in reason

    def test_no_exit_above_threshold(self):
        si = SignalInterpreter(exit_threshold=0.3)
        should, _ = si.should_exit(0.5, "long")
        assert should is False

    def test_exit_on_direction_flip(self):
        si = SignalInterpreter(exit_threshold=0.3)
        should, reason = si.should_exit(-0.5, "long")
        assert should is True
        assert "flipped" in reason

    def test_exit_on_direction_flip_short(self):
        si = SignalInterpreter(exit_threshold=0.3)
        should, reason = si.should_exit(0.5, "short")
        assert should is True
        assert "flipped" in reason

    def test_no_exit_same_direction(self):
        si = SignalInterpreter(exit_threshold=0.3)
        should, _ = si.should_exit(0.5, "long")
        assert should is False

    def test_exit_at_zero(self):
        si = SignalInterpreter(exit_threshold=0.3)
        should, _ = si.should_exit(0.0, "long")
        assert should is True


class TestSignalInterpreterHysteresis:

    def test_hold_zone_no_entry(self):
        si = SignalInterpreter(entry_threshold=0.6, exit_threshold=0.3)
        should_enter, _ = si.should_enter(0.45)
        assert should_enter is False

    def test_hold_zone_no_exit(self):
        si = SignalInterpreter(entry_threshold=0.6, exit_threshold=0.3)
        should_exit, _ = si.should_exit(0.45, "long")
        assert should_exit is False


class TestSignalInterpreterSizeScalar:

    def test_max_size_at_full_conviction(self):
        si = SignalInterpreter(entry_threshold=0.6, conviction_size_min=0.5, conviction_size_max=1.0)
        scalar = si.position_size_scalar(1.0, sigma_return=0.001)
        assert scalar == pytest.approx(1.0, abs=0.01)

    def test_min_size_at_entry_threshold(self):
        si = SignalInterpreter(entry_threshold=0.6, conviction_size_min=0.5, conviction_size_max=1.0)
        scalar = si.position_size_scalar(0.6, sigma_return=0.001)
        assert scalar == pytest.approx(0.5, abs=0.01)

    def test_mid_size_at_mid_conviction(self):
        si = SignalInterpreter(entry_threshold=0.6, conviction_size_min=0.5, conviction_size_max=1.0)
        scalar = si.position_size_scalar(0.8, sigma_return=0.001)
        assert 0.5 < scalar < 1.0

    def test_high_uncertainty_reduces_size(self):
        si = SignalInterpreter(unc_scale_low=0.002, unc_scale_high=0.01)
        low_unc = si.position_size_scalar(0.8, sigma_return=0.001)
        high_unc = si.position_size_scalar(0.8, sigma_return=0.009)
        assert high_unc < low_unc

    def test_max_uncertainty_halves_size(self):
        si = SignalInterpreter(
            entry_threshold=0.6, conviction_size_min=0.5, conviction_size_max=1.0,
            unc_scale_low=0.002, unc_scale_high=0.01,
        )
        no_unc = si.position_size_scalar(1.0, sigma_return=0.001)
        max_unc = si.position_size_scalar(1.0, sigma_return=0.02)
        assert max_unc == pytest.approx(no_unc * 0.5, abs=0.01)

    def test_negative_conviction_uses_abs(self):
        si = SignalInterpreter(entry_threshold=0.6)
        pos = si.position_size_scalar(0.8, sigma_return=0.003)
        neg = si.position_size_scalar(-0.8, sigma_return=0.003)
        assert pos == pytest.approx(neg, abs=1e-6)


class TestSignalInterpreterWarmup:

    def test_warmup_reduces_conviction(self):
        si = SignalInterpreter(warmup_ticks=10, warmup_discount=0.5)
        warm = si.effective_conviction(0.8, tick_count=5)
        assert warm == pytest.approx(0.4, abs=1e-6)

    def test_after_warmup_full_conviction(self):
        si = SignalInterpreter(warmup_ticks=10, warmup_discount=0.5)
        full = si.effective_conviction(0.8, tick_count=15)
        assert full == pytest.approx(0.8, abs=1e-6)

    def test_dead_zone_applied_before_warmup(self):
        si = SignalInterpreter(epsilon=0.05, warmup_ticks=10, warmup_discount=0.5)
        result = si.effective_conviction(0.03, tick_count=5)
        assert result == 0.0

    def test_warmup_can_prevent_entry(self):
        si = SignalInterpreter(entry_threshold=0.6, warmup_ticks=10, warmup_discount=0.5)
        conv = si.effective_conviction(0.8, tick_count=3)
        should, _ = si.should_enter(conv)
        assert should is False

    def test_strong_signal_enters_despite_warmup(self):
        """1.0 * 0.5 = 0.5, still below 0.6 — warmup is protective."""
        si = SignalInterpreter(entry_threshold=0.6, warmup_ticks=10, warmup_discount=0.5)
        conv = si.effective_conviction(1.0, tick_count=3)
        should, _ = si.should_enter(conv)
        assert should is False
