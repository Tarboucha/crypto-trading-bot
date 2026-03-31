"""Tests for RiskManager — limits, approval, reset."""
import pytest
from ownbot.engine.risk_manager import RiskManager
from ownbot.engine.position_manager import PositionManager
from ownbot.strategy.base import Signal


def make_signal(pair="ETH"):
    return Signal(pair=pair, direction="long", action="enter",
                  confidence=0.8, reason="test", timestamp=0)


class TestCanTrade:

    def test_allowed_when_no_limits_hit(self, risk, positions):
        allowed, reason = risk.can_trade(make_signal(), 10000, positions)
        assert allowed
        assert reason == "ok"

    def test_max_open_trades(self, risk, positions):
        for i in range(3):
            positions.open(pair=f"P{i}", direction="long", entry_price=100,
                           size=1, entry_time=0, strategy="test")
        allowed, reason = risk.can_trade(make_signal(), 10000, positions)
        assert not allowed
        assert "max open trades" in reason.lower()

    def test_already_has_position(self, risk, positions):
        positions.open(pair="ETH", direction="long", entry_price=100,
                       size=1, entry_time=0, strategy="test")
        allowed, reason = risk.can_trade(make_signal("ETH"), 10000, positions)
        assert not allowed
        assert "already have position" in reason.lower()

    def test_loss_limit_blocks(self, risk, positions):
        risk.update_period_pnl(-600)  # -6% of 10000, limit is 5%
        allowed, reason = risk.can_trade(make_signal(), 10000, positions)
        assert not allowed
        assert "loss limit" in reason.lower()

    def test_loss_limit_allows_within(self, risk, positions):
        risk.update_period_pnl(-400)  # -4%, limit is 5%
        allowed, reason = risk.can_trade(make_signal(), 10000, positions)
        assert allowed

    def test_max_drawdown_blocks(self, risk, positions):
        risk.update_peak_balance(10000)
        # Balance dropped to 8900 = 11% drawdown, limit is 10%
        allowed, reason = risk.can_trade(make_signal(), 8900, positions)
        assert not allowed
        assert "drawdown" in reason.lower()

    def test_max_drawdown_allows_within(self, risk, positions):
        risk.update_peak_balance(10000)
        allowed, reason = risk.can_trade(make_signal(), 9500, positions)
        assert allowed


class TestPositionSizing:

    def test_basic_size(self):
        risk = RiskManager(risk_per_trade_pct=1.0)
        size = risk.calculate_position_size(10000, 2000)
        # 1% of 10000 = 100, 100 / 2000 = 0.05
        assert abs(size - 0.05) < 0.0001

    def test_size_scales_with_balance(self):
        risk = RiskManager(risk_per_trade_pct=1.0)
        size_small = risk.calculate_position_size(1000, 2000)
        size_large = risk.calculate_position_size(10000, 2000)
        assert size_large == size_small * 10


class TestPeriodReset:

    def test_manual_reset(self, risk):
        risk.update_period_pnl(-100)
        assert risk.period_pnl == -100
        risk.reset_period()
        assert risk.period_pnl == 0.0

    def test_auto_reset_after_interval(self):
        import time
        risk = RiskManager(loss_limit_reset=1)  # 1 second interval
        risk.update_period_pnl(-100)
        risk._last_reset_ts = time.time() - 2  # simulate 2s ago
        risk._check_reset()
        assert risk.period_pnl == 0.0

    def test_no_auto_reset_session_mode(self):
        risk = RiskManager(loss_limit_reset="session")
        risk.update_period_pnl(-100)
        risk._check_reset()
        assert risk.period_pnl == -100  # not reset


class TestPeakBalance:

    def test_peak_updates_on_higher(self, risk):
        risk.update_peak_balance(10000)
        risk.update_peak_balance(11000)
        assert risk.peak_balance == 11000

    def test_peak_stays_on_lower(self, risk):
        risk.update_peak_balance(10000)
        risk.update_peak_balance(9000)
        assert risk.peak_balance == 10000
