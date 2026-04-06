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
        size, margin = risk.calculate_position_size(10000, 2000)
        # 1% of 10000 = 100 margin, 100 / 2000 = 0.05 size at 1x
        assert abs(size - 0.05) < 0.0001
        assert abs(margin - 100.0) < 0.01

    def test_size_scales_with_balance(self):
        risk = RiskManager(risk_per_trade_pct=1.0)
        size_small, _ = risk.calculate_position_size(1000, 2000)
        size_large, _ = risk.calculate_position_size(10000, 2000)
        assert size_large == size_small * 10

    def test_leveraged_size(self):
        risk = RiskManager(risk_per_trade_pct=1.0)
        size, margin = risk.calculate_position_size(10000, 2000, leverage=3.0)
        # margin = 100, size = (100 / 2000) * 3 = 0.15
        assert abs(size - 0.15) < 0.0001
        assert abs(margin - 100.0) < 0.01

    def test_leverage_does_not_change_margin(self):
        risk = RiskManager(risk_per_trade_pct=1.0)
        _, margin_1x = risk.calculate_position_size(10000, 2000, leverage=1.0)
        _, margin_5x = risk.calculate_position_size(10000, 2000, leverage=5.0)
        assert margin_1x == margin_5x


class TestLeverage:

    def test_validate_leverage_clamps_high(self):
        risk = RiskManager(max_leverage=5.0)
        assert risk.validate_leverage(10.0) == 5.0

    def test_validate_leverage_clamps_low(self):
        risk = RiskManager(max_leverage=5.0)
        assert risk.validate_leverage(0.5) == 1.0

    def test_validate_leverage_within_range(self):
        risk = RiskManager(max_leverage=5.0)
        assert risk.validate_leverage(3.0) == 3.0

    def test_adjust_stoploss_for_leverage(self):
        risk = RiskManager()
        # -3% stoploss at 2x → -1.5% price move
        assert abs(risk.adjust_stoploss_for_leverage(-0.03, 2.0) - (-0.015)) < 0.0001

    def test_adjust_stoploss_1x_unchanged(self):
        risk = RiskManager()
        assert risk.adjust_stoploss_for_leverage(-0.03, 1.0) == -0.03

    def test_liquidation_price_long(self):
        risk = RiskManager()
        liq = risk.calculate_liquidation_price(2000.0, 5.0, "long", 0.005)
        # liq = 2000 * (1 - (1/5) * (1 - 0.005)) = 2000 * (1 - 0.199) = 2000 * 0.801 = 1602.0
        assert abs(liq - 1602.0) < 0.1

    def test_liquidation_price_short(self):
        risk = RiskManager()
        liq = risk.calculate_liquidation_price(2000.0, 5.0, "short", 0.005)
        # liq = 2000 * (1 + (1/5) * (1 - 0.005)) = 2000 * 1.199 = 2398.0
        assert abs(liq - 2398.0) < 0.1

    def test_liquidation_buffer_long(self):
        risk = RiskManager(liquidation_buffer=0.05)
        # Liquidation at 1600, entry at 2000 → buffer = (2000-1600)*0.05 = 20
        buffered = risk.apply_liquidation_buffer(1600.0, 2000.0)
        assert buffered == 1620.0  # moved closer to entry

    def test_liquidation_buffer_short(self):
        risk = RiskManager(liquidation_buffer=0.05)
        # Liquidation at 2400, entry at 2000 → buffer = (2400-2000)*0.05 = 20
        buffered = risk.apply_liquidation_buffer(2400.0, 2000.0)
        assert buffered == 2380.0  # moved closer to entry

    def test_stoploss_or_liquidation_long(self):
        risk = RiskManager()
        # Long: more protective = higher price
        assert risk.stoploss_or_liquidation(1950.0, 1600.0, "long") == 1950.0
        assert risk.stoploss_or_liquidation(1500.0, 1600.0, "long") == 1600.0

    def test_stoploss_or_liquidation_short(self):
        risk = RiskManager()
        # Short: more protective = lower price
        assert risk.stoploss_or_liquidation(2050.0, 2400.0, "short") == 2050.0
        assert risk.stoploss_or_liquidation(2500.0, 2400.0, "short") == 2400.0

    def test_check_total_exposure_within(self):
        risk = RiskManager(max_exposure_pct=10.0)
        # New position: 0.05 * 2000 = 100 notional, 100/10000 = 1% < 10%
        assert risk.check_total_exposure([], 0.05, 2000.0, 10000.0) is True

    def test_check_total_exposure_exceeded(self):
        risk = RiskManager(max_exposure_pct=1.0)
        # New position: 1.0 * 2000 = 2000 notional, 2000/10000 = 20% > 1%
        assert risk.check_total_exposure([], 1.0, 2000.0, 10000.0) is False


class TestLeverageValidation:
    """Edge cases and parametrized tests for validate_leverage."""

    def test_validate_leverage_exact_max(self):
        risk = RiskManager(max_leverage=5.0)
        assert risk.validate_leverage(5.0) == 5.0

    def test_validate_leverage_exact_min(self):
        risk = RiskManager(max_leverage=5.0)
        assert risk.validate_leverage(1.0) == 1.0

    def test_validate_leverage_negative(self):
        risk = RiskManager(max_leverage=5.0)
        assert risk.validate_leverage(-2.0) == 1.0

    def test_validate_leverage_zero(self):
        risk = RiskManager(max_leverage=5.0)
        assert risk.validate_leverage(0.0) == 1.0

    def test_validate_leverage_default_max_is_1x(self):
        risk = RiskManager()  # default max_leverage=1.0
        assert risk.validate_leverage(3.0) == 1.0


class TestStoplossLeverageInteraction:
    """The core invariant: risk per trade stays constant regardless of leverage."""

    @pytest.mark.parametrize("leverage", [1.0, 2.0, 3.0, 5.0, 10.0])
    def test_risk_constant_across_leverage(self, leverage):
        """At any leverage, the dollar loss at stoploss should equal risk_per_trade_pct of balance."""
        risk = RiskManager(risk_per_trade_pct=1.0, max_leverage=10.0)
        balance = 10000.0
        entry_price = 2000.0

        size, margin = risk.calculate_position_size(balance, entry_price, leverage)
        sl_pct = -0.03  # -3% stoploss
        effective_sl_pct = risk.adjust_stoploss_for_leverage(sl_pct, leverage)

        # Dollar loss = size * entry_price * |effective_sl_pct|
        dollar_loss = size * entry_price * abs(effective_sl_pct)
        # Should always equal margin * |sl_pct| = risk_amount * |sl_pct|
        expected_loss = margin * abs(sl_pct)
        assert abs(dollar_loss - expected_loss) < 0.0001

    @pytest.mark.parametrize("leverage", [1.0, 2.0, 5.0])
    def test_stoploss_price_gets_tighter_with_leverage(self, leverage):
        """Higher leverage → tighter price-based stoploss."""
        risk = RiskManager(max_leverage=10.0)
        entry = 2000.0
        sl_pct = -0.03
        effective = risk.adjust_stoploss_for_leverage(sl_pct, leverage)
        sl_price = entry * (1 + effective)  # long
        # Higher leverage → smaller |effective| → sl_price closer to entry
        assert sl_price <= entry
        if leverage > 1.0:
            sl_price_1x = entry * (1 + sl_pct)
            assert sl_price > sl_price_1x  # tighter = closer to entry

    def test_adjust_stoploss_at_5x(self):
        risk = RiskManager()
        # -3% stoploss at 5x → -0.6% price move
        result = risk.adjust_stoploss_for_leverage(-0.03, 5.0)
        assert abs(result - (-0.006)) < 0.0001

    def test_adjust_stoploss_at_10x(self):
        risk = RiskManager()
        result = risk.adjust_stoploss_for_leverage(-0.03, 10.0)
        assert abs(result - (-0.003)) < 0.0001


class TestLiquidationPrice:
    """Parametrized liquidation price and buffer tests."""

    @pytest.mark.parametrize("leverage,expected_long_liq", [
        (2.0, 2000.0 * (1 - 0.5 * 0.995)),       # ~1005
        (3.0, 2000.0 * (1 - (1/3) * 0.995)),      # ~1336.67
        (5.0, 2000.0 * (1 - 0.2 * 0.995)),        # ~1602
        (10.0, 2000.0 * (1 - 0.1 * 0.995)),       # ~1801
    ])
    def test_liquidation_long_parametrized(self, leverage, expected_long_liq):
        risk = RiskManager()
        liq = risk.calculate_liquidation_price(2000.0, leverage, "long", 0.005)
        assert abs(liq - expected_long_liq) < 0.1

    @pytest.mark.parametrize("leverage,expected_short_liq", [
        (2.0, 2000.0 * (1 + 0.5 * 0.995)),       # ~2995
        (5.0, 2000.0 * (1 + 0.2 * 0.995)),        # ~2398
        (10.0, 2000.0 * (1 + 0.1 * 0.995)),       # ~2199
    ])
    def test_liquidation_short_parametrized(self, leverage, expected_short_liq):
        risk = RiskManager()
        liq = risk.calculate_liquidation_price(2000.0, leverage, "short", 0.005)
        assert abs(liq - expected_short_liq) < 0.1

    def test_higher_leverage_closer_liquidation_long(self):
        risk = RiskManager()
        liq_2x = risk.calculate_liquidation_price(2000.0, 2.0, "long")
        liq_5x = risk.calculate_liquidation_price(2000.0, 5.0, "long")
        liq_10x = risk.calculate_liquidation_price(2000.0, 10.0, "long")
        # Higher leverage → liquidation closer to entry
        assert liq_10x > liq_5x > liq_2x

    def test_higher_leverage_closer_liquidation_short(self):
        risk = RiskManager()
        liq_2x = risk.calculate_liquidation_price(2000.0, 2.0, "short")
        liq_5x = risk.calculate_liquidation_price(2000.0, 5.0, "short")
        liq_10x = risk.calculate_liquidation_price(2000.0, 10.0, "short")
        # Higher leverage → liquidation closer to entry
        assert liq_10x < liq_5x < liq_2x

    def test_liquidation_buffer_zero(self):
        risk = RiskManager(liquidation_buffer=0.0)
        buffered = risk.apply_liquidation_buffer(1600.0, 2000.0)
        assert buffered == 1600.0  # no change

    def test_liquidation_buffer_100pct(self):
        risk = RiskManager(liquidation_buffer=1.0)
        # 100% buffer → liquidation moves all the way to entry
        buffered = risk.apply_liquidation_buffer(1600.0, 2000.0)
        assert buffered == 2000.0


class TestTotalExposure:
    """Exposure checks with existing positions, including leveraged ones."""

    def test_exposure_with_existing_positions(self):
        risk = RiskManager(max_exposure_pct=10.0)
        from ownbot.engine.position_manager import Position
        existing = [
            Position(pair="BTC", direction="long", entry_price=60000.0,
                     size=0.01, entry_time=0),  # 600 notional
        ]
        # New: 0.05 * 2000 = 100 notional → total 700 / 10000 = 7% < 10%
        assert risk.check_total_exposure(existing, 0.05, 2000.0, 10000.0) is True

    def test_exposure_exceeded_with_existing_positions(self):
        risk = RiskManager(max_exposure_pct=5.0)
        from ownbot.engine.position_manager import Position
        existing = [
            Position(pair="BTC", direction="long", entry_price=60000.0,
                     size=0.01, entry_time=0),  # 600 notional = 6%
        ]
        # Already at 6% > 5% limit, any new position should fail
        assert risk.check_total_exposure(existing, 0.01, 2000.0, 10000.0) is False

    def test_leveraged_position_counts_full_notional(self):
        """A 3x leveraged position's notional = size * entry_price (size is already leveraged)."""
        risk = RiskManager(max_exposure_pct=5.0)
        from ownbot.engine.position_manager import Position
        # margin=100, leverage=3x → size=0.15, notional=0.15*2000=300 = 3%
        existing = [
            Position(pair="ETH", direction="long", entry_price=2000.0,
                     size=0.15, entry_time=0, leverage=3.0, margin=100.0),
        ]
        # Adding 0.05 * 2000 = 100 notional → total 400/10000 = 4% < 5%
        assert risk.check_total_exposure(existing, 0.05, 2000.0, 10000.0) is True
        # Adding 0.15 * 2000 = 300 notional → total 600/10000 = 6% > 5%
        assert risk.check_total_exposure(existing, 0.15, 2000.0, 10000.0) is False


class TestEndToEndLeverageFlow:
    """Integration test: full leverage pipeline from the plan's example."""

    def test_plan_example_5x_long(self):
        """Reproduce the exact example from LEVERAGE_INTEGRATION_PLAN.md."""
        risk = RiskManager(
            risk_per_trade_pct=1.0, max_leverage=5.0, liquidation_buffer=0.05,
        )
        balance = 10000.0
        entry = 2000.0

        # Strategy requests 5x
        leverage = risk.validate_leverage(5.0)
        assert leverage == 5.0

        # Calculate size and margin
        size, margin = risk.calculate_position_size(balance, entry, leverage)
        assert abs(margin - 100.0) < 0.01        # 1% of 10000
        assert abs(size - 0.25) < 0.0001         # (100/2000)*5 = 0.25
        notional = size * entry
        assert abs(notional - 500.0) < 0.01      # $500 position

        # Stoploss adjusted for leverage
        sl_pct = -0.03
        effective_sl = risk.adjust_stoploss_for_leverage(sl_pct, leverage)
        assert abs(effective_sl - (-0.006)) < 0.0001  # -0.6% price move
        sl_price = entry * (1 + effective_sl)
        assert abs(sl_price - 1988.0) < 0.1

        # Liquidation price
        liq = risk.calculate_liquidation_price(entry, leverage, "long", 0.005)
        assert abs(liq - 1602.0) < 0.1

        # Buffered liquidation
        liq_buffered = risk.apply_liquidation_buffer(liq, entry)
        # buffer = (2000-1602)*0.05 = 19.9 → buffered ≈ 1621.9
        assert abs(liq_buffered - 1621.9) < 0.5

        # SL ($1988) is more protective than liquidation ($1621.9) for long
        final_sl = risk.stoploss_or_liquidation(sl_price, liq_buffered, "long")
        assert abs(final_sl - sl_price) < 0.01  # SL wins

        # Dollar loss at stoploss = size * entry * |effective_sl|
        dollar_loss = size * entry * abs(effective_sl)
        assert abs(dollar_loss - 3.0) < 0.01  # $3 = 3% of $100 margin

    def test_plan_example_1x_backward_compat(self):
        """At 1x leverage, behavior is identical to pre-leverage code."""
        risk = RiskManager(risk_per_trade_pct=1.0, max_leverage=5.0)
        balance = 10000.0
        entry = 2000.0

        leverage = risk.validate_leverage(1.0)
        size, margin = risk.calculate_position_size(balance, entry, leverage)

        # Same as old: size = risk_amount / price
        assert abs(size - 0.05) < 0.0001
        assert abs(margin - 100.0) < 0.01

        # Stoploss unchanged at 1x
        sl_pct = -0.03
        effective = risk.adjust_stoploss_for_leverage(sl_pct, 1.0)
        assert effective == sl_pct


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
