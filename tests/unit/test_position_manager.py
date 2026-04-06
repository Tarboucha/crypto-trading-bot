"""Tests for PositionManager — P&L calculations, funding, open/close."""
import pytest
from ownbot.engine.position_manager import PositionManager


class TestOpenClose:

    def test_open_and_close(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test")
        assert positions.has_position("ETH")
        assert positions.count == 1

        closed = positions.close("ETH", 2100.0, 2000, "test exit")
        assert not positions.has_position("ETH")
        assert positions.count == 0
        assert closed.reason == "test exit"

    def test_close_nonexistent_raises(self, positions):
        with pytest.raises(RuntimeError, match="No open position"):
            positions.close("FAKE", 100.0, 1000, "test")

    def test_duplicate_open_raises(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test")
        with pytest.raises(RuntimeError, match="Already have"):
            positions.open(pair="ETH", direction="long", entry_price=2000.0,
                           size=1.0, entry_time=1000, strategy="test")

    def test_multiple_pairs(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test")
        positions.open(pair="BTC", direction="short", entry_price=60000.0,
                       size=0.1, entry_time=1000, strategy="test")
        assert positions.count == 2
        assert positions.has_position("ETH")
        assert positions.has_position("BTC")

    def test_get_position(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test")
        pos = positions.get_position("ETH")
        assert pos.direction == "long"
        assert pos.entry_price == 2000.0
        assert pos.size == 1.0

    def test_get_nonexistent_returns_none(self, positions):
        assert positions.get_position("FAKE") is None


class TestPnLCalculation:
    """P&L math — wrong here = lose money."""

    @pytest.mark.parametrize("direction,entry,exit_price,expected_pct", [
        ("long", 2000.0, 2100.0, 0.05),       # +5%
        ("long", 2000.0, 1900.0, -0.05),      # -5%
        ("short", 2000.0, 1900.0, 0.05),      # +5%
        ("short", 2000.0, 2100.0, -0.05),     # -5%
        ("long", 2000.0, 2000.0, 0.0),        # breakeven
        ("short", 2000.0, 2000.0, 0.0),       # breakeven
        ("long", 100.0, 200.0, 1.0),          # +100%
        ("long", 100.0, 50.0, -0.5),          # -50%
        ("short", 100.0, 50.0, 0.5),          # +50%
        ("short", 100.0, 200.0, -1.0),        # -100%
    ])
    def test_profit_pct(self, positions, direction, entry, exit_price, expected_pct):
        positions.open(pair="ETH", direction=direction, entry_price=entry,
                       size=1.0, entry_time=1000, strategy="test")
        closed = positions.close("ETH", exit_price, 2000, "test")
        assert abs(closed.profit_pct - expected_pct) < 0.0001

    @pytest.mark.parametrize("direction,entry,exit_price,size,expected_abs", [
        ("long", 2000.0, 2100.0, 1.0, 100.0),      # 1 ETH * (2100-2000) = 100
        ("long", 2000.0, 2100.0, 0.5, 50.0),        # 0.5 ETH * 100 = 50
        ("short", 2000.0, 1900.0, 1.0, 100.0),      # 1 ETH * (2000-1900) = 100
        ("long", 2000.0, 1900.0, 1.0, -100.0),      # loss
        ("long", 2000.0, 2000.0, 1.0, 0.0),         # breakeven
    ])
    def test_profit_abs(self, positions, direction, entry, exit_price, size, expected_abs):
        positions.open(pair="ETH", direction=direction, entry_price=entry,
                       size=size, entry_time=1000, strategy="test")
        closed = positions.close("ETH", exit_price, 2000, "test")
        assert abs(closed.profit_abs - expected_abs) < 0.01

    def test_closed_trade_carries_metadata(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="kronos",
                       stoploss=1950.0, take_profit=2100.0)
        closed = positions.close("ETH", 2050.0, 2000, "smart exit")
        assert closed.pair == "ETH"
        assert closed.direction == "long"
        assert closed.entry_price == 2000.0
        assert closed.exit_price == 2050.0
        assert closed.entry_time == 1000
        assert closed.exit_time == 2000
        assert closed.strategy == "kronos"
        assert closed.reason == "smart exit"


class TestFunding:

    def test_apply_funding_long_pays_positive(self, positions):
        """Long pays when funding rate is positive."""
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test")
        amount = positions.apply_funding("ETH", 0.001)  # +0.1% rate
        assert amount < 0  # long pays
        pos = positions.get_position("ETH")
        assert pos.cumulative_funding < 0
        assert pos.funding_events == 1

    def test_apply_funding_short_receives_positive(self, positions):
        """Short receives when funding rate is positive."""
        positions.open(pair="ETH", direction="short", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test")
        amount = positions.apply_funding("ETH", 0.001)
        assert amount > 0  # short receives

    def test_funding_accumulates(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test")
        positions.apply_funding("ETH", 0.001)
        positions.apply_funding("ETH", 0.001)
        positions.apply_funding("ETH", 0.001)
        pos = positions.get_position("ETH")
        assert pos.funding_events == 3
        assert pos.cumulative_funding < 0

    def test_funding_included_in_close_pnl(self, positions):
        """Funding should be added to profit_abs on close."""
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test")
        positions.apply_funding("ETH", 0.001)  # pays ~$2
        closed = positions.close("ETH", 2000.0, 2000, "test")
        # Price P&L = 0, but funding was paid
        assert closed.funding_pnl < 0
        assert closed.profit_abs == closed.funding_pnl

    def test_apply_funding_nonexistent_returns_none(self, positions):
        assert positions.apply_funding("FAKE", 0.001) is None


class TestStoplosssTakeprofit:

    def test_stoploss_long(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test",
                       stoploss=1950.0, take_profit=2100.0)
        closed = positions.check_stoploss_takeprofit("ETH", 1940.0, 2000)
        assert closed is not None
        assert closed.reason == "stoploss hit"
        assert closed.profit_pct < 0

    def test_takeprofit_long(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test",
                       stoploss=1950.0, take_profit=2100.0)
        closed = positions.check_stoploss_takeprofit("ETH", 2150.0, 2000)
        assert closed is not None
        assert closed.reason == "takeprofit hit"
        assert closed.profit_pct > 0

    def test_stoploss_short(self, positions):
        positions.open(pair="ETH", direction="short", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test",
                       stoploss=2050.0, take_profit=1900.0)
        closed = positions.check_stoploss_takeprofit("ETH", 2060.0, 2000)
        assert closed is not None
        assert closed.reason == "stoploss hit"

    def test_no_sl_tp_returns_none(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test")
        assert positions.check_stoploss_takeprofit("ETH", 1500.0, 2000) is None

    def test_price_between_sl_tp_returns_none(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=1000, strategy="test",
                       stoploss=1950.0, take_profit=2100.0)
        assert positions.check_stoploss_takeprofit("ETH", 2050.0, 2000) is None


class TestLeveragePosition:

    def test_open_with_leverage(self, positions):
        pos = positions.open(
            pair="ETH", direction="long", entry_price=2000.0,
            size=0.15, entry_time=1000, strategy="test",
            leverage=3.0, margin=100.0, liquidation_price=1602.0,
        )
        assert pos.leverage == 3.0
        assert pos.margin == 100.0
        assert pos.liquidation_price == 1602.0

    def test_default_leverage_is_1x(self, positions):
        pos = positions.open(
            pair="ETH", direction="long", entry_price=2000.0,
            size=0.05, entry_time=1000, strategy="test",
        )
        assert pos.leverage == 1.0
        assert pos.margin == 0.0

    def test_closed_trade_carries_leverage(self, positions):
        positions.open(
            pair="ETH", direction="long", entry_price=2000.0,
            size=0.15, entry_time=1000, strategy="test",
            leverage=3.0, margin=100.0,
        )
        closed = positions.close("ETH", 2100.0, 2000, "test")
        assert closed.leverage == 3.0
        assert closed.margin == 100.0

    def test_leveraged_pnl_abs(self, positions):
        """3x leverage on 1 ETH: size=3.0 (leveraged), entry=2000, exit=2100."""
        positions.open(
            pair="ETH", direction="long", entry_price=2000.0,
            size=3.0, entry_time=1000, strategy="test",
            leverage=3.0, margin=2000.0,
        )
        closed = positions.close("ETH", 2100.0, 2000, "test")
        # profit_abs = profit_pct * size * entry_price = 0.05 * 3.0 * 2000 = 300
        assert abs(closed.profit_abs - 300.0) < 0.01


class TestTrailingStoploss:

    def test_trailing_moves_up_on_new_high_long(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=0, strategy="test",
                       stoploss=1970.0, trailing_stop=True,
                       trailing_distance_pct=1.5)
        # Price moves up — SL should follow
        positions.check_stoploss_takeprofit("ETH", 2100.0, 1000)
        pos = positions.get_position("ETH")
        expected_sl = 2100.0 * (1 - 0.015)  # 2068.50
        assert abs(pos.stoploss - expected_sl) < 0.01
        assert pos.stoploss > 1970.0

    def test_trailing_never_moves_down(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=0, strategy="test",
                       stoploss=1970.0, trailing_stop=True,
                       trailing_distance_pct=1.5)
        # Price goes to 2100 → SL trails to ~2068.50
        positions.check_stoploss_takeprofit("ETH", 2100.0, 1000)
        sl_after_high = positions.get_position("ETH").stoploss
        # Price dips to 2080 (above trailing SL) → SL should NOT move down
        positions.check_stoploss_takeprofit("ETH", 2080.0, 2000)
        sl_after_dip = positions.get_position("ETH").stoploss
        assert sl_after_dip == sl_after_high

    def test_trailing_activation_threshold(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=0, strategy="test",
                       stoploss=1970.0, trailing_stop=True,
                       trailing_distance_pct=1.5,
                       trailing_activate_pct=2.0)
        # Price at +1% — below activation
        positions.check_stoploss_takeprofit("ETH", 2020.0, 1000)
        assert positions.get_position("ETH").stoploss == 1970.0  # unchanged
        # Price at +3% — above activation
        positions.check_stoploss_takeprofit("ETH", 2060.0, 2000)
        assert positions.get_position("ETH").stoploss > 1970.0

    def test_trailing_disabled_doesnt_move(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=0, strategy="test",
                       stoploss=1970.0, trailing_stop=False)
        positions.check_stoploss_takeprofit("ETH", 2200.0, 1000)
        assert positions.get_position("ETH").stoploss == 1970.0

    def test_trailing_short_moves_down(self, positions):
        positions.open(pair="ETH", direction="short", entry_price=2000.0,
                       size=1.0, entry_time=0, strategy="test",
                       stoploss=2030.0, trailing_stop=True,
                       trailing_distance_pct=1.5)
        # Price drops — SL should follow down
        positions.check_stoploss_takeprofit("ETH", 1900.0, 1000)
        pos = positions.get_position("ETH")
        expected_sl = 1900.0 * (1 + 0.015)  # 1928.50
        assert abs(pos.stoploss - expected_sl) < 0.01
        assert pos.stoploss < 2030.0

    def test_trailing_short_never_moves_up(self, positions):
        positions.open(pair="ETH", direction="short", entry_price=2000.0,
                       size=1.0, entry_time=0, strategy="test",
                       stoploss=2030.0, trailing_stop=True,
                       trailing_distance_pct=1.5)
        # Price drops to 1900 → SL trails to ~1928.50
        positions.check_stoploss_takeprofit("ETH", 1900.0, 1000)
        sl_after_low = positions.get_position("ETH").stoploss
        # Price bounces to 1920 (below trailing SL) → SL should NOT move up
        positions.check_stoploss_takeprofit("ETH", 1920.0, 2000)
        sl_after_bounce = positions.get_position("ETH").stoploss
        assert sl_after_bounce == sl_after_low

    def test_trailing_triggers_stoploss(self, positions):
        positions.open(pair="ETH", direction="long", entry_price=2000.0,
                       size=1.0, entry_time=0, strategy="test",
                       stoploss=1970.0, trailing_stop=True,
                       trailing_distance_pct=1.5)
        # Price goes up, SL trails
        positions.check_stoploss_takeprofit("ETH", 2200.0, 1000)
        # SL should be at ~2167
        # Price drops to trailing SL
        closed = positions.check_stoploss_takeprofit("ETH", 2160.0, 2000)
        assert closed is not None
        assert closed.reason == "stoploss hit"
        assert closed.profit_pct > 0  # still profitable despite SL hit
