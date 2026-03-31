"""Shared test fixtures."""
import pytest
from unittest.mock import AsyncMock

from shared.events.bus import EventBus
from shared.api.exchange.base import Ticker, Balance, OrderResult, Position
from ownbot.engine.position_manager import PositionManager
from ownbot.engine.risk_manager import RiskManager
from ownbot.strategy.base import Signal
from shared.pairs import parse_pair


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def positions():
    return PositionManager()


@pytest.fixture
def risk():
    return RiskManager(
        max_open_trades=3,
        risk_per_trade_pct=1.0,
        max_exposure_pct=10.0,
        loss_limit_pct=5.0,
        max_drawdown_pct=10.0,
    )


@pytest.fixture
def signal_long():
    return Signal(
        pair="ETH/USDT:USDT", direction="long", action="enter",
        confidence=0.8, reason="test", timestamp=1000000,
    )


@pytest.fixture
def signal_short():
    return Signal(
        pair="ETH/USDT:USDT", direction="short", action="enter",
        confidence=0.8, reason="test", timestamp=1000000,
    )


@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()
    exchange.get_ticker.return_value = Ticker(
        symbol="ETH", last=2100.0, bid=2099.5, ask=2100.5, volume=1000.0,
    )
    exchange.get_positions.return_value = []
    exchange.get_balance.return_value = Balance(total=10000.0, free=9000.0, used=1000.0)
    exchange.place_order.return_value = OrderResult(
        order_id="test-1", symbol="ETH", side="buy",
        order_type="market", amount=0.01, price=2100.0,
        status="filled", filled_size=0.01, fill_price=2100.0,
    )
    exchange.get_funding_rate.return_value = 0.0003
    exchange.cancel_order.return_value = True
    return exchange


@pytest.fixture
def eth_pair():
    return parse_pair("ETH/USDT", trading_mode="futures")


@pytest.fixture
def btc_pair():
    return parse_pair("BTC/USDT", trading_mode="futures")
