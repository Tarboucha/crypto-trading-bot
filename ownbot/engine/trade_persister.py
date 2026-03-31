"""Persists signals and trades to DB — subscribes to bus events."""
import logging

from shared.events.component import Component
from shared.events.trading import (
    SignalEntry, SignalExit, PositionOpened, PositionClosed, FundingApplied,
)
from shared.api.db.signal_repo import SignalRepo
from shared.api.db.trade_repo import TradeRepo

logger = logging.getLogger(__name__)


class TradePersister(Component):
    """Writes signals and trades to the database.

    Trades are written on OPEN (status="open") and updated on CLOSE (status="closed").
    Full lifecycle persisted — survives bot restarts.
    """

    def __init__(self, mode: str = "paper"):
        self.mode = mode
        self.signal_repo = SignalRepo()
        self.trade_repo = TradeRepo()

    async def on_signal_entry(self, event: SignalEntry):
        await self.signal_repo.store_signal(
            pair=event.pair, strategy=event.strategy,
            timeframe=event.timeframe, direction=event.direction,
            action="enter", confidence=event.confidence,
            reason=event.reason, timestamp=int(event.timestamp * 1000),
        )

    async def on_signal_exit(self, event: SignalExit):
        await self.signal_repo.store_signal(
            pair=event.pair, strategy=event.strategy,
            timeframe="", direction=event.direction,
            action="exit", confidence=event.confidence,
            reason=event.reason, timestamp=int(event.timestamp * 1000),
        )

    async def on_position_opened(self, event: PositionOpened):
        await self.trade_repo.open_trade(
            pair=event.pair, strategy=event.strategy,
            direction=event.direction,
            entry_price=event.entry_price, size=event.size,
            entry_time=int(event.timestamp * 1000),
            stoploss=event.stoploss, take_profit=event.take_profit,
            mode=self.mode,
        )

    async def on_position_closed(self, event: PositionClosed):
        await self.trade_repo.close_trade(
            pair=event.pair,
            exit_price=event.exit_price,
            exit_time=int(event.timestamp * 1000),
            profit_pct=event.profit_pct,
            profit_abs=event.profit_abs,
            funding_pnl=event.funding_pnl,
            reason=event.reason,
        )

    async def on_funding_applied(self, event: FundingApplied):
        await self.trade_repo.update_funding(
            pair=event.pair,
            cumulative_funding=event.cumulative,
            funding_events=0,  # event doesn't carry count, repo can increment
        )
