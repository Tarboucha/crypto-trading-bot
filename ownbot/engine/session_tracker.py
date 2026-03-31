"""Tracks bot sessions and events — subscribes to bus events."""
import logging

from shared.events.component import Component
from shared.events.trading import (
    SignalEntry, SignalExit, SignalRejected,
    OrderFilled, PositionOpened, PositionClosed,
)
from shared.events.system import EngineStarted, EngineStopped
from shared.api.db.session_repo import SessionRepo

logger = logging.getLogger(__name__)


class SessionTracker(Component):
    def __init__(self):
        self.session_id: int | None = None
        self.repo = SessionRepo()
        self._total_signals = 0
        self._total_trades = 0
        self._wins = 0
        self._losses = 0
        self._pnl_pct = 0.0
        self._pnl_abs = 0.0

    async def on_engine_started(self, event: EngineStarted):
        self.session_id = await self.repo.create_session(
            mode=event.mode, strategy=event.strategy,
            pairs=list(event.pairs), config={},
        )
        logger.info("Session #%d started.", self.session_id)

    async def on_engine_stopped(self, event: EngineStopped):
        if self.session_id is None:
            return
        await self.repo.close_session(
            session_id=self.session_id,
            stop_reason=event.reason,
            total_signals=self._total_signals,
            total_trades=self._total_trades,
            wins=self._wins, losses=self._losses,
            pnl_pct=self._pnl_pct, pnl_abs=self._pnl_abs,
        )
        logger.info(
            "Session #%d stopped (%s). Trades: %d (%dW/%dL) PnL: %.2f%%",
            self.session_id, event.reason, self._total_trades,
            self._wins, self._losses, self._pnl_pct * 100,
        )
        self.session_id = None

    async def on_signal_entry(self, event: SignalEntry):
        self._total_signals += 1
        await self._log("SIGNAL_ENTRY", event.pair, {
            "direction": event.direction, "confidence": event.confidence,
            "reason": event.reason, "strategy": event.strategy,
        })

    async def on_signal_exit(self, event: SignalExit):
        self._total_signals += 1
        await self._log("SIGNAL_EXIT", event.pair, {
            "direction": event.direction, "confidence": event.confidence,
            "reason": event.reason,
        })

    async def on_signal_rejected(self, event: SignalRejected):
        await self._log("SIGNAL_REJECTED", event.pair, {
            "direction": event.direction, "reason": event.reason,
        }, level="warning")

    async def on_order_filled(self, event: OrderFilled):
        await self._log("ORDER_FILLED", event.pair, {
            "side": event.side, "size": event.size,
            "price": event.price, "order_id": event.order_id,
        })

    async def on_position_opened(self, event: PositionOpened):
        self._total_trades += 1
        await self._log("POSITION_OPENED", event.pair, {
            "direction": event.direction, "entry_price": event.entry_price,
            "size": event.size, "stoploss": event.stoploss,
            "take_profit": event.take_profit, "strategy": event.strategy,
        })

    async def on_position_closed(self, event: PositionClosed):
        if event.profit_pct > 0:
            self._wins += 1
        else:
            self._losses += 1
        self._pnl_pct += event.profit_pct
        self._pnl_abs += event.profit_abs
        await self._log("POSITION_CLOSED", event.pair, {
            "direction": event.direction, "entry_price": event.entry_price,
            "exit_price": event.exit_price, "profit_pct": event.profit_pct,
            "profit_abs": event.profit_abs, "funding_pnl": event.funding_pnl,
            "reason": event.reason,
        })

    async def _log(self, event_type: str, pair: str | None = None,
                   details: dict | None = None, level: str = "info"):
        if self.session_id is None:
            return
        await self.repo.log_event(self.session_id, event_type, pair, details, level)
