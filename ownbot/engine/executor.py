"""Order executors for different modes: paper, live, backtest."""
import logging
import time
from dataclasses import dataclass

from ownbot.strategy.base import Signal
from ownbot.loggers.messages import log_order_submitted

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    pair: str
    side: str               # "buy" | "sell"
    order_type: str          # "market" | "limit"
    price: float
    size: float
    success: bool
    order_id: str = ""


def _resolve_side(signal: Signal) -> str:
    """Determine order side from signal direction and action."""
    if signal.action == "enter":
        return "buy" if signal.direction == "long" else "sell"
    else:
        return "sell" if signal.direction == "long" else "buy"


class PaperExecutor:
    """Simulates order fills at current market price. No real orders."""

    def __init__(self):
        self._order_counter = 0

    async def execute(
        self,
        signal: Signal,
        current_price: float,
        size: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> ExecutionResult:
        self._order_counter += 1
        side = _resolve_side(signal)

        if order_type == "limit" and limit_price is not None:
            if signal.action == "enter":
                if signal.direction == "long" and current_price > limit_price:
                    return ExecutionResult(signal.pair, side, "limit", limit_price, size, success=False)
                if signal.direction == "short" and current_price < limit_price:
                    return ExecutionResult(signal.pair, side, "limit", limit_price, size, success=False)
            fill_price = limit_price
        else:
            fill_price = current_price

        logger.info("[PAPER] %s %s %.6f %s @ %.2f",
                    side.upper(), signal.pair, size, order_type, fill_price)

        return ExecutionResult(
            pair=signal.pair, side=side, order_type=order_type,
            price=fill_price, size=size, success=True,
            order_id=f"paper-{self._order_counter}",
        )


class LiveExecutor:
    """Places real orders on the exchange and tracks until filled or timeout."""

    def __init__(self, exchange, poll_interval_s: float = 1.0, timeout_s: float = 30.0):
        self.exchange = exchange
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s

    async def execute(
        self,
        signal: Signal,
        current_price: float,
        size: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> ExecutionResult:
        from ownbot.engine.order_tracker import OrderTracker

        side = _resolve_side(signal)
        price = limit_price if order_type == "limit" else None

        log_order_submitted(logger, signal.pair, side, size,
                            price or current_price, order_type)

        # Place order on exchange
        order = await self.exchange.place_order(
            symbol=signal.pair,
            side=side,
            order_type=order_type,
            amount=size,
            price=price,
        )

        # Track until filled, cancelled, or timeout
        tracker = OrderTracker(
            self.exchange,
            poll_interval_s=self.poll_interval_s,
            timeout_s=self.timeout_s,
        )
        final = await tracker.track(order)

        # Determine success
        success = final.status == "filled"
        fill_price = final.fill_price or final.price or current_price
        fill_size = final.filled_size or size if success else 0.0

        if not success:
            logger.warning("[%s] Order %s not filled — status: %s",
                           signal.pair, final.order_id, final.status)

        return ExecutionResult(
            pair=signal.pair,
            side=side,
            order_type=order_type,
            price=fill_price,
            size=fill_size,
            success=success,
            order_id=final.order_id,
        )


class BacktestExecutor:
    """Simulates fills using candle data for backtesting."""

    def __init__(self):
        self._order_counter = 0

    def execute(
        self,
        signal: Signal,
        candle_close: float,
        size: float,
    ) -> ExecutionResult:
        self._order_counter += 1
        side = _resolve_side(signal)

        return ExecutionResult(
            pair=signal.pair, side=side, order_type="market",
            price=candle_close, size=size, success=True,
            order_id=f"bt-{self._order_counter}",
        )
