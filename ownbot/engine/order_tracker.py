"""Order tracking — polls exchange until order fills, times out, or is cancelled."""
import asyncio
import logging
import time

from shared.api.exchange.base import BaseExchange, OrderResult
from ownbot.loggers.messages import log_order_filled, log_order_cancelled

logger = logging.getLogger(__name__)


class OrderTracker:
    """Tracks order lifecycle: placed → filled/timeout/cancelled.

    After placing an order, call track() to poll until resolution.

    Config:
        poll_interval_s: seconds between status checks (default 1)
        timeout_s: cancel order after this many seconds (default 30)
        max_retries: max poll attempts before giving up (default 30)
    """

    def __init__(
        self,
        exchange: BaseExchange,
        poll_interval_s: float = 1.0,
        timeout_s: float = 30.0,
    ):
        self.exchange = exchange
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s

    async def track(self, order: OrderResult) -> OrderResult:
        """Poll order status until filled, cancelled, or timeout.

        Args:
            order: The initial OrderResult from place_order()

        Returns:
            Updated OrderResult with final status
        """
        # Already filled (market orders usually fill instantly)
        if order.status == "filled":
            log_order_filled(logger, order.symbol, order.side,
                             order.filled_size or order.amount,
                             order.fill_price or order.price or 0.0)
            return order

        # Rejected immediately
        if order.status in ("rejected", "cancelled"):
            log_order_cancelled(logger, order.symbol, f"exchange {order.status}")
            return order

        # No order ID — something went wrong
        if not order.order_id:
            order.status = "rejected"
            log_order_cancelled(logger, order.symbol, "no order ID returned")
            return order

        # Poll until resolution
        start = time.time()
        logger.debug("[%s] Tracking order %s — polling every %.1fs, timeout %.1fs",
                     order.symbol, order.order_id, self.poll_interval_s, self.timeout_s)

        while True:
            elapsed = time.time() - start

            # Timeout — cancel the order
            if elapsed >= self.timeout_s:
                logger.warning("[%s] Order %s timed out after %.0fs — cancelling",
                               order.symbol, order.order_id, elapsed)
                try:
                    cancelled = await self.exchange.cancel_order(order.order_id, order.symbol)
                    if cancelled:
                        order.status = "cancelled"
                        log_order_cancelled(logger, order.symbol,
                                            f"timeout after {elapsed:.0f}s")
                    else:
                        # Cancel failed — order might have filled between check and cancel
                        status = await self.exchange.get_order_status(order.order_id, order.symbol)
                        order.status = status.status
                        order.filled_size = status.filled_size
                        order.fill_price = status.fill_price
                except Exception as e:
                    logger.error("[%s] Failed to cancel order %s: %s",
                                 order.symbol, order.order_id, e)
                    order.status = "cancelled"
                return order

            # Poll exchange
            await asyncio.sleep(self.poll_interval_s)

            try:
                status = await self.exchange.get_order_status(order.order_id, order.symbol)
            except Exception as e:
                logger.warning("[%s] Failed to poll order %s: %s — retrying",
                               order.symbol, order.order_id, e)
                continue

            if status.status == "filled":
                order.status = "filled"
                order.filled_size = status.filled_size or order.amount
                order.fill_price = status.fill_price or order.price or 0.0
                log_order_filled(logger, order.symbol, order.side,
                                 order.filled_size, order.fill_price)
                return order

            if status.status in ("cancelled", "rejected"):
                order.status = status.status
                log_order_cancelled(logger, order.symbol, status.status)
                return order

            # Still open — log progress at debug level
            logger.debug("[%s] Order %s still open (%.0fs elapsed)",
                         order.symbol, order.order_id, elapsed)
