"""Fetches funding rates and applies to open positions — subscribes to tick events."""
import logging
import time

from shared.events.component import Component
from shared.events.trading import FundingApplied
from ownbot.engine.position_manager import PositionManager

logger = logging.getLogger(__name__)

DEFAULT_SETTLEMENT_INTERVAL_S = 3600  # Hyperliquid: 1 hour


class FundingTracker(Component):
    def __init__(self, exchange=None, positions: PositionManager | None = None,
                 bus=None, fetch_interval_s: int = 3600):
        self.exchange = exchange
        self.positions = positions
        self.bus = bus
        self.fetch_interval_s = fetch_interval_s
        self._rates: dict[str, float] = {}
        self._last_fetch_ts: float = 0
        self._last_settlement_ts: float = 0

    async def on_tick_start(self, event) -> None:
        """Called every tick via event bus."""
        if not self.positions:
            return

        now = time.time()

        if now - self._last_fetch_ts >= self.fetch_interval_s:
            await self._fetch_rates()
            self._last_fetch_ts = now

        if now - self._last_settlement_ts >= DEFAULT_SETTLEMENT_INTERVAL_S:
            await self._apply_funding()
            self._last_settlement_ts = now

    async def _fetch_rates(self) -> None:
        if not self.exchange:
            return
        for pair in list(self.positions.open_positions.keys()):
            try:
                rate = await self.exchange.get_funding_rate(pair)
                self._rates[pair] = rate
            except Exception as e:
                logger.warning("[%s] Failed to fetch funding rate: %s", pair, e)

    async def _apply_funding(self) -> None:
        for pair in list(self.positions.open_positions.keys()):
            rate = self._rates.get(pair, 0.0)
            if rate == 0.0:
                continue
            amount = self.positions.apply_funding(pair, rate)
            if amount is not None and self.bus and abs(amount) > 0.001:
                pos = self.positions.get_position(pair)
                await self.bus.publish(FundingApplied(
                    pair=pair, rate=rate, amount=amount,
                    cumulative=pos.cumulative_funding if pos else 0.0,
                ))

    def get_rate(self, pair: str) -> float:
        return self._rates.get(pair, 0.0)

    def set_rate(self, pair: str, rate: float) -> None:
        self._rates[pair] = rate
