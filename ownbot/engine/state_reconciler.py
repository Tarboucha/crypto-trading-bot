"""State reconciler — verifies bot state matches exchange reality.

Two stages:
1. Startup: load from DB + verify against exchange before first tick
2. Continuous: periodic check on tick.start, fixes mismatches

Inspired by NautilusTrader's reconciliation system.
"""
import asyncio
import logging
import time

from shared.events.component import Component
from shared.events.trading import PositionOpened, PositionClosed
from shared.events.system import ReconcileMismatch
from shared.api.db.trade_repo import TradeRepo
from ownbot.engine.position_manager import PositionManager

logger = logging.getLogger(__name__)


class StateReconciler(Component):

    def __init__(
        self,
        exchange,
        positions: PositionManager,
        bus,
        interval_s: float = 60.0,
        startup_delay_s: float = 5.0,
    ):
        self.exchange = exchange
        self.positions = positions
        self.bus = bus
        self.trade_repo = TradeRepo()
        self.interval_s = interval_s
        self.startup_delay_s = startup_delay_s
        self._last_reconcile_ts: float = 0

    # --- Stage 1: Startup reconciliation ---

    async def startup_reconcile(self) -> list[str]:
        """Full reconcile on startup.
        1. Load open trades from DB (full metadata)
        2. Verify against exchange (source of truth)
        """
        actions = []

        # Step 1: Load from DB — gives us strategy, SL/TP, funding history
        db_trades = await self.trade_repo.get_open_trades()
        if db_trades:
            logger.info("Startup: loading %d open positions from DB...", len(db_trades))

        for t in db_trades:
            pair = t["pair"]
            if self.positions.has_position(pair):
                continue

            self.positions.open(
                pair=pair,
                direction=t["direction"],
                entry_price=t["entry_price"],
                size=t["size"],
                entry_time=t["entry_time"],
                strategy=t["strategy"],
                stoploss=t["stoploss"],
                take_profit=t["take_profit"],
            )
            # Restore funding state
            pos = self.positions.get_position(pair)
            pos.cumulative_funding = t["cumulative_funding"]
            pos.funding_events = t["funding_events"]

            logger.info(
                "[%s] Loaded from DB: %s %.6f @ %.2f (strategy=%s, SL=%s, TP=%s, funding=$%.4f)",
                pair, t["direction"], t["size"], t["entry_price"],
                t["strategy"], t["stoploss"], t["take_profit"], t["cumulative_funding"],
            )
            actions.append(f"loaded_db:{pair}")

        # Step 2: Verify against exchange
        if self.exchange:
            logger.info("Startup: verifying against exchange...")
            await self._reconcile()

        if not actions:
            logger.info("Startup reconciliation: no open positions.")

        # Wait for stabilization
        if self.startup_delay_s > 0:
            logger.debug("Waiting %.1fs for stabilization...", self.startup_delay_s)
            await asyncio.sleep(self.startup_delay_s)

        self._last_reconcile_ts = time.time()
        return actions

    # --- Stage 2: Continuous reconciliation ---

    async def on_tick_start(self, event) -> None:
        """Subscribed to tick.start — runs periodic reconciliation."""
        if not self.exchange:
            return

        now = time.time()
        if now - self._last_reconcile_ts < self.interval_s:
            return

        await self._reconcile()
        self._last_reconcile_ts = now

    async def _reconcile(self) -> None:
        """Compare bot state with exchange, fix mismatches."""
        try:
            exchange_positions = await self.exchange.get_positions()
        except Exception as e:
            logger.warning("Reconciliation failed — couldn't reach exchange: %s", e)
            return

        exchange_map = {p.symbol: p for p in exchange_positions if p.size != 0}
        bot_pairs = set(self.positions.open_positions.keys())
        exchange_pairs = set(exchange_map.keys())

        # Bot has position, exchange doesn't → phantom
        for pair in bot_pairs - exchange_pairs:
            logger.warning("[%s] Phantom position — not on exchange, closing", pair)

            try:
                ticker = await self.exchange.get_ticker(pair)
                exit_price = ticker.last
            except Exception:
                exit_price = 0.0

            closed = self.positions.close(
                pair, exit_price, int(time.time() * 1000),
                "reconciled: not on exchange",
            )

            await self.bus.publish(PositionClosed(
                pair=closed.pair, direction=closed.direction,
                entry_price=closed.entry_price, exit_price=closed.exit_price,
                size=closed.size, profit_pct=closed.profit_pct,
                profit_abs=closed.profit_abs, funding_pnl=closed.funding_pnl,
                entry_time=closed.entry_time, reason=closed.reason,
                strategy=closed.strategy,
            ))
            await self.bus.publish(ReconcileMismatch(
                pair=pair, mismatch_type="phantom",
                details="Position in bot but not on exchange",
            ))

        # Exchange has position, bot doesn't → untracked
        for pair in exchange_pairs - bot_pairs:
            pos = exchange_map[pair]
            logger.warning("[%s] Untracked position on exchange — recovering", pair)

            self.positions.open(
                pair=pair, direction=pos.side,
                entry_price=pos.entry_price, size=pos.size,
                entry_time=int(time.time() * 1000), strategy="recovered",
            )

            await self.bus.publish(PositionOpened(
                pair=pair, direction=pos.side,
                entry_price=pos.entry_price, size=pos.size,
                strategy="recovered",
            ))
            await self.bus.publish(ReconcileMismatch(
                pair=pair, mismatch_type="untracked",
                details=f"Position on exchange not in bot: {pos.side} {pos.size} @ {pos.entry_price}",
            ))

        # Both have but size differs
        for pair in bot_pairs & exchange_pairs:
            bot_pos = self.positions.get_position(pair)
            exch_pos = exchange_map[pair]
            if abs(bot_pos.size - exch_pos.size) > 0.0001:
                logger.warning(
                    "[%s] Size mismatch: bot=%.6f exchange=%.6f — updating to exchange",
                    pair, bot_pos.size, exch_pos.size,
                )
                bot_pos.size = exch_pos.size

                await self.bus.publish(ReconcileMismatch(
                    pair=pair, mismatch_type="size_diff",
                    details=f"Bot={bot_pos.size} Exchange={exch_pos.size}. Updated to exchange.",
                ))
