"""Trading engine — the main loop. Publishes events, never calls components directly."""
import asyncio
import logging
import signal
import time
import traceback

import pandas as pd

from shared.api.db.candle_repo import CandleRepo
from shared.events.bus import EventBus
from shared.events.command_queue import CommandQueue
from shared.events.trading import (
    SignalEntry, SignalExit, SignalRejected,
    OrderSubmitted, OrderFilled, OrderCancelled,
    PositionOpened, PositionClosed, StoplossHit, TakeprofitHit,
)
from shared.events.system import TickStart, TickComplete, EngineStarted, EngineStopped, StrategyError
from shared.events.system import ExchangeError as ExchangeErrorEvent
from shared.api.errors import PermanentError, RetryableError
from shared.events.commands import StopCommand, ForceCloseCommand, PauseCommand, ResumeCommand
from shared.events.component import Component
from ownbot.strategy.base import BaseStrategy
from ownbot.engine.position_manager import PositionManager
from ownbot.engine.risk_manager import RiskManager
from ownbot.engine.executor import PaperExecutor, LiveExecutor

logger = logging.getLogger(__name__)

TIMEFRAME_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


class TradingEngine(Component):
    def __init__(
        self,
        strategy: BaseStrategy,
        bus: EventBus,
        positions: PositionManager,
        mode: str = "paper",
        exchange=None,
        exchange_name: str = "hyperliquid",
        risk_config: dict | None = None,
        cmd_queue: CommandQueue | None = None,
    ):
        self.strategy = strategy
        self.bus = bus
        self.positions = positions
        self.mode = mode
        self.exchange = exchange
        self.exchange_name = exchange_name
        self.cmd_queue = cmd_queue

        self.candle_repo = CandleRepo()

        rc = risk_config or {}
        self.risk = RiskManager(
            max_open_trades=rc.get("max_open_trades", 3),
            risk_per_trade_pct=rc.get("risk_per_trade_pct", 0.3),
            max_exposure_pct=rc.get("max_exposure_pct", 4.0),
            loss_limit_pct=rc.get("loss_limit_pct", 2.0),
            loss_limit_reset=rc.get("loss_limit_reset", "daily"),
            max_drawdown_pct=rc.get("max_drawdown_pct", 8.0),
        )

        if mode == "live" and exchange:
            self.executor = LiveExecutor(exchange)
        else:
            self.executor = PaperExecutor()

        self._running = False
        self._paused = False
        self._tick_count = 0

    # --- Command handlers (subscribed via Component.register) ---

    async def on_command_stop(self, event: StopCommand):
        logger.info("Stop command received: %s", event.reason)
        self._running = False

    async def on_command_pause(self, event: PauseCommand):
        logger.info("Paused: %s", event.reason)
        self._paused = True

    async def on_command_resume(self, event: ResumeCommand):
        logger.info("Resumed")
        self._paused = False

    async def on_command_force_close(self, event: ForceCloseCommand):
        if not self.positions.has_position(event.pair):
            logger.warning("[%s] Force close — no position", event.pair)
            return

        current_price = await self.get_current_price(event.pair)
        current_time = int(time.time() * 1000)
        closed = self.positions.close(event.pair, current_price, current_time, event.reason)

        await self.bus.publish(PositionClosed(
            pair=closed.pair, direction=closed.direction,
            entry_price=closed.entry_price, exit_price=closed.exit_price,
            size=closed.size, profit_pct=closed.profit_pct,
            profit_abs=closed.profit_abs, funding_pnl=closed.funding_pnl,
            entry_time=closed.entry_time, reason=closed.reason,
            strategy=closed.strategy,
        ))

    # --- Data helpers ---

    async def fetch_candles(self, pair: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        return await self.candle_repo.get_candles(pair, timeframe, limit=limit)

    async def get_current_price(self, pair: str, exchange_symbol: str | None = None) -> float:
        if self.exchange:
            ticker = await self.exchange.get_ticker(exchange_symbol or pair)
            return ticker.last

        df = await self.fetch_candles(pair, self.strategy.timeframes[0], limit=1)
        if len(df) > 0:
            return float(df.iloc[-1]["close"])
        return 0.0

    async def get_balance(self) -> float:
        if self.exchange and self.mode == "live":
            balance = await self.exchange.get_balance()
            return balance.total
        return 10000.0

    # --- Main tick ---

    async def tick(self, pairs) -> None:
        """Single engine tick: process commands, check positions, evaluate strategy."""
        self._tick_count += 1
        tick_start = time.time()

        # Process pending commands from REST API
        if self.cmd_queue:
            await self.cmd_queue.process(self.bus)

        await self.bus.publish(TickStart(tick_number=self._tick_count))

        if self._paused:
            logger.debug("Tick %d skipped (paused)", self._tick_count)
            return

        balance = await self.get_balance()
        self.risk.update_peak_balance(balance)

        for trading_pair in pairs:
            pair = trading_pair.symbol if hasattr(trading_pair, "symbol") else str(trading_pair)
            exchange_symbol = trading_pair.exchange_id(self.exchange_name) if hasattr(trading_pair, "exchange_id") else pair

            try:
                current_price = await self.get_current_price(pair, exchange_symbol)
                current_time = int(time.time() * 1000)

                # 1. Check stoploss/takeprofit
                if self.positions.has_position(pair):
                    closed = self.positions.check_stoploss_takeprofit(pair, current_price, current_time)
                    if closed:
                        event_cls = StoplossHit if "stoploss" in closed.reason else TakeprofitHit
                        await self.bus.publish(event_cls(
                            pair=closed.pair, direction=closed.direction,
                            entry_price=closed.entry_price, exit_price=closed.exit_price,
                            profit_pct=closed.profit_pct, profit_abs=closed.profit_abs,
                        ))
                        await self.bus.publish(PositionClosed(
                            pair=closed.pair, direction=closed.direction,
                            entry_price=closed.entry_price, exit_price=closed.exit_price,
                            size=closed.size, profit_pct=closed.profit_pct,
                            profit_abs=closed.profit_abs, funding_pnl=closed.funding_pnl,
                            reason=closed.reason, strategy=closed.strategy,
                        ))
                        continue

                # 2. Fetch candles
                data = {}
                limit = self.strategy.startup_candle_count
                for tf in self.strategy.timeframes:
                    data[tf] = await self.fetch_candles(pair, tf, limit=limit)

                for extra in self.strategy.required_pairs():
                    if extra != pair:
                        for tf in self.strategy.timeframes:
                            data[f"{extra}_{tf}"] = await self.fetch_candles(extra, tf, limit=limit)

                # 3. Evaluate strategy (isolated — strategy errors don't crash engine)
                try:
                    strat_signal = self.strategy.evaluate(pair, data, has_position=self.positions.has_position(pair))
                except Exception as e:
                    logger.error("[%s] Strategy error: %s — skipping", pair, e)
                    await self.bus.publish(StrategyError(
                        pair=pair, strategy=self.strategy.name, error=str(e),
                    ))
                    continue
                if not strat_signal:
                    continue

                # 4. Publish signal event
                signal_ts = float(strat_signal.timestamp / 1000) if strat_signal.timestamp else 0.0
                if strat_signal.action == "enter":
                    await self.bus.publish(SignalEntry(
                        pair=pair, direction=strat_signal.direction,
                        confidence=strat_signal.confidence, reason=strat_signal.reason,
                        strategy=self.strategy.name, timeframe=strat_signal.timeframe,
                        timestamp=signal_ts,
                    ))
                else:
                    await self.bus.publish(SignalExit(
                        pair=pair, direction=strat_signal.direction,
                        confidence=strat_signal.confidence, reason=strat_signal.reason,
                        strategy=self.strategy.name,
                        timestamp=signal_ts,
                    ))

                # 5. Act on signal
                if strat_signal.action == "enter" and not self.positions.has_position(pair):
                    allowed, reject_reason = self.risk.can_trade(strat_signal, balance, self.positions)
                    if not allowed:
                        await self.bus.publish(SignalRejected(
                            pair=pair, direction=strat_signal.direction, reason=reject_reason,
                        ))
                        continue

                    size = self.risk.calculate_position_size(balance, current_price)

                    side = "buy" if strat_signal.direction == "long" else "sell"
                    await self.bus.publish(OrderSubmitted(
                        pair=pair, side=side, order_type="market",
                        size=size, price=current_price,
                    ))

                    result = await self.executor.execute(strat_signal, current_price, size)

                    if result.success:
                        await self.bus.publish(OrderFilled(
                            pair=pair, side=result.side, size=result.size,
                            price=result.price, order_id=result.order_id,
                        ))

                        # Use dynamic SL/TP if strategy provides them, else compute from config
                        dynamic_sl = self.strategy.params.pop("_dynamic_sl", None)
                        dynamic_tp = self.strategy.params.pop("_dynamic_tp", None)

                        if dynamic_sl is not None and dynamic_tp is not None:
                            sl = dynamic_sl
                            tp = dynamic_tp
                        else:
                            sl_pct = self.strategy.params.get("stoploss", -3.0) / 100
                            tp_pct = self.strategy.params.get("take_profit", 6.0) / 100
                            if strat_signal.direction == "long":
                                sl = result.price * (1 + sl_pct)
                                tp = result.price * (1 + tp_pct)
                            else:
                                sl = result.price * (1 - sl_pct)
                                tp = result.price * (1 - tp_pct)

                        trailing = self.strategy.params.get("trailing_stop", False)
                        trailing_dist = self.strategy.params.get("trailing_stop_distance", 1.5)
                        trailing_activate = self.strategy.params.get("trailing_stop_activate", 0.0)

                        self.positions.open(
                            pair=pair, direction=strat_signal.direction,
                            entry_price=result.price, size=result.size,
                            entry_time=current_time, strategy=self.strategy.name,
                            stoploss=sl, take_profit=tp,
                            trailing_stop=trailing,
                            trailing_distance_pct=trailing_dist,
                            trailing_activate_pct=trailing_activate,
                        )

                        await self.bus.publish(PositionOpened(
                            pair=pair, direction=strat_signal.direction,
                            entry_price=result.price, size=result.size,
                            stoploss=sl, take_profit=tp, strategy=self.strategy.name,
                        ))
                    else:
                        await self.bus.publish(OrderCancelled(
                            pair=pair, order_id=result.order_id, reason="not filled",
                        ))

                elif strat_signal.action == "exit" and self.positions.has_position(pair):
                    pos = self.positions.get_position(pair)
                    if pos.direction != strat_signal.direction:
                        continue

                    exit_side = "sell" if strat_signal.direction == "long" else "buy"
                    await self.bus.publish(OrderSubmitted(
                        pair=pair, side=exit_side, order_type="market",
                        size=pos.size, price=current_price,
                    ))

                    result = await self.executor.execute(strat_signal, current_price, pos.size)
                    if result.success:
                        await self.bus.publish(OrderFilled(
                            pair=pair, side=result.side, size=result.size,
                            price=result.price, order_id=result.order_id,
                        ))

                        closed = self.positions.close(pair, result.price, current_time, strat_signal.reason)

                        await self.bus.publish(PositionClosed(
                            pair=closed.pair, direction=closed.direction,
                            entry_price=closed.entry_price, exit_price=closed.exit_price,
                            size=closed.size, profit_pct=closed.profit_pct,
                            profit_abs=closed.profit_abs, funding_pnl=closed.funding_pnl,
                            reason=closed.reason, strategy=closed.strategy,
                        ))

            except PermanentError as e:
                logger.error("[%s] Permanent error: %s", pair, e)
                await self.bus.publish(ExchangeErrorEvent(
                    pair=pair, error=str(e), retryable=False,
                ))
            except RetryableError as e:
                logger.warning("[%s] Retries exhausted: %s", pair, e)
                await self.bus.publish(ExchangeErrorEvent(
                    pair=pair, error=str(e), retryable=True,
                ))
            except Exception as e:
                logger.error("[%s] Unexpected error: %s", pair, e)
                logger.debug(traceback.format_exc())

        elapsed = time.time() - tick_start
        await self.bus.publish(TickComplete(
            tick_number=self._tick_count,
            pairs_processed=len(pairs),
            elapsed_s=elapsed,
        ))

    # --- Main loop ---

    async def run(self, pairs) -> None:
        """Main loop: tick every candle period."""
        min_tf = min(self.strategy.timeframes, key=lambda t: TIMEFRAME_SECONDS[t])
        interval = TIMEFRAME_SECONDS[min_tf]

        # Bind command queue to this event loop
        if self.cmd_queue:
            self.cmd_queue.bind_loop(asyncio.get_running_loop())

        # Register command handlers
        self.register(self.bus)

        pair_strs = [str(p) for p in pairs]
        await self.bus.publish(EngineStarted(
            mode=self.mode, strategy=self.strategy.name,
            pairs=tuple(pair_strs),
        ))

        logger.info(
            "Engine started — mode=%s, strategy=%s, pairs=%s, interval=%ds",
            self.mode, self.strategy.name, pair_strs, interval,
        )

        # Signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal, sig)

        self._running = True
        stop_reason = "user"
        try:
            await self.tick(pairs)
            while self._running:
                await asyncio.sleep(interval)
                await self.tick(pairs)
        except KeyboardInterrupt:
            stop_reason = "user (ctrl+c)"
        except asyncio.CancelledError:
            stop_reason = "cancelled"
        except Exception as e:
            stop_reason = f"error: {e}"
            logger.error("Engine crashed: %s", e)
        finally:
            self._running = False
            await self.bus.publish(EngineStopped(reason=stop_reason))
            logger.info("Engine shut down (%s).", stop_reason)

    def _handle_signal(self, sig) -> None:
        sig_name = signal.Signals(sig).name
        logger.info("Received %s — shutting down gracefully...", sig_name)
        self._running = False

    def stop(self) -> None:
        self._running = False
