"""Backtester — replays historical candles through a strategy."""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from shared.api.db.candle_repo import CandleRepo
from ownbot.strategy.base import BaseStrategy
from ownbot.engine.position_manager import PositionManager, ClosedTrade
from ownbot.engine.risk_manager import RiskManager
from ownbot.backtester.result import BacktestResult
from shared.costs import TradingCosts

logger = logging.getLogger(__name__)

# Default path to CSV data
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "binance"


class Backtester:
    def __init__(
        self,
        strategy: BaseStrategy,
        pairs,
        timeframe: str,
        initial_balance: float = 10000.0,
        risk_config: dict | None = None,
        data_source: str = "csv",  # "csv" or "db"
        data_dir: str | None = None,
        costs: TradingCosts | None = None,
    ):
        self.strategy = strategy
        self.pairs = pairs  # list[TradingPair] or list[str]
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.data_source = data_source
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.costs = costs or TradingCosts()

        rc = risk_config or {}
        self.risk = RiskManager(
            max_open_trades=rc.get("max_open_trades", 3),
            risk_per_trade_pct=rc.get("risk_per_trade_pct", 0.3),
            max_exposure_pct=rc.get("max_exposure_pct", 4.0),
            loss_limit_pct=rc.get("loss_limit_pct", 2.0),
            loss_limit_reset=rc.get("loss_limit_reset", "daily"),
            max_drawdown_pct=rc.get("max_drawdown_pct", 8.0),
            max_leverage=rc.get("max_leverage", 1.0),
            liquidation_buffer=rc.get("liquidation_buffer", 0.05),
        )

    async def fetch_candles(
        self, pair, timeframe: str, start_ms: int | None = None, end_ms: int | None = None
    ) -> pd.DataFrame:
        """Fetch candles from configured data source (csv or db).

        Args:
            pair: TradingPair object or string
        """
        # Extract the right identifier depending on source
        if hasattr(pair, 'csv_symbol'):
            csv_pair = pair.base  # CandleRepo.get_candles_csv uses PAIR_TO_SYMBOL internally
            db_pair = pair.base
        else:
            csv_pair = pair
            db_pair = pair

        if self.data_source == "csv":
            return CandleRepo.get_candles_csv(csv_pair, timeframe, self.data_dir, start_ms, end_ms)
        else:
            return await CandleRepo.get_candles(db_pair, timeframe, start_ms=start_ms, end_ms=end_ms)

    async def run(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        days: int | None = None,
        window: int = 100,
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            start_date: "YYYY-MM-DD" start date
            end_date: "YYYY-MM-DD" end date
            days: Alternative to start/end — last N days
            window: Number of candles to feed the strategy (rolling window)
        """
        # Calculate time range
        now_ms = int(time.time() * 1000)

        if days:
            start_ms = now_ms - (days * 86400 * 1000)
            end_ms = now_ms
            start_date = start_date or datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            end_date = end_date or datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) if start_date else None
            end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) if end_date else now_ms
            start_date = start_date or "earliest"
            end_date = end_date or datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        logger.info(
            "Backtesting %s on %s (%s) from %s — %s to %s",
            self.strategy.name, self.pairs, self.timeframe, self.data_source, start_date, end_date,
        )

        all_trades: list[ClosedTrade] = []
        all_fees: float = 0.0

        # Use strategy's startup_candle_count if larger than default window
        effective_window = max(window, self.strategy.startup_candle_count)

        for pair in self.pairs:
            trades, fees = await self._run_pair(pair, start_ms, end_ms, effective_window)
            all_trades.extend(trades)
            all_fees += fees
            logger.info("%s — %d trades, $%.2f fees", pair, len(trades), fees)

        all_trades.sort(key=lambda t: t.entry_time)

        result = BacktestResult.from_trades(
            trades=all_trades,
            initial_balance=self.initial_balance,
            start_date=start_date,
            end_date=end_date,
            strategy=self.strategy.name,
            pairs=[str(p) for p in self.pairs],
            timeframe=self.timeframe,
            params=self.strategy.params,
        )
        result.total_fees = all_fees

        return result

    async def _run_pair(
        self, trading_pair, start_ms: int | None, end_ms: int | None, window: int
    ) -> tuple[list[ClosedTrade], float]:
        """Run backtest for a single pair."""
        # Extract string name for strategies/positions
        pair = trading_pair.symbol if hasattr(trading_pair, 'symbol') else str(trading_pair)

        df = await self.fetch_candles(trading_pair, self.timeframe, start_ms, end_ms)

        if len(df) < window + 1:
            logger.warning(
                "%s — not enough candles (%d < %d). Skipping.",
                pair, len(df), window + 1,
            )
            return [], 0.0

        logger.info("%s — %d candles loaded.", pair, len(df))

        # Pre-fetch extra pairs the strategy needs
        extra_dfs: dict[str, pd.DataFrame] = {}
        for extra in self.strategy.required_pairs():
            if extra != pair:
                extra_df = await self.fetch_candles(extra, self.timeframe, start_ms, end_ms)
                if len(extra_df) > 0:
                    extra_dfs[extra] = extra_df
                    logger.info("%s — %d extra candles for %s.", pair, len(extra_df), extra)

        positions = PositionManager()
        trades: list[ClosedTrade] = []
        balance = self.initial_balance
        total_fees = 0.0

        for i in range(window, len(df)):
            chunk = df.iloc[i - window:i + 1].copy()
            candle = df.iloc[i]
            ts = int(candle["timestamp"])
            close = float(candle["close"])
            high = float(candle["high"])
            low = float(candle["low"])

            # 1. Check stoploss/takeprofit against candle high/low
            if positions.has_position(pair):
                pos = positions.get_position(pair)

                # Check stoploss — use low for long, high for short
                if pos.direction == "long" and pos.stoploss and low <= pos.stoploss:
                    exit_price = self.costs.apply_exit_price(pos.stoploss, "long", high, low)
                    fee = self.costs.fee_for_trade(exit_price, pos.size)
                    total_fees += fee
                    closed = positions.close(pair, exit_price, ts, "stoploss hit")
                    closed.profit_abs -= fee
                    closed.strategy = self.strategy.name
                    trades.append(closed)
                    balance += closed.profit_abs
                    continue
                elif pos.direction == "short" and pos.stoploss and high >= pos.stoploss:
                    exit_price = self.costs.apply_exit_price(pos.stoploss, "short", high, low)
                    fee = self.costs.fee_for_trade(exit_price, pos.size)
                    total_fees += fee
                    closed = positions.close(pair, exit_price, ts, "stoploss hit")
                    closed.profit_abs -= fee
                    closed.strategy = self.strategy.name
                    trades.append(closed)
                    balance += closed.profit_abs
                    continue

                # Check takeprofit — use high for long, low for short
                if pos.direction == "long" and pos.take_profit and high >= pos.take_profit:
                    exit_price = self.costs.apply_exit_price(pos.take_profit, "long", high, low)
                    fee = self.costs.fee_for_trade(exit_price, pos.size)
                    total_fees += fee
                    closed = positions.close(pair, exit_price, ts, "takeprofit hit")
                    closed.profit_abs -= fee
                    closed.strategy = self.strategy.name
                    trades.append(closed)
                    balance += closed.profit_abs
                    continue
                elif pos.direction == "short" and pos.take_profit and low <= pos.take_profit:
                    exit_price = self.costs.apply_exit_price(pos.take_profit, "short", high, low)
                    fee = self.costs.fee_for_trade(exit_price, pos.size)
                    total_fees += fee
                    closed = positions.close(pair, exit_price, ts, "takeprofit hit")
                    closed.profit_abs -= fee
                    closed.strategy = self.strategy.name
                    trades.append(closed)
                    balance += closed.profit_abs
                    continue

            # 2. Run strategy
            data = {self.timeframe: chunk}

            # Add extra pair data (sliced to same window)
            for extra, extra_df in extra_dfs.items():
                mask = extra_df["timestamp"] <= ts
                idx = mask.sum()
                if idx >= window:
                    data[f"{extra}_{self.timeframe}"] = extra_df.iloc[idx - window:idx + 1].copy()

            signal = self.strategy.evaluate(pair, data, has_position=positions.has_position(pair))
            if not signal:
                continue

            # 3. Act on signal
            if signal.action == "enter" and not positions.has_position(pair):
                # Apply costs to entry price
                entry_price = self.costs.apply_entry_price(close, signal.direction, high, low)

                # Leverage pipeline (mirrors engine.py)
                leverage = self.strategy.leverage(pair, signal.direction, data)
                leverage = self.risk.validate_leverage(leverage)
                size, margin = self.risk.calculate_position_size(balance, entry_price, leverage)

                # Check total exposure
                open_pos_list = list(positions.open_positions.values())
                if not self.risk.check_total_exposure(open_pos_list, size, entry_price, balance):
                    continue

                fee = self.costs.fee_for_trade(entry_price, size)
                total_fees += fee
                balance -= fee

                # Dynamic SL/TP or compute from config
                dynamic_sl = self.strategy.params.pop("_dynamic_sl", None)
                dynamic_tp = self.strategy.params.pop("_dynamic_tp", None)

                if dynamic_sl is not None and dynamic_tp is not None:
                    sl = dynamic_sl
                    tp = dynamic_tp
                else:
                    sl_pct = self.strategy.params.get("stoploss", -3.0) / 100
                    tp_pct = self.strategy.params.get("take_profit", 6.0) / 100
                    effective_sl_pct = self.risk.adjust_stoploss_for_leverage(sl_pct, leverage)
                    if signal.direction == "long":
                        sl = entry_price * (1 + effective_sl_pct)
                        tp = entry_price * (1 + tp_pct)
                    else:
                        sl = entry_price * (1 - effective_sl_pct)
                        tp = entry_price * (1 - tp_pct)

                # Liquidation protection
                liquidation_price = 0.0
                if leverage > 1.0:
                    liquidation_price = self.risk.calculate_liquidation_price(
                        entry_price, leverage, signal.direction,
                    )
                    liquidation_buffered = self.risk.apply_liquidation_buffer(
                        liquidation_price, entry_price,
                    )
                    sl = self.risk.stoploss_or_liquidation(sl, liquidation_buffered, signal.direction)
                    liquidation_price = liquidation_buffered

                positions.open(
                    pair=pair,
                    direction=signal.direction,
                    entry_price=entry_price,
                    size=size,
                    entry_time=ts,
                    strategy=self.strategy.name,
                    stoploss=sl,
                    take_profit=tp,
                    leverage=leverage,
                    margin=margin,
                    liquidation_price=liquidation_price,
                )

            elif signal.action == "exit" and positions.has_position(pair):
                pos = positions.get_position(pair)
                if pos.direction != signal.direction:
                    continue
                # Apply costs to exit price
                exit_price = self.costs.apply_exit_price(close, signal.direction, high, low)
                fee = self.costs.fee_for_trade(exit_price, pos.size)
                total_fees += fee
                closed = positions.close(pair, exit_price, ts, signal.reason)
                closed.profit_abs -= fee
                closed.strategy = self.strategy.name
                trades.append(closed)
                balance += closed.profit_abs

        # Close any remaining open positions at last candle close
        last = df.iloc[-1]
        last_close = float(last["close"])
        last_high = float(last["high"])
        last_low = float(last["low"])
        last_ts = int(last["timestamp"])
        for pair_key in list(positions.open_positions.keys()):
            pos = positions.get_position(pair_key)
            exit_price = self.costs.apply_exit_price(last_close, pos.direction, last_high, last_low)
            fee = self.costs.fee_for_trade(exit_price, pos.size)
            total_fees += fee
            closed = positions.close(pair_key, exit_price, last_ts, "backtest ended")
            closed.profit_abs -= fee
            closed.strategy = self.strategy.name
            trades.append(closed)

        logger.info("%s — total fees: $%.2f", pair, total_fees)
        return trades, total_fees
