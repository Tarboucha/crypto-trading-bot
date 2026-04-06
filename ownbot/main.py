import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from ownbot.config import load_config
from ownbot.loggers import setup_logging
from shared.api.exchange.hyperliquid import HyperliquidExchange
from shared.events.bus import EventBus
from shared.events.command_queue import CommandQueue
from shared.costs import TradingCosts
from shared.db.engine import init_db

from ownbot.engine.engine import TradingEngine
from ownbot.engine.position_manager import PositionManager
from ownbot.engine.risk_manager import RiskManager
from ownbot.engine.session_tracker import SessionTracker
from ownbot.engine.funding_tracker import FundingTracker
from ownbot.engine.trade_persister import TradePersister
from ownbot.engine.state_reconciler import StateReconciler

from ownbot.strategy.trend_follow import TrendFollowStrategy
from ownbot.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from ownbot.strategy.ml_filtered_rsi import MLFilteredRSIStrategy
from ownbot.strategy.momentum_breakout import MomentumBreakoutStrategy
from ownbot.strategy.macro_mr import MacroMRStrategy
from ownbot.strategy.momentum_hysteresis import MomentumHysteresisStrategy
from ownbot.strategy.kronos_strategy import KronosStrategy
from ownbot.strategy.kronos_forecast_strategy import KronosForecastStrategy

STRATEGIES = {
    "trend_follow": TrendFollowStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "ml_filtered_rsi": MLFilteredRSIStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
    "macro_mr": MacroMRStrategy,
    "momentum_hysteresis": MomentumHysteresisStrategy,
    "kronos": KronosStrategy,
    "kronos_forecast": KronosForecastStrategy,
}

logger = logging.getLogger("ownbot")
DEFAULT_CONFIG = str(Path(__file__).parent / "config.toml")


def parse_args() -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="count", default=0)
    common.add_argument("--logfile", type=str, default=None)
    common.add_argument("--config", type=str, default=DEFAULT_CONFIG)

    parser = argparse.ArgumentParser(description="OwnBot Trading Bot", parents=[common])
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("trade", parents=[common], help="Run the trading bot")

    bt = subparsers.add_parser("backtest", parents=[common], help="Backtest a strategy")
    bt.add_argument("--days", type=int, default=None)
    bt.add_argument("--start", type=str, default=None)
    bt.add_argument("--end", type=str, default=None)
    bt.add_argument("--pairs", nargs="+", default=None)
    bt.add_argument("--timeframe", type=str, default=None)
    bt.add_argument("--param", action="append", default=[])
    bt.add_argument("--balance", type=float, default=10000.0)
    bt.add_argument("--strategy", type=str, default=None)
    bt.add_argument("--source", type=str, default="csv", choices=["csv", "db"])
    bt.add_argument("--data-dir", type=str, default=None)

    return parser.parse_args()


def _get_risk_config(config):
    return {
        "max_open_trades": config.strategy.params.get("max_open_trades", 3),
        "risk_per_trade_pct": config.risk.risk_per_trade_pct,
        "max_exposure_pct": config.risk.max_exposure_pct,
        "loss_limit_pct": config.risk.loss_limit_pct,
        "loss_limit_reset": config.risk.loss_limit_reset,
        "max_drawdown_pct": config.risk.max_drawdown_pct,
        "max_leverage": config.risk.max_leverage,
        "liquidation_buffer": config.risk.liquidation_buffer,
    }


def _get_strategy(name: str, params: dict):
    cls = STRATEGIES.get(name)
    if not cls:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return cls(params=params)


async def run_trade(args, config) -> None:
    """Run the live/paper trading bot."""
    logger.info("OwnBot starting up...")
    logger.info("Mode: %s | Pairs: %s", config.mode, config.pairs)

    await init_db()

    exchange = HyperliquidExchange(config.exchange)
    await exchange.connect()

    try:
        bus = EventBus()
        cmd_queue = CommandQueue()
        positions = PositionManager()

        strategy = _get_strategy(config.strategy.name, config.strategy.params)
        tracker = SessionTracker()
        risk = RiskManager(**_get_risk_config(config))
        funding = FundingTracker(exchange=exchange, positions=positions, bus=bus)
        persister = TradePersister(mode=config.mode)

        reconciler = StateReconciler(
            exchange=exchange, positions=positions, bus=bus,
            interval_s=60.0, startup_delay_s=5.0,
        )

        tracker.register(bus)
        risk.register(bus)
        funding.register(bus)
        persister.register(bus)
        reconciler.register(bus)

        # Startup reconciliation — recover positions from exchange
        if config.mode == "live":
            await reconciler.startup_reconcile()

        engine = TradingEngine(
            strategy=strategy,
            bus=bus,
            positions=positions,
            mode=config.mode,
            exchange=exchange if config.mode == "live" else None,
            exchange_name=config.exchange.name,
            risk_config=_get_risk_config(config),
            cmd_queue=cmd_queue,
        )
        await engine.run(pairs=config.pairs)
    finally:
        await exchange.close()


async def run_backtest(args, config) -> None:
    """Run a backtest."""
    from ownbot.backtester.runner import Backtester
    from ownbot.backtester.report import print_report
    from shared.pairs import parse_pairs

    if args.source == "db":
        await init_db()

    strategy_name = args.strategy or config.strategy.name

    # Load params from the strategy's own TOML config
    import tomllib
    strategy_toml = Path(__file__).parent / "strategy" / "configs" / f"{strategy_name}.toml"
    if strategy_toml.exists():
        with open(strategy_toml, "rb") as f:
            params = tomllib.load(f)
    else:
        params = dict(config.strategy.params)

    # CLI --param overrides
    for p in args.param:
        key, value = p.split("=", 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        params[key] = value

    pairs = parse_pairs(args.pairs, trading_mode="futures") if args.pairs else config.pairs
    timeframe = args.timeframe or config.timeframe
    params["timeframe"] = timeframe

    strategy = _get_strategy(strategy_name, params)

    costs = TradingCosts(
        fee_pct=config.costs.fee_pct / 100,
        slippage_pct=config.costs.slippage_pct / 100,
        spread_mode=config.costs.spread_mode,
        spread_fixed_pct=config.costs.spread_fixed_pct / 100,
        spread_factor=config.costs.spread_factor,
    )

    backtester = Backtester(
        strategy=strategy,
        pairs=pairs,
        timeframe=timeframe,
        initial_balance=args.balance,
        risk_config=_get_risk_config(config),
        data_source=args.source,
        data_dir=args.data_dir,
        costs=costs,
    )

    if not args.days and not args.start:
        args.days = 7

    result = await backtester.run(
        start_date=args.start,
        end_date=args.end,
        days=args.days,
    )
    print_report(result)


async def run() -> None:
    args = parse_args()
    config = load_config(args.config)

    verbosity = args.verbose if args.verbose else config.logging.verbosity
    setup_logging(verbosity=verbosity, logfile=args.logfile or config.logging.logfile)

    command = args.command or "trade"

    if command == "backtest":
        await run_backtest(args, config)
    else:
        await run_trade(args, config)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
