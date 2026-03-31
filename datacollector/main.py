"""Data Collector — standalone program that fetches and stores candle data."""
import argparse
import asyncio
import logging
import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

from datacollector.collector import CandleCollector, TIMEFRAME_SECONDS
from shared.db.engine import init_db

logger = logging.getLogger("datacollector")

DEFAULT_CONFIG = str(Path(__file__).parent / "config.toml")


def setup_logging(verbosity: int = 1) -> None:
    from rich.logging import RichHandler

    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbosity, logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(name)-40s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OwnBot Data Collector")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity (-v=INFO, -vv=DEBUG)"
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Config file path")
    parser.add_argument(
        "--once", action="store_true", help="Run once and exit (don't loop)"
    )
    return parser.parse_args()


async def run() -> None:
    args = parse_args()

    # Load .env
    env_path = Path(__file__).parent.parent / "ownbot" / ".env"
    load_dotenv(env_path)

    # Load config
    config = load_config(args.config)
    verbosity = args.verbose if args.verbose else config.get("logging", {}).get("verbosity", 1)
    setup_logging(verbosity)

    pairs = config["exchange"]["pairs"]
    timeframes = config["exchange"]["timeframes"]
    initial_limit = config["exchange"].get("initial_limit", 500)

    logger.info("Data Collector starting...")
    logger.info("Pairs: %s | Timeframes: %s", pairs, timeframes)

    # Init DB tables
    await init_db()

    # Create collector
    collector = CandleCollector(
        pairs=pairs,
        timeframes=timeframes,
        initial_limit=initial_limit,
    )
    await collector.connect()

    try:
        # Initial backfill
        logger.info("Running initial backfill...")
        for pair in pairs:
            for tf in timeframes:
                await collector.backfill(pair, tf)
        logger.info("Backfill complete.")

        if args.once:
            logger.info("--once flag set. Exiting.")
            return

        # Continuous collection loop
        # Poll at the rate of the smallest timeframe
        min_tf = min(timeframes, key=lambda t: TIMEFRAME_SECONDS[t])
        poll_interval = TIMEFRAME_SECONDS[min_tf]
        logger.info("Polling every %ds (based on %s timeframe).", poll_interval, min_tf)

        while True:
            await asyncio.sleep(poll_interval)
            count = await collector.run_once()
            if count > 0:
                logger.info("Collected %d new candle(s).", count)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await collector.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
