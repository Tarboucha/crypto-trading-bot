"""Test: run TrendFollow strategy on candles stored in DB."""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "ownbot" / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select
from shared.db.engine import init_db, get_session
from shared.db.models import Candle
from ownbot.strategy.trend_follow import TrendFollowStrategy
import pandas as pd


async def main():
    await init_db()

    # Fetch ETH 5m candles from DB
    async with get_session() as session:
        result = await session.execute(
            select(Candle)
            .where(Candle.pair == "ETH", Candle.timeframe == "5m")
            .order_by(Candle.timestamp)
        )
        rows = result.scalars().all()

    print(f"Loaded {len(rows)} ETH/5m candles from DB.")

    # Convert to DataFrame
    df = pd.DataFrame([{
        "timestamp": c.timestamp,
        "open": c.open,
        "high": c.high,
        "low": c.low,
        "close": c.close,
        "volume": c.volume,
    } for c in rows])

    # Create strategy
    strategy = TrendFollowStrategy(params={
        "fast_ma": 5,
        "slow_ma": 20,
        "rsi_period": 14,
    })

    # Simulate: walk through candles one by one (like backtesting)
    signals = []
    window = 100  # need at least this many candles for indicators

    for i in range(window, len(df)):
        chunk = df.iloc[i - window:i + 1].copy()
        signal = strategy.evaluate("ETH", {"5m": chunk})
        if signal:
            signals.append(signal)
            print(f"  [{signal.action.upper():5}] {signal.direction:5} | "
                  f"confidence={signal.confidence:.2f} | {signal.reason}")

    print(f"\nTotal signals: {len(signals)}")
    enters = [s for s in signals if s.action == "enter"]
    exits = [s for s in signals if s.action == "exit"]
    print(f"  Entries: {len(enters)} | Exits: {len(exits)}")


if __name__ == "__main__":
    asyncio.run(main())
