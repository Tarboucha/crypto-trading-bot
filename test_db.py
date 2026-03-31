"""Quick test: connect to Supabase and create the candles table."""
import asyncio
import sys
import os

from dotenv import load_dotenv

# Load .env from ownbot/
load_dotenv(os.path.join(os.path.dirname(__file__), "ownbot", ".env"))

# Add project root to path so shared module is importable
sys.path.insert(0, os.path.dirname(__file__))

from shared.db.engine import init_db, get_session
from shared.db.models import Candle


async def main():
    # 1. Create tables
    print("Connecting to Supabase...")
    await init_db()
    print("✓ candles table created.")

    # 2. Insert a test row
    async with get_session() as session:
        test_candle = Candle(
            pair="ETH",
            timeframe="5m",
            timestamp=1700000000000,
            open=2100.0,
            high=2110.0,
            low=2095.0,
            close=2105.0,
            volume=1500.0,
        )
        session.add(test_candle)
        await session.commit()
        print("✓ Test candle inserted.")

    # 3. Read it back
    async with get_session() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(Candle).where(Candle.pair == "ETH")
        )
        row = result.scalar_one()
        print(f"✓ Read back: {row}")

    # 4. Clean up test data
    async with get_session() as session:
        from sqlalchemy import delete
        await session.execute(
            delete(Candle).where(Candle.timestamp == 1700000000000)
        )
        await session.commit()
        print("✓ Test data cleaned up.")

    print("\nAll good! Supabase connection works.")


if __name__ == "__main__":
    asyncio.run(main())
