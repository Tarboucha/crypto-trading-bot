"""Candle data access — single place for all candle DB operations."""
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import select, func, insert

from shared.db.engine import get_session
from shared.db.models import Candle
from shared.pairs import PAIR_TO_SYMBOL

logger = logging.getLogger(__name__)


class CandleRepo:
    """Repository for candle data — DB and CSV access."""

    # --- DB operations ---

    @staticmethod
    async def get_candles(
        pair: str, timeframe: str, limit: int = 100,
        start_ms: int | None = None, end_ms: int | None = None,
    ) -> pd.DataFrame:
        """Fetch candles from DB."""
        async with get_session() as session:
            query = select(Candle).where(
                Candle.pair == pair, Candle.timeframe == timeframe,
            )
            if start_ms is not None:
                query = query.where(Candle.timestamp >= start_ms)
            if end_ms is not None:
                query = query.where(Candle.timestamp <= end_ms)
            query = query.order_by(Candle.timestamp.desc()).limit(limit)

            result = await session.execute(query)
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rows.reverse()
        return pd.DataFrame([{
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        } for c in rows])

    @staticmethod
    async def get_last_timestamp(pair: str, timeframe: str) -> int | None:
        """Get the most recent stored candle timestamp."""
        async with get_session() as session:
            result = await session.execute(
                select(func.max(Candle.timestamp)).where(
                    Candle.pair == pair,
                    Candle.timeframe == timeframe,
                )
            )
            return result.scalar()

    @staticmethod
    async def store_candles(candles: list[dict]) -> int:
        """Upsert candles into DB. Returns number of rows affected."""
        if not candles:
            return 0

        async with get_session() as session:
            stmt = insert(Candle).values(candles)
            stmt = stmt.on_conflict_do_update(
                index_elements=["pair", "timeframe", "timestamp"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                },
            )
            await session.execute(stmt)
            await session.commit()

        return len(candles)

    # --- CSV operations ---

    @staticmethod
    def get_candles_csv(
        pair: str, timeframe: str, data_dir: str | Path,
        start_ms: int | None = None, end_ms: int | None = None,
    ) -> pd.DataFrame:
        """Load candles from local CSV files."""
        data_dir = Path(data_dir)
        symbol = PAIR_TO_SYMBOL.get(pair, pair)
        csv_dir = data_dir / symbol / timeframe

        if not csv_dir.exists():
            logger.warning("No CSV data at %s", csv_dir)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s", csv_dir)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        if start_ms is not None:
            df = df[df["timestamp"] >= start_ms]
        if end_ms is not None:
            df = df[df["timestamp"] <= end_ms]

        return df.reset_index(drop=True)
