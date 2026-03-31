"""Funding rate data access — DB and CSV operations."""
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import select, func, insert

from shared.db.engine import get_session
from shared.db.models import FundingRate

logger = logging.getLogger(__name__)


class FundingRepo:

    # --- DB operations ---

    @staticmethod
    async def get_rates(
        pair: str, start_ms: int, end_ms: int,
    ) -> list[dict]:
        """Get funding rates for a pair in a time range."""
        async with get_session() as session:
            result = await session.execute(
                select(FundingRate)
                .where(
                    FundingRate.pair == pair,
                    FundingRate.timestamp >= start_ms,
                    FundingRate.timestamp <= end_ms,
                )
                .order_by(FundingRate.timestamp)
            )
            rows = result.scalars().all()

        return [{"pair": r.pair, "timestamp": r.timestamp, "rate": r.rate} for r in rows]

    @staticmethod
    async def get_latest_rate(pair: str) -> float | None:
        """Get the most recent funding rate for a pair."""
        async with get_session() as session:
            result = await session.execute(
                select(FundingRate.rate)
                .where(FundingRate.pair == pair)
                .order_by(FundingRate.timestamp.desc())
                .limit(1)
            )
            return result.scalar()

    @staticmethod
    async def store_rates(rates: list[dict]) -> int:
        """Upsert funding rates. Each dict has: pair, timestamp, rate."""
        if not rates:
            return 0

        async with get_session() as session:
            stmt = insert(FundingRate).values(rates)
            stmt = stmt.on_conflict_do_update(
                index_elements=["pair", "timestamp"],
                set_={"rate": stmt.excluded.rate},
            )
            await session.execute(stmt)
            await session.commit()

        return len(rates)

    # --- CSV operations (for backtesting) ---

    @staticmethod
    def get_rates_csv(
        pair: str, data_dir: str | Path,
        start_ms: int | None = None, end_ms: int | None = None,
    ) -> pd.DataFrame:
        """Load funding rates from CSV files."""
        data_dir = Path(data_dir)
        csv_path = data_dir / "funding_rates" / f"{pair}_funding.csv"

        if not csv_path.exists():
            logger.warning("No funding CSV at %s", csv_path)
            return pd.DataFrame(columns=["timestamp", "rate"])

        df = pd.read_csv(csv_path)

        # Normalize column names
        if "time" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        if "fundingRate" in df.columns and "rate" not in df.columns:
            df = df.rename(columns={"fundingRate": "rate"})

        df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

        if start_ms is not None:
            df = df[df["timestamp"] >= start_ms]
        if end_ms is not None:
            df = df[df["timestamp"] <= end_ms]

        return df[["timestamp", "rate"]].reset_index(drop=True)

    @staticmethod
    def get_rate_at_timestamp(funding_df: pd.DataFrame, timestamp: int) -> float:
        """Get the funding rate active at a given timestamp.

        Returns the most recent rate before or at the timestamp.
        """
        if funding_df.empty:
            return 0.0

        mask = funding_df["timestamp"] <= timestamp
        if not mask.any():
            return 0.0

        return float(funding_df.loc[mask, "rate"].iloc[-1])
