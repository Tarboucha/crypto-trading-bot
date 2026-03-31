"""Trade data access — single place for all trade DB operations."""
import logging

from sqlalchemy import select, update

from shared.db.engine import get_session
from shared.db.models import TradeRecord

logger = logging.getLogger(__name__)


class TradeRepo:

    @staticmethod
    async def open_trade(
        pair: str, strategy: str, direction: str,
        entry_price: float, size: float, entry_time: int,
        stoploss: float | None = None, take_profit: float | None = None,
        mode: str = "paper",
    ) -> int:
        """Insert a new trade with status='open'. Returns trade ID."""
        async with get_session() as session:
            record = TradeRecord(
                pair=pair, strategy=strategy, direction=direction,
                entry_price=entry_price, size=size, entry_time=entry_time,
                stoploss=stoploss, take_profit=take_profit,
                mode=mode, status="open",
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record.id

    @staticmethod
    async def close_trade(
        pair: str, exit_price: float, exit_time: int,
        profit_pct: float, profit_abs: float,
        funding_pnl: float = 0.0, reason: str = "",
    ) -> None:
        """Update an open trade to closed with exit details."""
        async with get_session() as session:
            await session.execute(
                update(TradeRecord)
                .where(TradeRecord.pair == pair, TradeRecord.status == "open")
                .values(
                    exit_price=exit_price,
                    exit_time=exit_time,
                    profit_pct=profit_pct,
                    profit_abs=profit_abs,
                    funding_pnl=funding_pnl,
                    reason=reason,
                    status="closed",
                )
            )
            await session.commit()

    @staticmethod
    async def update_funding(
        pair: str, cumulative_funding: float, funding_events: int,
    ) -> None:
        """Update funding fields on an open trade."""
        async with get_session() as session:
            await session.execute(
                update(TradeRecord)
                .where(TradeRecord.pair == pair, TradeRecord.status == "open")
                .values(
                    cumulative_funding=cumulative_funding,
                    funding_events=funding_events,
                )
            )
            await session.commit()

    @staticmethod
    async def get_open_trades() -> list[dict]:
        """Get all open trades (for startup recovery)."""
        async with get_session() as session:
            result = await session.execute(
                select(TradeRecord).where(TradeRecord.status == "open")
            )
            rows = result.scalars().all()

        return [{
            "pair": r.pair,
            "direction": r.direction,
            "entry_price": r.entry_price,
            "size": r.size,
            "entry_time": r.entry_time,
            "strategy": r.strategy,
            "stoploss": r.stoploss,
            "take_profit": r.take_profit,
            "cumulative_funding": r.cumulative_funding,
            "funding_events": r.funding_events,
        } for r in rows]

    @staticmethod
    async def store_trade(
        pair: str, strategy: str, direction: str,
        entry_price: float, exit_price: float,
        size: float, profit_pct: float, profit_abs: float,
        entry_time: int, exit_time: int,
        reason: str, mode: str = "paper",
    ) -> int:
        """Store a closed trade directly (legacy, used by backtester)."""
        async with get_session() as session:
            record = TradeRecord(
                pair=pair, strategy=strategy, direction=direction,
                entry_price=entry_price, exit_price=exit_price,
                size=size, profit_pct=profit_pct, profit_abs=profit_abs,
                entry_time=entry_time, exit_time=exit_time,
                reason=reason, mode=mode, status="closed",
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record.id
