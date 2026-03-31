"""Session data access — single place for all session/event DB operations."""
import json
import logging
from datetime import datetime

from sqlalchemy import select, update

from shared.db.engine import get_session
from shared.db.models import BotSession, BotEvent

logger = logging.getLogger(__name__)


class SessionRepo:

    @staticmethod
    async def create_session(
        mode: str, strategy: str, pairs: list[str], config: dict,
    ) -> int:
        """Create a new bot session and return its ID."""
        async with get_session() as session:
            record = BotSession(
                mode=mode,
                strategy=strategy,
                pairs=json.dumps(pairs),
                config=json.dumps(config),
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record.id

    @staticmethod
    async def close_session(
        session_id: int, stop_reason: str,
        total_signals: int, total_trades: int,
        wins: int, losses: int,
        pnl_pct: float, pnl_abs: float,
    ) -> None:
        """Finalize a bot session with stats."""
        async with get_session() as session:
            # Get started_at for duration calculation
            result = await session.execute(
                select(BotSession.started_at).where(BotSession.id == session_id)
            )
            started_at = result.scalar()
            duration_s = int((datetime.utcnow() - started_at).total_seconds()) if started_at else 0

            await session.execute(
                update(BotSession)
                .where(BotSession.id == session_id)
                .values(
                    stopped_at=datetime.utcnow(),
                    duration_s=duration_s,
                    total_signals=total_signals,
                    total_trades=total_trades,
                    wins=wins,
                    losses=losses,
                    pnl_pct=pnl_pct,
                    pnl_abs=pnl_abs,
                    stop_reason=stop_reason,
                )
            )
            await session.commit()

    @staticmethod
    async def log_event(
        session_id: int, event_type: str,
        pair: str | None = None,
        details: dict | None = None,
        level: str = "info",
    ) -> None:
        """Log a bot event."""
        async with get_session() as session:
            event = BotEvent(
                session_id=session_id,
                event_type=event_type,
                pair=pair,
                details=json.dumps(details) if details else None,
                level=level,
            )
            session.add(event)
            await session.commit()
