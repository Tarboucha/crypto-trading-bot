"""Signal data access — single place for all signal DB operations."""
import logging

from shared.db.engine import get_session
from shared.db.models import SignalRecord

logger = logging.getLogger(__name__)


class SignalRepo:

    @staticmethod
    async def store_signal(
        pair: str, strategy: str, timeframe: str,
        direction: str, action: str, confidence: float,
        reason: str, timestamp: int,
    ) -> int:
        """Store a signal and return its ID."""
        async with get_session() as session:
            record = SignalRecord(
                pair=pair,
                strategy=strategy,
                timeframe=timeframe,
                direction=direction,
                action=action,
                confidence=confidence,
                reason=reason,
                timestamp=timestamp,
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record.id
