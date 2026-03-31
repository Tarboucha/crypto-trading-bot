import os
import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from shared.db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


def _get_database_url() -> str:
    """Get database URL from environment, converting to async driver if needed."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not set in environment / .env file")

    # Strip quotes if present (from .env)
    url = url.strip('"').strip("'")

    # Remove pgbouncer param (not supported by asyncpg)
    url = url.replace("?pgbouncer=true", "").replace("&pgbouncer=true", "")

    # Convert postgresql:// to postgresql+asyncpg:// for async support
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    return url


def get_engine():
    global _engine
    if _engine is None:
        url = _get_database_url()
        _engine = create_async_engine(
            url,
            echo=False,
            connect_args={"statement_cache_size": 0},
        )
        logger.debug("Database engine created.")
    return _engine


def get_session() -> AsyncSession:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory()


async def init_db() -> None:
    """Create all tables if they don't exist."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized.")
