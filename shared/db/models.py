from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, String, Text, PrimaryKeyConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Candle(Base):
    __tablename__ = "candles"

    pair: Mapped[str] = mapped_column(String, nullable=False)
    timeframe: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("pair", "timeframe", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<Candle {self.pair} {self.timeframe} {self.timestamp} c={self.close}>"


class SignalRecord(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(String, nullable=False)
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    timeframe: Mapped[str] = mapped_column(String, nullable=False)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<Signal {self.pair} {self.strategy} {self.action} {self.direction}>"


class FundingRate(Base):
    __tablename__ = "funding_rates"

    pair: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)  # settlement time (ms)
    rate: Mapped[float] = mapped_column(Float, nullable=False)          # e.g. +0.0003

    __table_args__ = (
        PrimaryKeyConstraint("pair", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<FundingRate {self.pair} {self.timestamp} rate={self.rate}>"


class TradeRecord(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(String, nullable=False)
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    profit_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_abs: Mapped[float | None] = mapped_column(Float, nullable=True)
    stoploss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    funding_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    cumulative_funding: Mapped[float] = mapped_column(Float, default=0.0)
    funding_events: Mapped[int] = mapped_column(Integer, default=0)
    entry_time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    exit_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    mode: Mapped[str] = mapped_column(String, nullable=False, default="paper")
    status: Mapped[str] = mapped_column(String, nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<Trade {self.pair} {self.direction} {self.status} pnl={self.profit_pct}>"


class BotSession(Base):
    __tablename__ = "bot_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_s: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    pairs: Mapped[str] = mapped_column(Text, nullable=False)           # JSON
    config: Mapped[str] = mapped_column(Text, nullable=False)          # JSON
    total_signals: Mapped[int] = mapped_column(Integer, default=0)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)
    pnl_abs: Mapped[float] = mapped_column(Float, default=0.0)
    stop_reason: Mapped[str | None] = mapped_column(String, nullable=True)

    def __repr__(self) -> str:
        return f"<Session #{self.id} {self.mode} {self.strategy} trades={self.total_trades}>"


class BotEvent(Base):
    __tablename__ = "bot_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("bot_sessions.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    pair: Mapped[str | None] = mapped_column(String, nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)   # JSON
    level: Mapped[str] = mapped_column(String, default="info")

    def __repr__(self) -> str:
        return f"<Event {self.event_type} {self.pair or ''} [{self.level}]>"
