"""Hyperliquid exchange adapter with retry and error classification."""
import logging
import time

import pandas as pd
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

from dataclasses import dataclass

from shared.api.exchange.base import (
    BaseExchange, Balance, OrderResult, Position, Ticker,
)
from shared.api.errors import (
    RetryableError, RateLimitError, PermanentError, AuthError,
    InsufficientFundsError, ExchangeDownError,
    OrderRejectedError, OrderUnknownStateError,
)
from shared.api.retry import retry


@dataclass
class ExchangeConfig:
    name: str = "hyperliquid"
    sandbox: bool = True
    key: str = ""
    secret: str = ""


logger = logging.getLogger(__name__)

TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}


def _classify_error(e: Exception) -> Exception:
    """Convert raw exceptions to our error types."""
    msg = str(e).lower()
    if "rate limit" in msg or "429" in msg:
        return RateLimitError(str(e), wait_seconds=10.0)
    if "insufficient" in msg or "not enough" in msg:
        return InsufficientFundsError(str(e))
    if "unauthorized" in msg or "invalid api" in msg or "forbidden" in msg:
        return AuthError(str(e))
    if "connection" in msg or "timeout" in msg or "unreachable" in msg:
        return ExchangeDownError(str(e))
    if "rejected" in msg or "invalid order" in msg:
        return OrderRejectedError(str(e))
    return RetryableError(str(e))


class HyperliquidExchange(BaseExchange):
    """Hyperliquid exchange adapter with retry and error classification."""

    def __init__(self, config: ExchangeConfig) -> None:
        self.config = config
        self.info: Info | None = None
        self.exchange: Exchange | None = None
        self._base_url = (
            constants.TESTNET_API_URL if config.sandbox else constants.MAINNET_API_URL
        )

    @retry(max_attempts=3, backoff=[2, 5, 10])
    async def connect(self) -> None:
        logger.info(
            "Connecting to Hyperliquid %s...",
            "testnet" if self.config.sandbox else "mainnet",
        )
        try:
            self.info = Info(self._base_url, skip_ws=True)
        except Exception as e:
            raise _classify_error(e) from e

        if self.config.secret:
            self.exchange = Exchange(
                wallet=self.config.secret,
                base_url=self._base_url,
            )
            logger.info("Authenticated — trading enabled.")
        else:
            logger.warning("No API secret provided — read-only mode.")

        logger.info("Connected to Hyperliquid.")

    async def close(self) -> None:
        logger.info("Disconnecting from Hyperliquid.")
        self.info = None
        self.exchange = None

    @retry(max_attempts=3, backoff=[1, 2, 5])
    async def get_ticker(self, symbol: str) -> Ticker:
        self._check_connected()
        try:
            all_mids = self.info.all_mids()
            mid = float(all_mids.get(symbol, 0))

            l2 = self.info.l2_snapshot(symbol)
            bid = float(l2["levels"][0][0]["px"]) if l2["levels"][0] else mid
            ask = float(l2["levels"][1][0]["px"]) if l2["levels"][1] else mid

            return Ticker(symbol=symbol, last=mid, bid=bid, ask=ask, volume=0.0)
        except Exception as e:
            raise _classify_error(e) from e

    @retry(max_attempts=3, backoff=[1, 2, 5])
    async def get_candles(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> pd.DataFrame:
        self._check_connected()
        interval = TIMEFRAME_MAP.get(timeframe)
        if not interval:
            raise PermanentError(f"Unsupported timeframe: {timeframe}")

        now_ms = int(time.time() * 1000)
        tf_seconds = _timeframe_to_seconds(timeframe)
        start_ms = now_ms - (limit * tf_seconds * 1000)

        try:
            candles = self.info.candles_snapshot(symbol, interval, start_ms, now_ms)
        except Exception as e:
            raise _classify_error(e) from e

        if not candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles)
        df = df.rename(columns={
            "T": "timestamp", "o": "open", "h": "high",
            "l": "low", "c": "close", "v": "volume",
        })
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        return df.tail(limit).reset_index(drop=True)

    @retry(max_attempts=3, backoff=[1, 2, 5])
    async def get_balance(self) -> Balance:
        self._check_connected()
        if not self.config.key:
            raise AuthError("API key required to fetch balance.")

        try:
            state = self.info.user_state(self.config.key)
        except Exception as e:
            raise _classify_error(e) from e

        margin = state.get("marginSummary", {})
        return Balance(
            total=float(margin.get("accountValue", 0)),
            free=float(margin.get("totalRawUsd", 0)),
            used=float(margin.get("totalMarginUsed", 0)),
        )

    @retry(max_attempts=3, backoff=[1, 2, 5])
    async def get_positions(self) -> list[Position]:
        self._check_connected()
        if not self.config.key:
            raise AuthError("API key required to fetch positions.")

        try:
            state = self.info.user_state(self.config.key)
        except Exception as e:
            raise _classify_error(e) from e

        positions = []
        for pos in state.get("assetPositions", []):
            p = pos.get("position", {})
            size = float(p.get("szi", 0))
            if size == 0:
                continue
            positions.append(Position(
                symbol=p.get("coin", ""),
                side="long" if size > 0 else "short",
                size=abs(size),
                entry_price=float(p.get("entryPx", 0)),
                unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                leverage=float(p.get("leverage", {}).get("value", 1)),
            ))
        return positions

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
    ) -> OrderResult:
        """Place an order. NO retry — duplicate orders are dangerous.
        Raises OrderRejectedError or OrderUnknownStateError on failure."""
        self._check_connected()
        if not self.exchange:
            raise AuthError("API secret required to place orders.")

        is_buy = side == "buy"

        if order_type == "market":
            ticker = await self.get_ticker(symbol)
            price = ticker.ask * 1.005 if is_buy else ticker.bid * 0.995

        try:
            result = self.exchange.order(
                coin=symbol,
                is_buy=is_buy,
                sz=amount,
                limit_px=price,
                order_type={"limit": {"tif": "Gtc"}} if order_type == "limit" else {"limit": {"tif": "Ioc"}},
            )
        except ConnectionError as e:
            # Network failed — we don't know if order reached exchange
            raise OrderUnknownStateError(pair=symbol) from e
        except Exception as e:
            classified = _classify_error(e)
            if isinstance(classified, InsufficientFundsError):
                raise OrderRejectedError(str(e), pair=symbol) from e
            raise classified from e

        status = result.get("response", {}).get("data", {}).get("statuses", [{}])[0]

        # Check for error in response
        if "error" in status:
            raise OrderRejectedError(status["error"], pair=symbol)

        order_id = status.get("resting", {}).get("oid", "") or status.get("filled", {}).get("oid", "")
        is_filled = "filled" in status
        fill_status = "filled" if is_filled else "open"

        return OrderResult(
            order_id=str(order_id),
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            status=fill_status,
            filled_size=amount if is_filled else 0.0,
            fill_price=price if is_filled else 0.0,
        )

    @retry(max_attempts=3, backoff=[1, 2, 5])
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        self._check_connected()
        if not self.exchange:
            raise AuthError("API secret required to cancel orders.")

        try:
            result = self.exchange.cancel(coin=symbol, oid=int(order_id))
        except Exception as e:
            raise _classify_error(e) from e

        return result.get("status") == "ok"

    @retry(max_attempts=5, backoff=[1, 1, 2, 3, 5])
    async def get_order_status(self, order_id: str, symbol: str) -> OrderResult:
        self._check_connected()
        if not self.config.key:
            raise AuthError("API key required to check order status.")

        try:
            open_orders = self.info.open_orders(self.config.key)
        except Exception as e:
            raise _classify_error(e) from e

        for order in open_orders:
            if str(order.get("oid", "")) == order_id:
                return OrderResult(
                    order_id=order_id, symbol=symbol,
                    side="buy" if order.get("side") == "B" else "sell",
                    order_type="limit",
                    amount=float(order.get("sz", 0)),
                    price=float(order.get("limitPx", 0)),
                    status="open",
                    filled_size=0.0, fill_price=0.0,
                )

        # Not in open orders → assumed filled
        return OrderResult(
            order_id=order_id, symbol=symbol,
            side="", order_type="", amount=0.0, price=None,
            status="filled", filled_size=0.0, fill_price=0.0,
        )

    @retry(max_attempts=3, backoff=[1, 2, 5])
    async def get_funding_rate(self, symbol: str) -> float:
        self._check_connected()
        try:
            meta = self.info.meta()
        except Exception as e:
            raise _classify_error(e) from e

        for asset in meta.get("universe", []):
            if asset.get("name") == symbol:
                return float(asset.get("funding", 0.0))
        return 0.0

    @retry(max_attempts=3, backoff=[1, 2, 5])
    async def get_funding_history(self, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
        self._check_connected()
        try:
            raw = self.info.funding_history(symbol, start_ms, end_ms)
        except Exception as e:
            raise _classify_error(e) from e

        return [
            {"timestamp": int(entry["time"]), "rate": float(entry["fundingRate"])}
            for entry in raw
        ]

    def _check_connected(self) -> None:
        if self.info is None:
            raise ExchangeDownError("Not connected. Call connect() first.")


def _timeframe_to_seconds(timeframe: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86400}
    return int(timeframe[:-1]) * units[timeframe[-1]]
