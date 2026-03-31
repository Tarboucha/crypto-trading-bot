"""Core collector: fetches candles from exchange and stores them in DB."""
import logging
import time

from hyperliquid.info import Info
from hyperliquid.utils import constants

from shared.api.db.candle_repo import CandleRepo

logger = logging.getLogger(__name__)

TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class CandleCollector:
    def __init__(self, pairs: list[str], timeframes: list[str], initial_limit: int = 500):
        self.pairs = pairs
        self.timeframes = timeframes
        self.initial_limit = initial_limit
        self.info: Info | None = None
        self.candle_repo = CandleRepo()

    async def connect(self) -> None:
        logger.info("Connecting to Hyperliquid mainnet...")
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        logger.info("Connected.")

    async def close(self) -> None:
        self.info = None
        logger.info("Disconnected.")

    def fetch_candles_from_api(
        self, pair: str, timeframe: str, start_ms: int, end_ms: int
    ) -> list[dict]:
        """Fetch candles from Hyperliquid API."""
        raw = self.info.candles_snapshot(pair, timeframe, start_ms, end_ms)
        if not raw:
            return []

        candles = []
        for c in raw:
            candles.append({
                "pair": pair,
                "timeframe": timeframe,
                "timestamp": c["T"],
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            })
        return candles

    async def backfill(self, pair: str, timeframe: str) -> int:
        """Backfill missing candles for a pair/timeframe."""
        now_ms = int(time.time() * 1000)
        last_ts = await self.candle_repo.get_last_timestamp(pair, timeframe)

        if last_ts is None:
            tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000
            start_ms = now_ms - (self.initial_limit * tf_ms)
            logger.info(
                "%s/%s — first run, fetching last %d candles...",
                pair, timeframe, self.initial_limit,
            )
        else:
            start_ms = last_ts + 1
            gap_seconds = (now_ms - last_ts) / 1000
            gap_candles = int(gap_seconds / TIMEFRAME_SECONDS[timeframe])

            if gap_candles <= 1:
                logger.debug("%s/%s — up to date.", pair, timeframe)
                return 0

            logger.info(
                "%s/%s — filling gap of ~%d candles...",
                pair, timeframe, gap_candles,
            )

        candles = self.fetch_candles_from_api(pair, timeframe, start_ms, now_ms)
        stored = await self.candle_repo.store_candles(candles)
        logger.info("%s/%s — stored %d candles.", pair, timeframe, stored)
        return stored

    async def collect_latest(self, pair: str, timeframe: str) -> int:
        """Fetch only the latest candle(s) since last stored."""
        now_ms = int(time.time() * 1000)
        last_ts = await self.candle_repo.get_last_timestamp(pair, timeframe)

        if last_ts is None:
            return await self.backfill(pair, timeframe)

        candles = self.fetch_candles_from_api(pair, timeframe, last_ts, now_ms)
        if not candles:
            return 0

        stored = await self.candle_repo.store_candles(candles)
        if stored > 0:
            logger.debug("%s/%s — +%d candle(s).", pair, timeframe, stored)
        return stored

    async def run_once(self) -> int:
        """Single collection cycle: fetch latest candles for all pairs/timeframes."""
        total = 0
        for pair in self.pairs:
            for tf in self.timeframes:
                count = await self.collect_latest(pair, tf)
                total += count
        return total
