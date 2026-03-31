"""Download historical data from Hyperliquid API: funding rates + candles. Save as CSV."""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from hyperliquid.info import Info
from hyperliquid.utils import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "hyperliquid"


# --- Funding Rates ---

def fetch_funding_history(info: Info, coin: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch funding rates in weekly chunks."""
    all_rates = []
    chunk_ms = 7 * 24 * 3600 * 1000  # 7 days
    current = start_ms

    while current < end_ms:
        chunk_end = min(current + chunk_ms, end_ms)
        logger.info(
            "  Funding %s: %s → %s",
            coin,
            datetime.fromtimestamp(current / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            datetime.fromtimestamp(chunk_end / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
        )

        rates = info.funding_history(coin, current, chunk_end)
        all_rates.extend(rates)
        current = chunk_end
        time.sleep(0.3)

    return all_rates


def save_funding(coin: str, rates: list[dict]) -> None:
    if not rates:
        logger.warning("No funding data for %s", coin)
        return

    df = pd.DataFrame(rates)
    df = df.rename(columns={
        "coin": "pair",
        "fundingRate": "funding_rate",
        "premium": "premium",
        "time": "timestamp",
    })
    df["funding_rate"] = pd.to_numeric(df["funding_rate"])
    df["premium"] = pd.to_numeric(df["premium"])
    df["timestamp"] = df["timestamp"].astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = DATA_DIR / "funding_rates" / f"{coin}_funding.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("  → Saved %d funding records to %s", len(df), out_path.name)


# --- Candles ---

TIMEFRAME_MAP = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def fetch_candles(info: Info, coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch candles in chunks (max ~500 candles per request)."""
    all_candles = []
    tf_seconds = TIMEFRAME_MAP[interval]
    chunk_candles = 500
    chunk_ms = chunk_candles * tf_seconds * 1000
    current = start_ms

    while current < end_ms:
        chunk_end = min(current + chunk_ms, end_ms)
        logger.info(
            "  Candles %s %s: %s → %s",
            coin, interval,
            datetime.fromtimestamp(current / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            datetime.fromtimestamp(chunk_end / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
        )

        candles = info.candles_snapshot(coin, interval, current, chunk_end)
        all_candles.extend(candles)
        current = chunk_end
        time.sleep(0.3)

    return all_candles


def save_candles(coin: str, interval: str, candles: list[dict]) -> None:
    if not candles:
        logger.warning("No candle data for %s %s", coin, interval)
        return

    df = pd.DataFrame(candles)
    df = df.rename(columns={
        "T": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    })
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df["timestamp"] = df["timestamp"].astype(int)

    # Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    out_path = DATA_DIR / "candles" / coin / f"{coin}_{interval}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("  → Saved %d candles to %s", len(df), out_path.name)


# --- Main ---

def main():
    coins = ["ETH", "BTC"]
    intervals = ["1h"]  # 1h matches funding interval, add more if needed
    months = 6

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (months * 30 * 24 * 3600 * 1000)

    logger.info("Downloading %d months of Hyperliquid data for %s", months, coins)

    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    for coin in coins:
        # Funding rates
        logger.info("Fetching %s funding rates...", coin)
        rates = fetch_funding_history(info, coin, start_ms, now_ms)
        save_funding(coin, rates)

        # Candles
        for interval in intervals:
            logger.info("Fetching %s %s candles...", coin, interval)
            candles = fetch_candles(info, coin, interval, start_ms, now_ms)
            save_candles(coin, interval, candles)

    logger.info("Done! Files saved to %s", DATA_DIR)


if __name__ == "__main__":
    main()
