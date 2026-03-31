"""Download historical candle data from Binance Vision and save as CSV files.

Usage:
    python scripts/download_binance_data.py --pairs BTCUSDT ETHUSDT --timeframes 1m 5m 1h 1d --start 2019-09 --end 2026-02
"""
import argparse
import asyncio
import io
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
DATA_DIR = Path(__file__).parent.parent / "data" / "binance"

CSV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def get_months(start: str, end: str) -> list[tuple[int, int]]:
    """Get list of (year, month) from start to end.

    Args:
        start: "YYYY-MM" e.g. "2019-09"
        end: "YYYY-MM" e.g. "2026-02"
    """
    s_year, s_month = map(int, start.split("-"))
    e_year, e_month = map(int, end.split("-"))

    months = []
    y, m = s_year, s_month
    while (y, m) <= (e_year, e_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


async def download_and_save(
    client: httpx.AsyncClient, symbol: str, interval: str, year: int, month: int
) -> bool:
    """Download a single ZIP from Binance Vision, extract and save as CSV."""
    filename = f"{symbol}-{interval}-{year}-{month:02d}"
    url = f"{BASE_URL}/{symbol}/{interval}/{filename}.zip"
    out_path = DATA_DIR / symbol / interval / f"{filename}.csv"

    if out_path.exists():
        logger.debug("Already exists: %s — skipping.", out_path.name)
        return True

    logger.info("Downloading %s.zip ...", filename)

    try:
        resp = await client.get(url)
        if resp.status_code == 404:
            logger.warning("Not found: %s (data may not be available yet)", filename)
            return False
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("Failed to download %s: %s", filename, e)
        return False

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=CSV_COLUMNS)
            if str(df.iloc[0]["open"]) == "open":
                df = df.iloc[1:].reset_index(drop=True)

    df = df[["close_time", "open", "high", "low", "close", "volume"]].copy()
    df = df.rename(columns={"close_time": "timestamp"})

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df["timestamp"] = df["timestamp"].astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("  → Saved %d candles to %s", len(df), out_path.name)
    return True


async def main():
    parser = argparse.ArgumentParser(description="Download Binance historical data")
    parser.add_argument("--pairs", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["5m"])
    parser.add_argument("--start", type=str, default="2019-09", help="Start month YYYY-MM")
    parser.add_argument("--end", type=str, default="2026-02", help="End month YYYY-MM")
    args = parser.parse_args()

    months = get_months(args.start, args.end)
    total = len(args.pairs) * len(args.timeframes) * len(months)

    logger.info(
        "Downloading %s × %s from %s to %s (%d files)",
        args.pairs, args.timeframes, args.start, args.end, total,
    )

    downloaded = 0
    async with httpx.AsyncClient(timeout=60.0) as client:
        for symbol in args.pairs:
            for interval in args.timeframes:
                for year, month in months:
                    ok = await download_and_save(client, symbol, interval, year, month)
                    if ok:
                        downloaded += 1

    logger.info("Done! %d/%d files downloaded to %s", downloaded, total, DATA_DIR)


if __name__ == "__main__":
    asyncio.run(main())
