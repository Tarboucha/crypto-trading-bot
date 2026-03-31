"""Download historical funding rate data from Hyperliquid API and save as CSV."""
import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from hyperliquid.info import Info
from hyperliquid.utils import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "hyperliquid" / "funding_rates"


def get_funding_history(info: Info, coin: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch funding rates in chunks to avoid API limits."""
    all_rates = []
    chunk_ms = 7 * 24 * 3600 * 1000  # 7 days per request
    current = start_ms

    while current < end_ms:
        chunk_end = min(current + chunk_ms, end_ms)
        logger.info(
            "  Fetching %s funding %s → %s",
            coin,
            datetime.fromtimestamp(current / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            datetime.fromtimestamp(chunk_end / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
        )

        rates = info.funding_history(coin, current, chunk_end)
        all_rates.extend(rates)
        current = chunk_end

        # Rate limit: small delay between requests
        time.sleep(0.5)

    return all_rates


def save_funding_csv(coin: str, rates: list[dict]) -> Path:
    """Save funding rates to CSV."""
    if not rates:
        logger.warning("No funding data for %s", coin)
        return None

    df = pd.DataFrame(rates)
    df = df.rename(columns={
        "coin": "pair",
        "fundingRate": "funding_rate",
        "premium": "premium",
        "time": "timestamp",
    })

    # Convert to numeric
    df["funding_rate"] = pd.to_numeric(df["funding_rate"])
    df["premium"] = pd.to_numeric(df["premium"])
    df["timestamp"] = df["timestamp"].astype(int)

    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save
    out_path = DATA_DIR / f"{coin}_funding.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logger.info("Saved %d funding records to %s", len(df), out_path)
    return out_path


def main():
    coins = ["ETH", "BTC"]
    months = 6

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (months * 30 * 24 * 3600 * 1000)

    logger.info(
        "Downloading %d months of funding data for %s",
        months, coins,
    )

    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    for coin in coins:
        logger.info("Fetching %s ...", coin)
        rates = get_funding_history(info, coin, start_ms, now_ms)
        save_funding_csv(coin, rates)

    logger.info("Done! Files saved to %s", DATA_DIR)


if __name__ == "__main__":
    main()
