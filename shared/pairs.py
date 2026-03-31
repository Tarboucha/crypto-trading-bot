"""Centralized pair system — follows CCXT convention.

Symbol format:
    Spot:       BASE/QUOTE              → ETH/USDT
    Perpetual:  BASE/QUOTE:SETTLE       → ETH/USDT:USDT
    Futures:    BASE/QUOTE:SETTLE-EXPIRY → ETH/USDT:USDT-230630

The user selects active pairs in config.toml.
Validated against supported bases/quotes at startup.
"""
from dataclasses import dataclass


# Supported base currencies
SUPPORTED_BASES = [
    "ETH", "BTC", "SOL", "DOGE", "XRP", "AVAX", "LINK", "ADA",
]

# Supported quote currencies
SUPPORTED_QUOTES = ["USDT", "USDC", "BTC", "ETH", "EUR"]


@dataclass(frozen=True)
class TradingPair:
    """Unified trading pair representation."""
    base: str               # "ETH"
    quote: str              # "USDT"
    market_type: str = "swap"   # "spot" | "swap" (perp) | "future"
    settle: str | None = None   # "USDT" for linear, "ETH" for inverse, None for spot
    expiry: str | None = None   # "230630" for dated futures, None for perps/spot

    @property
    def symbol(self) -> str:
        """Unified symbol string: ETH/USDT or ETH/USDT:USDT"""
        s = f"{self.base}/{self.quote}"
        if self.settle:
            s += f":{self.settle}"
        if self.expiry:
            s += f"-{self.expiry}"
        return s

    @property
    def short_name(self) -> str:
        """Short name for display: ETH or ETH/BTC"""
        if self.quote == "USDT":
            return self.base
        return f"{self.base}/{self.quote}"

    def exchange_id(self, exchange: str) -> str:
        """Convert to exchange-specific symbol format."""
        formatters = {
            "hyperliquid": self._id_hyperliquid,
            "binance": self._id_binance,
            "aster": self._id_aster,
        }
        formatter = formatters.get(exchange)
        if formatter:
            return formatter()
        return self.symbol

    def _id_hyperliquid(self) -> str:
        # Hyperliquid uses just the base: "ETH"
        return self.base

    def _id_binance(self) -> str:
        # Binance uses concatenated: "ETHUSDT"
        return f"{self.base}{self.quote}"

    def _id_aster(self) -> str:
        # Placeholder — update when Aster API is known
        return f"{self.base}{self.quote}"

    @property
    def csv_symbol(self) -> str:
        """Symbol for CSV file paths (Binance format)."""
        return f"{self.base}{self.quote}"

    def __str__(self) -> str:
        return self.symbol

    def __repr__(self) -> str:
        return f"TradingPair({self.symbol})"


# --- Numeric encoding for ML features ---

PAIR_ENCODING = {base: i for i, base in enumerate(SUPPORTED_BASES)}

# Legacy mapping for CSV paths (Binance format)
PAIR_TO_SYMBOL = {base: f"{base}USDT" for base in SUPPORTED_BASES}


# --- Parsing ---

def parse_pair(symbol: str, trading_mode: str = "futures", settle: str = "USDT") -> TradingPair:
    """Parse a pair string into a TradingPair.

    Accepts multiple formats:
        "ETH"                → ETH/USDT:USDT (perp, using defaults)
        "ETH/USDT"           → ETH/USDT:USDT (perp) or ETH/USDT (spot)
        "ETH/USDT:USDT"      → ETH/USDT:USDT (explicit perp)
        "ETH/BTC"            → ETH/BTC (spot) or ETH/BTC:BTC (perp)

    Args:
        symbol: Pair string in any supported format
        trading_mode: "spot" or "futures" — determines market_type
        settle: Default settlement currency for futures
    """
    expiry = None

    # Handle expiry: ETH/USDT:USDT-230630
    if "-" in symbol and ":" in symbol:
        symbol, expiry = symbol.rsplit("-", 1)

    # Handle settle: ETH/USDT:USDT
    if ":" in symbol:
        symbol, settle = symbol.split(":", 1)

    # Handle base/quote: ETH/USDT
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
    else:
        # Just base name: "ETH" → assume USDT quote
        base = symbol
        quote = "USDT"

    # Determine market type
    if trading_mode == "spot":
        market_type = "spot"
        pair_settle = None
    elif expiry:
        market_type = "future"
        pair_settle = settle
    else:
        market_type = "swap"
        pair_settle = settle

    return TradingPair(
        base=base.upper(),
        quote=quote.upper(),
        market_type=market_type,
        settle=pair_settle,
        expiry=expiry,
    )


def parse_pairs(
    symbols: list[str], trading_mode: str = "futures", settle: str = "USDT"
) -> list[TradingPair]:
    """Parse a list of pair strings into TradingPair objects."""
    return [parse_pair(s, trading_mode, settle) for s in symbols]


def validate_pairs(pairs: list[TradingPair]) -> list[TradingPair]:
    """Validate parsed pairs against supported bases/quotes.

    Raises ValueError if any base or quote is unsupported.
    """
    for pair in pairs:
        if pair.base not in SUPPORTED_BASES:
            raise ValueError(
                f"Unsupported base currency: {pair.base}. "
                f"Supported: {SUPPORTED_BASES}"
            )
        if pair.quote not in SUPPORTED_QUOTES:
            raise ValueError(
                f"Unsupported quote currency: {pair.quote}. "
                f"Supported: {SUPPORTED_QUOTES}"
            )
    return pairs


def get_symbol(pair: str, exchange: str = "binance") -> str:
    """Legacy helper — get exchange symbol from a base currency name."""
    p = parse_pair(pair)
    return p.exchange_id(exchange)
