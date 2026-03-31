"""Exchange factory — create exchange instances by name."""
from shared.api.exchange.base import BaseExchange
from shared.api.exchange.hyperliquid import HyperliquidExchange, ExchangeConfig


def create_exchange(name: str, config: ExchangeConfig) -> BaseExchange:
    """Create an exchange instance by name.

    Args:
        name: Exchange name ("hyperliquid", "binance", "aster")
        config: Exchange configuration

    Returns:
        BaseExchange instance
    """
    exchanges = {
        "hyperliquid": HyperliquidExchange,
        # "binance": BinanceExchange,   # future
        # "aster": AsterExchange,       # future
    }

    cls = exchanges.get(name)
    if cls is None:
        raise ValueError(f"Unknown exchange: {name}. Available: {list(exchanges.keys())}")

    return cls(config)
