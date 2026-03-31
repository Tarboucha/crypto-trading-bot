"""Trading costs: fees, spread, slippage."""
from dataclasses import dataclass


@dataclass
class TradingCosts:
    fee_pct: float = 0.00035         # 0.035% taker fee per side
    slippage_pct: float = 0.0001     # 0.01% slippage per side
    spread_mode: str = "estimated"   # "estimated" or "fixed"
    spread_fixed_pct: float = 0.0001 # 0.01% half-spread (used if mode=fixed)
    spread_factor: float = 0.1       # fraction of candle range (used if mode=estimated)

    def estimate_half_spread(self, high: float, low: float, close: float) -> float:
        """Estimate half-spread from candle range."""
        if self.spread_mode == "fixed":
            return self.spread_fixed_pct

        range_pct = (high - low) / close if close > 0 else 0
        spread = range_pct * self.spread_factor
        # Clamp between 0.005% and 0.5%
        return max(0.00005, min(spread, 0.005))

    def total_cost_pct(self, high: float, low: float, close: float) -> float:
        """Total cost per side as a fraction (spread + fee + slippage)."""
        half_spread = self.estimate_half_spread(high, low, close)
        return half_spread + self.fee_pct + self.slippage_pct

    def apply_entry_price(
        self, close: float, direction: str, high: float, low: float
    ) -> float:
        """Adjust entry price for costs. Returns the realistic fill price."""
        cost = self.total_cost_pct(high, low, close)
        if direction == "long":
            return close * (1 + cost)   # buy at ask (higher)
        else:
            return close * (1 - cost)   # sell at bid (lower)

    def apply_exit_price(
        self, close: float, direction: str, high: float, low: float
    ) -> float:
        """Adjust exit price for costs. Returns the realistic fill price."""
        cost = self.total_cost_pct(high, low, close)
        if direction == "long":
            return close * (1 - cost)   # sell at bid (lower)
        else:
            return close * (1 + cost)   # buy at ask (higher)

    def fee_for_trade(self, price: float, size: float) -> float:
        """Calculate fee in dollars for one side of a trade."""
        return price * size * self.fee_pct
