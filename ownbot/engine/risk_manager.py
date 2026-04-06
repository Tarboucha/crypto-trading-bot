"""Risk management — checks limits before any trade."""
import logging
import time

from shared.events.component import Component
from ownbot.engine.position_manager import PositionManager
from ownbot.strategy.base import Signal

logger = logging.getLogger(__name__)

# Reset interval presets in seconds
RESET_INTERVALS = {
    "hourly": 3600,
    "4h": 14400,
    "8h": 28800,
    "daily": 86400,
    "weekly": 604800,
    "session": 0,  # never auto-reset, only on bot restart
}


class RiskManager(Component):
    def __init__(
        self,
        max_open_trades: int = 3,
        risk_per_trade_pct: float = 0.3,
        max_exposure_pct: float = 4.0,
        loss_limit_pct: float = 2.0,
        loss_limit_reset: str = "daily",  # "hourly" | "4h" | "8h" | "daily" | "weekly" | "session" | seconds as int
        max_drawdown_pct: float = 8.0,
        max_leverage: float = 1.0,
        liquidation_buffer: float = 0.05,
    ):
        self.max_open_trades = max_open_trades
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_exposure_pct = max_exposure_pct
        self.loss_limit_pct = loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_leverage = max_leverage
        self.liquidation_buffer = liquidation_buffer

        # Loss limit reset
        if isinstance(loss_limit_reset, int):
            self.reset_interval_s = loss_limit_reset
        else:
            self.reset_interval_s = RESET_INTERVALS.get(loss_limit_reset, 86400)

        self.period_pnl: float = 0.0
        self.peak_balance: float = 0.0
        self._last_reset_ts: float = time.time()

    def can_trade(
        self,
        signal: Signal,
        balance: float,
        positions: PositionManager,
    ) -> tuple[bool, str]:
        """Check if a trade is allowed. Returns (allowed, reason)."""

        # Check if loss limit period should reset
        self._check_reset()

        # Max open trades
        if positions.count >= self.max_open_trades:
            reason = f"Max open trades reached ({self.max_open_trades})"
            logger.warning(reason)
            return False, reason

        # Already have position for this pair
        if positions.has_position(signal.pair):
            reason = f"Already have position for {signal.pair}"
            logger.warning(reason)
            return False, reason

        # Period loss limit
        if balance > 0:
            period_loss_pct = (self.period_pnl / balance) * 100
            if period_loss_pct < -self.loss_limit_pct:
                reason = f"Loss limit hit ({period_loss_pct:.2f}% in current period)"
                logger.warning(reason)
                return False, reason

        # Max drawdown
        if self.peak_balance > 0:
            drawdown = ((self.peak_balance - balance) / self.peak_balance) * 100
            if drawdown > self.max_drawdown_pct:
                reason = f"Max drawdown hit ({drawdown:.2f}%)"
                logger.warning(reason)
                return False, reason

        return True, "ok"

    def calculate_position_size(
        self, balance: float, entry_price: float, leverage: float = 1.0,
    ) -> tuple[float, float]:
        """Calculate position size and margin.

        Returns:
            (size, margin) where:
            - margin = collateral put up (based on risk_per_trade_pct)
            - size = leveraged position size (margin * leverage / price)
        """
        margin = balance * (self.risk_per_trade_pct / 100)
        size = (margin / entry_price) * leverage
        return size, margin

    def validate_leverage(self, requested: float) -> float:
        """Clamp leverage to [1.0, max_leverage]."""
        return min(max(requested, 1.0), self.max_leverage)

    def calculate_liquidation_price(
        self, entry_price: float, leverage: float,
        direction: str, maintenance_margin: float = 0.005,
    ) -> float:
        """Calculate liquidation price for a leveraged position."""
        if direction == "long":
            return entry_price * (1 - (1 / leverage) * (1 - maintenance_margin))
        else:
            return entry_price * (1 + (1 / leverage) * (1 - maintenance_margin))

    def apply_liquidation_buffer(
        self, liquidation_price: float, entry_price: float,
    ) -> float:
        """Apply safety buffer to liquidation price (moves it closer to entry)."""
        buffer = abs(entry_price - liquidation_price) * self.liquidation_buffer
        if liquidation_price < entry_price:  # long
            return liquidation_price + buffer
        else:  # short
            return liquidation_price - buffer

    def stoploss_or_liquidation(
        self, stoploss: float, liquidation: float, direction: str,
    ) -> float:
        """Return the more protective of stoploss and liquidation price."""
        if direction == "long":
            return max(stoploss, liquidation)
        else:
            return min(stoploss, liquidation)

    def adjust_stoploss_for_leverage(self, stoploss_pct: float, leverage: float) -> float:
        """Adjust stoploss percentage for leverage.

        A -3% stoploss at 2x leverage triggers at -1.5% price move.
        This keeps the risk per trade constant regardless of leverage.
        """
        return stoploss_pct / leverage

    def check_total_exposure(
        self, positions: list, new_size: float,
        new_price: float, balance: float,
    ) -> bool:
        """Check if adding a new position exceeds max portfolio exposure."""
        current_notional = sum(p.size * p.entry_price for p in positions)
        new_notional = new_size * new_price
        total_exposure_pct = (current_notional + new_notional) / balance * 100
        return total_exposure_pct <= self.max_exposure_pct

    def update_period_pnl(self, pnl: float) -> None:
        self.period_pnl += pnl

    def update_peak_balance(self, balance: float) -> None:
        if balance > self.peak_balance:
            self.peak_balance = balance

    def reset_period(self) -> None:
        """Manually reset the loss limit period."""
        self.period_pnl = 0.0
        self._last_reset_ts = time.time()
        logger.info("Loss limit period reset.")

    def _check_reset(self) -> None:
        """Auto-reset if the configured interval has elapsed."""
        if self.reset_interval_s <= 0:
            return  # session mode — no auto-reset

        elapsed = time.time() - self._last_reset_ts
        if elapsed >= self.reset_interval_s:
            self.reset_period()

    # --- Event handlers (auto-subscribed via Component.register) ---

    async def on_position_closed(self, event) -> None:
        self.update_period_pnl(event.profit_abs)
