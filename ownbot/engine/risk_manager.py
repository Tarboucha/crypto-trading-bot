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
    ):
        self.max_open_trades = max_open_trades
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_exposure_pct = max_exposure_pct
        self.loss_limit_pct = loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct

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

    def calculate_position_size(self, balance: float, entry_price: float) -> float:
        """Calculate position size based on risk per trade."""
        risk_amount = balance * (self.risk_per_trade_pct / 100)
        size = risk_amount / entry_price
        return size

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
