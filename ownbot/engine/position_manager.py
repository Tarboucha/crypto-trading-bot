"""Tracks open and closed positions, including funding fees and trailing stoploss."""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Position:
    pair: str
    direction: str          # "long" | "short"
    entry_price: float
    size: float
    entry_time: int
    stoploss: float | None = None
    take_profit: float | None = None
    strategy: str = ""
    cumulative_funding: float = 0.0
    funding_events: int = 0
    # Trailing stoploss
    trailing_stop: bool = False
    trailing_distance_pct: float = 0.0
    trailing_activate_pct: float = 0.0
    peak_price: float = 0.0
    trough_price: float = float("inf")


@dataclass
class ClosedTrade:
    pair: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    profit_pct: float
    profit_abs: float
    entry_time: int
    exit_time: int
    reason: str
    funding_pnl: float = 0.0
    strategy: str = ""


class PositionManager:
    def __init__(self):
        self.open_positions: dict[str, Position] = {}

    def has_position(self, pair: str) -> bool:
        return pair in self.open_positions

    def get_position(self, pair: str) -> Position | None:
        return self.open_positions.get(pair)

    def open(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        size: float,
        entry_time: int,
        strategy: str = "",
        stoploss: float | None = None,
        take_profit: float | None = None,
        trailing_stop: bool = False,
        trailing_distance_pct: float = 0.0,
        trailing_activate_pct: float = 0.0,
    ) -> Position:
        if self.has_position(pair):
            raise RuntimeError(f"Already have an open position for {pair}")

        pos = Position(
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            size=size,
            entry_time=entry_time,
            strategy=strategy,
            stoploss=stoploss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            trailing_distance_pct=trailing_distance_pct,
            trailing_activate_pct=trailing_activate_pct,
            peak_price=entry_price,
            trough_price=entry_price,
        )
        self.open_positions[pair] = pos

        trail_str = f", trail={trailing_distance_pct}%" if trailing_stop else ""
        logger.info(
            "Opened %s %s @ %.2f (size=%.4f, sl=%s, tp=%s%s)",
            direction, pair, entry_price, size,
            f"{stoploss:.2f}" if stoploss else "none",
            f"{take_profit:.2f}" if take_profit else "none",
            trail_str,
        )
        return pos

    def close(self, pair: str, exit_price: float, exit_time: int, reason: str) -> ClosedTrade:
        pos = self.open_positions.pop(pair, None)
        if pos is None:
            raise RuntimeError(f"No open position for {pair}")

        if pos.direction == "long":
            profit_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            profit_pct = (pos.entry_price - exit_price) / pos.entry_price

        price_pnl = profit_pct * pos.size * pos.entry_price
        profit_abs = price_pnl + pos.cumulative_funding

        trade = ClosedTrade(
            pair=pair,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            profit_pct=profit_pct,
            profit_abs=profit_abs,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            reason=reason,
            strategy=pos.strategy,
            funding_pnl=pos.cumulative_funding,
        )

        funding_str = f" | funding: ${pos.cumulative_funding:+.4f}" if pos.funding_events > 0 else ""
        logger.info(
            "Closed %s %s @ %.2f → %.2f (pnl=%.2f%%, $%.2f%s) reason: %s",
            pos.direction, pair, pos.entry_price, exit_price,
            profit_pct * 100, profit_abs, funding_str, reason,
        )
        return trade

    def apply_funding(self, pair: str, rate: float) -> float | None:
        pos = self.get_position(pair)
        if pos is None:
            return None

        notional = pos.size * pos.entry_price

        if pos.direction == "long":
            funding = -notional * rate
        else:
            funding = notional * rate

        pos.cumulative_funding += funding
        pos.funding_events += 1

        if abs(funding) > 0.001:
            logger.debug(
                "[%s] Funding: rate=%+.5f, amount=$%+.4f (cumulative: $%+.4f, events: %d)",
                pair, rate, funding, pos.cumulative_funding, pos.funding_events,
            )

        return funding

    def _update_trailing_sl(self, pos: Position, current_price: float) -> None:
        """Update trailing stoploss if price made a new peak/trough."""
        if not pos.trailing_stop:
            return

        distance = pos.trailing_distance_pct / 100

        if pos.direction == "long":
            if current_price > pos.peak_price:
                pos.peak_price = current_price

            # Check activation threshold
            profit_pct = (pos.peak_price - pos.entry_price) / pos.entry_price
            if profit_pct < pos.trailing_activate_pct / 100:
                return

            new_sl = pos.peak_price * (1 - distance)
            if pos.stoploss is None or new_sl > pos.stoploss:
                logger.debug(
                    "[%s] Trailing SL: %.2f → %.2f (peak=%.2f)",
                    pos.pair, pos.stoploss or 0, new_sl, pos.peak_price,
                )
                pos.stoploss = new_sl

        else:  # short
            if current_price < pos.trough_price:
                pos.trough_price = current_price

            profit_pct = (pos.entry_price - pos.trough_price) / pos.entry_price
            if profit_pct < pos.trailing_activate_pct / 100:
                return

            new_sl = pos.trough_price * (1 + distance)
            if pos.stoploss is None or new_sl < pos.stoploss:
                logger.debug(
                    "[%s] Trailing SL: %.2f → %.2f (trough=%.2f)",
                    pos.pair, pos.stoploss or 0, new_sl, pos.trough_price,
                )
                pos.stoploss = new_sl

    def check_stoploss_takeprofit(self, pair: str, current_price: float, current_time: int) -> ClosedTrade | None:
        pos = self.get_position(pair)
        if pos is None:
            return None

        # Update trailing SL before checking
        self._update_trailing_sl(pos, current_price)

        if pos.direction == "long":
            if pos.stoploss and current_price <= pos.stoploss:
                return self.close(pair, current_price, current_time, "stoploss hit")
            if pos.take_profit and current_price >= pos.take_profit:
                return self.close(pair, current_price, current_time, "takeprofit hit")
        else:
            if pos.stoploss and current_price >= pos.stoploss:
                return self.close(pair, current_price, current_time, "stoploss hit")
            if pos.take_profit and current_price <= pos.take_profit:
                return self.close(pair, current_price, current_time, "takeprofit hit")

        return None

    @property
    def count(self) -> int:
        return len(self.open_positions)
