"""Backtest result dataclass."""
from dataclasses import dataclass, field

from ownbot.engine.position_manager import ClosedTrade


@dataclass
class BacktestResult:
    # Summary
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0

    # PnL
    total_pnl_pct: float = 0.0
    total_pnl_abs: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    profit_factor: float = 0.0

    # Costs
    total_fees: float = 0.0
    gross_pnl_abs: float = 0.0   # P&L before fees

    # Risk
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Time
    avg_trade_duration_s: int = 0
    start_date: str = ""
    end_date: str = ""

    # Config
    strategy: str = ""
    pairs: list[str] = field(default_factory=list)
    timeframe: str = ""
    params: dict = field(default_factory=dict)

    # Raw data
    trades: list[ClosedTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @staticmethod
    def from_trades(
        trades: list[ClosedTrade],
        initial_balance: float,
        start_date: str,
        end_date: str,
        strategy: str = "",
        pairs: list[str] | None = None,
        timeframe: str = "",
        params: dict | None = None,
    ) -> "BacktestResult":
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            pairs=pairs or [],
            timeframe=timeframe,
            params=params or {},
            trades=trades,
        )

        if not trades:
            result.equity_curve = [initial_balance]
            return result

        result.total_trades = len(trades)
        result.wins = sum(1 for t in trades if t.profit_pct > 0)
        result.losses = sum(1 for t in trades if t.profit_pct <= 0)
        result.win_rate = result.wins / result.total_trades if result.total_trades else 0

        # PnL
        pnls = [t.profit_pct for t in trades]
        win_pnls = [p for p in pnls if p > 0]
        loss_pnls = [p for p in pnls if p <= 0]

        result.total_pnl_pct = sum(pnls)
        result.total_pnl_abs = sum(t.profit_abs for t in trades)
        result.avg_win_pct = sum(win_pnls) / len(win_pnls) if win_pnls else 0
        result.avg_loss_pct = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
        result.best_trade_pct = max(pnls) if pnls else 0
        result.worst_trade_pct = min(pnls) if pnls else 0

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Equity curve
        balance = initial_balance
        curve = [balance]
        for t in trades:
            balance += t.profit_abs
            curve.append(balance)
        result.equity_curve = curve

        # Max drawdown
        peak = curve[0]
        max_dd_abs = 0.0
        max_dd_pct = 0.0
        for val in curve:
            if val > peak:
                peak = val
            dd_abs = peak - val
            dd_pct = dd_abs / peak if peak > 0 else 0
            if dd_abs > max_dd_abs:
                max_dd_abs = dd_abs
                max_dd_pct = dd_pct
        result.max_drawdown_abs = max_dd_abs
        result.max_drawdown_pct = max_dd_pct

        # Sharpe & Sortino (annualized, assuming daily returns)
        import statistics
        if len(pnls) > 1:
            mean_r = statistics.mean(pnls)
            std_r = statistics.stdev(pnls)
            result.sharpe_ratio = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0

            downside = [p for p in pnls if p < 0]
            if downside:
                down_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
                result.sortino_ratio = (mean_r / down_std) * (252 ** 0.5) if down_std > 0 else 0

        # Avg trade duration
        durations = [t.exit_time - t.entry_time for t in trades if t.exit_time and t.entry_time]
        if durations:
            result.avg_trade_duration_s = int(sum(durations) / len(durations) / 1000)

        return result
