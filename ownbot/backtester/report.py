"""Rich console output for backtest results."""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from ownbot.backtester.result import BacktestResult


def print_report(result: BacktestResult) -> None:
    console = Console()

    # Header
    title = f"BACKTEST — {result.strategy} on {', '.join(result.pairs)} ({result.timeframe})"
    subtitle = f"{result.start_date} → {result.end_date}"

    # Summary table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    # Trades
    table.add_row("Total trades", str(result.total_trades))
    wr_color = "green" if result.win_rate > 0.5 else "red"
    table.add_row(
        "Win rate",
        f"[{wr_color}]{result.win_rate * 100:.1f}%[/] ({result.wins}W / {result.losses}L)",
    )
    pf_color = "green" if result.profit_factor > 1 else "red"
    table.add_row("Profit factor", f"[{pf_color}]{result.profit_factor:.2f}[/]")
    table.add_row("", "")

    # PnL
    pnl_color = "green" if result.total_pnl_pct > 0 else "red"
    table.add_row(
        "Total PnL",
        f"[{pnl_color}]{result.total_pnl_pct * 100:+.2f}% (${result.total_pnl_abs:+.2f})[/]",
    )
    table.add_row("Avg win", f"[green]{result.avg_win_pct * 100:+.2f}%[/]")
    table.add_row("Avg loss", f"[red]{result.avg_loss_pct * 100:+.2f}%[/]")
    table.add_row("Best trade", f"[green]{result.best_trade_pct * 100:+.2f}%[/]")
    table.add_row("Worst trade", f"[red]{result.worst_trade_pct * 100:+.2f}%[/]")
    table.add_row("Total fees", f"[red]${result.total_fees:.2f}[/]")
    table.add_row("", "")

    # Risk
    table.add_row(
        "Max drawdown",
        f"[red]{result.max_drawdown_pct * 100:.2f}% (${result.max_drawdown_abs:.2f})[/]",
    )
    sharpe_color = "green" if result.sharpe_ratio > 1 else "yellow" if result.sharpe_ratio > 0 else "red"
    table.add_row("Sharpe ratio", f"[{sharpe_color}]{result.sharpe_ratio:.2f}[/]")
    table.add_row("Sortino ratio", f"{result.sortino_ratio:.2f}")
    table.add_row("", "")

    # Time
    duration_m = result.avg_trade_duration_s // 60
    duration_h = duration_m // 60
    duration_m = duration_m % 60
    table.add_row("Avg duration", f"{duration_h}h {duration_m}m")

    # Params
    if result.params:
        table.add_row("", "")
        table.add_row("[dim]Parameters[/]", "")
        for k, v in result.params.items():
            table.add_row(f"  [dim]{k}[/]", f"[dim]{v}[/]")

    panel = Panel(table, title=title, subtitle=subtitle, border_style="bold blue")
    console.print()
    console.print(panel)
    console.print()

    # Trade list (last 10)
    if result.trades:
        trade_table = Table(title=f"Last {min(10, len(result.trades))} trades")
        trade_table.add_column("#", style="dim")
        trade_table.add_column("Pair")
        trade_table.add_column("Dir")
        trade_table.add_column("Entry", justify="right")
        trade_table.add_column("Exit", justify="right")
        trade_table.add_column("PnL %", justify="right")
        trade_table.add_column("PnL $", justify="right")
        trade_table.add_column("Reason")

        for i, t in enumerate(result.trades[-10:], 1):
            pnl_color = "green" if t.profit_pct > 0 else "red"
            trade_table.add_row(
                str(i),
                t.pair,
                t.direction,
                f"{t.entry_price:.2f}",
                f"{t.exit_price:.2f}",
                f"[{pnl_color}]{t.profit_pct * 100:+.2f}%[/]",
                f"[{pnl_color}]${t.profit_abs:+.2f}[/]",
                t.reason,
            )

        console.print(trade_table)
        console.print()
