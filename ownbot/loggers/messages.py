"""Standardized log message helpers — consistent format across the bot."""
import logging


# --- Startup / Shutdown ---

def log_startup(logger: logging.Logger, config) -> None:
    logger.info("OwnBot starting...")
    logger.info("Strategy: %s", config.strategy.name)
    logger.info("Mode: %s | Timeframe: %s", config.mode, config.timeframe)
    logger.info("Pairs: %s", ", ".join(str(p) for p in config.pairs))


def log_connected(logger: logging.Logger, exchange_name: str, network: str) -> None:
    logger.info("Connected to %s (%s)", exchange_name, network)


def log_db_connected(logger: logging.Logger, host: str = "") -> None:
    logger.info("Database connected%s", f" ({host})" if host else "")


def log_model_loaded(logger: logging.Logger, model_name: str, device: str, **kwargs) -> None:
    extra = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info("Model loaded: %s (device=%s%s)", model_name, device, f", {extra}" if extra else "")


def log_session_started(logger: logging.Logger, session_id: int, interval_s: int) -> None:
    logger.info("Session #%d started — polling every %ds", session_id, interval_s)


def log_session_stopped(logger: logging.Logger, session_id: int, reason: str,
                        duration: str, trades: int, wins: int, losses: int,
                        pnl_pct: float, pnl_abs: float, fees: float = 0.0) -> None:
    logger.info("Session #%d stopped (%s)", session_id, reason)
    logger.info("═" * 45)
    logger.info("Duration: %s | Trades: %d (%dW/%dL)", duration, trades, wins, losses)
    logger.info("PnL: %+.2f%% ($%+.2f) | Fees: $%.2f", pnl_pct * 100, pnl_abs, fees)
    logger.info("═" * 45)


# --- Config Warnings ---

def log_missing_env(logger: logging.Logger, var_name: str, consequence: str) -> None:
    logger.warning("%s not set — %s", var_name, consequence)


def log_missing_file(logger: logging.Logger, path: str, consequence: str) -> None:
    logger.warning("File not found: %s — %s", path, consequence)


# --- Trade Lifecycle ---

def log_signal(logger: logging.Logger, pair: str, action: str, direction: str,
               confidence: float, reason: str) -> None:
    logger.info("[%s] Signal: %s %s — %s (%.0f%%)",
                pair, action.upper(), direction.upper(), reason, confidence * 100)


def log_order_submitted(logger: logging.Logger, pair: str, side: str,
                        size: float, price: float, order_type: str = "limit") -> None:
    logger.info("[%s] %s %s order submitted: %.6f @ %.2f",
                pair, order_type.capitalize(), side, size, price)


def log_order_filled(logger: logging.Logger, pair: str, side: str,
                     size: float, price: float, fee: float = 0.0) -> None:
    logger.info("[%s] %s order filled: %.6f @ %.2f (fee: $%.3f)",
                pair, side.capitalize(), size, price, fee)


def log_order_cancelled(logger: logging.Logger, pair: str, reason: str) -> None:
    logger.warning("[%s] Order cancelled — %s", pair, reason)


def log_position_opened(logger: logging.Logger, pair: str, trade_id: int,
                        direction: str, size: float, price: float,
                        sl: float, tp: float) -> None:
    logger.info("[%s] Position opened (#%d): %s %.6f @ %.2f | SL=%.2f TP=%.2f",
                pair, trade_id, direction.upper(), size, price, sl, tp)


def log_position_update(logger: logging.Logger, pair: str, direction: str,
                        size: float, unrealized_pct: float, unrealized_abs: float,
                        candles_held: int) -> None:
    logger.debug("[%s] Position: %s %.6f | unrealized: %+.2f%% ($%+.2f) | held: %d candles",
                 pair, direction.upper(), size, unrealized_pct * 100, unrealized_abs, candles_held)


def log_trade_closed(logger: logging.Logger, pair: str, trade_id: int,
                     direction: str, pnl_pct: float, pnl_abs: float,
                     entry_price: float, exit_price: float,
                     duration: str, reason: str) -> None:
    logger.info("[%s] Trade closed (#%d): %s | %+.2f%% ($%+.2f) | %.2f→%.2f | %s | %s",
                pair, trade_id, direction.upper(), pnl_pct * 100, pnl_abs,
                entry_price, exit_price, duration, reason)


def log_stoploss_hit(logger: logging.Logger, pair: str, trade_id: int,
                     direction: str, pnl_pct: float, pnl_abs: float,
                     entry_price: float, exit_price: float, duration: str) -> None:
    logger.warning("[%s] Stoploss hit (#%d): %s | %.2f%% ($%.2f) | %.2f→%.2f | %s",
                   pair, trade_id, direction.upper(), pnl_pct * 100, pnl_abs,
                   entry_price, exit_price, duration)


def log_takeprofit_hit(logger: logging.Logger, pair: str, trade_id: int,
                       direction: str, pnl_pct: float, pnl_abs: float,
                       entry_price: float, exit_price: float, duration: str) -> None:
    logger.info("[%s] Takeprofit hit (#%d): %s | +%.2f%% ($+%.2f) | %.2f→%.2f | %s",
                pair, trade_id, direction.upper(), pnl_pct * 100, pnl_abs,
                entry_price, exit_price, duration)


# --- Risk Events ---

def log_risk_rejected(logger: logging.Logger, pair: str, reason: str) -> None:
    logger.warning("[%s] Entry rejected: %s", pair, reason)


def log_loss_limit_reset(logger: logging.Logger, previous_pnl: float) -> None:
    logger.info("Loss limit period reset (was %+.2f%%)", previous_pnl * 100)


# --- Exchange Events ---

def log_exchange_error(logger: logging.Logger, pair: str, error: str) -> None:
    logger.error("[%s] Exchange error: %s", pair, error)


def log_reconnecting(logger: logging.Logger, exchange: str, wait_s: int) -> None:
    logger.error("%s connection lost — reconnecting in %ds...", exchange, wait_s)


def log_rate_limit(logger: logging.Logger, exchange: str, wait_s: int) -> None:
    logger.warning("%s rate limit hit — waiting %ds", exchange, wait_s)


# --- Data Events (DEBUG) ---

def log_candle(logger: logging.Logger, pair: str, o: float, h: float,
               l: float, c: float, v: float) -> None:
    logger.debug("[%s] Candle: O=%.2f H=%.2f L=%.2f C=%.2f V=%.1f", pair, o, h, l, c, v)


def log_inference(logger: logging.Logger, pair: str, model: str,
                  value: float, threshold: float) -> None:
    action = "entry" if value >= threshold else "no entry"
    logger.debug("[%s] %s: %.2f (threshold: %.2f) — %s", pair, model, value, threshold, action)


def log_tick_complete(logger: logging.Logger, pairs_count: int, elapsed_s: float) -> None:
    logger.debug("Tick completed: %d pairs in %.2fs", pairs_count, elapsed_s)
