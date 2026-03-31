# crypto-trading-bot

An event-driven cryptocurrency trading bot for perpetual futures on Hyperliquid. Supports live trading, paper trading, and backtesting with pluggable strategies and integrated risk management.

## Architecture

```
ownbot/          Main trading bot (engine, strategies, backtester, config)
shared/          Shared kernel (event bus, database, exchange API, pairs, costs)
datacollector/   Standalone service to backfill OHLCV candles from exchanges
scripts/         Utility scripts (data download from Binance, Hyperliquid)
tests/           Unit and integration tests
```

### Key Design Patterns

- **Event-driven** — central `EventBus` with pub/sub decouples all components (strategies, risk manager, position manager, trade persister)
- **Async I/O** — non-blocking exchange and database operations throughout
- **Strategy interface** — all strategies implement `BaseStrategy`; works identically in live, paper, and backtest modes
- **Risk management** — per-trade sizing, max drawdown, exposure limits, daily loss limits

## Prerequisites

- Python 3.12+
- PostgreSQL (async via `asyncpg`)
- Hyperliquid API keys (for live/paper trading)

## Setup

```bash
# Clone and create virtual environment
git clone git@github.com:Tarboucha/crypto-trading-bot.git
cd crypto-trading-bot
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your credentials:
#   HYPERLIQUID_KEY=...
#   HYPERLIQUID_SECRET=...
#   DATABASE_URL=postgresql+asyncpg://user:pass@host/db
```

## Configuration

Edit `ownbot/config.toml` to configure:

- **mode** — `paper`, `live`, or `backtest`
- **pairs** — trading pairs (e.g., `["ETH/USDT", "BTC/USDT"]`)
- **timeframe** — candle timeframe (e.g., `5m`)
- **strategy** — which strategy to run
- **risk** — drawdown limits, position sizing, exposure caps
- **costs** — fees, slippage, spread modeling

Strategy-specific parameters are in `ownbot/strategy/configs/`.

## Usage

### Trading (live or paper)
```bash
python -m ownbot trade
python -m ownbot trade --config custom.toml -vv
```

### Backtesting
```bash
python -m ownbot backtest --strategy rsi_mean_reversion --days 7 --balance 10000 --pairs ETH BTC
```

### Data Collection
```bash
# Continuous candle collection to database
python -m datacollector

# Single run
python -m datacollector --once
```

### Download Historical Data
```bash
python scripts/download_binance_data.py
python scripts/download_hyperliquid_data.py
python scripts/download_funding_rates.py
```

## Available Strategies

| Strategy | Description |
|----------|-------------|
| `trend_follow` | Trend-following with moving averages |
| `rsi_mean_reversion` | RSI oversold/overbought with Bollinger Band confirmation |
| `ml_filtered_rsi` | RSI signals filtered by an XGBoost win probability model |

## Project Structure

```
ownbot/
  engine/           Trading engine, position manager, risk manager, executor
  strategy/         Strategy implementations and configs
  backtester/       Historical backtest runner and reporting
  config/           TOML-based configuration with dataclasses
  loggers/          Logging setup (Rich terminal output)
  ui/               Console display utilities

shared/
  events/           Event bus, trading/system/command events, base component
  db/               SQLAlchemy models and async repositories (candles, trades, signals)
  api/exchange/     Exchange abstraction (Hyperliquid adapter)
  pairs.py          Unified trading pair system
  costs.py          Fee and slippage modeling

datacollector/
  collector.py      Candle fetcher with backfill and continuous polling
  main.py           Entry point

scripts/
  download_*.py     Historical data download utilities
```
