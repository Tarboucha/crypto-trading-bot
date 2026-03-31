import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from shared.api.exchange.hyperliquid import ExchangeConfig


@dataclass
class RiskConfig:
    max_drawdown_pct: float = 8.0
    risk_per_trade_pct: float = 0.3
    max_exposure_pct: float = 4.0
    loss_limit_pct: float = 2.0
    loss_limit_reset: str = "daily"  # "hourly" | "4h" | "8h" | "daily" | "weekly" | "session"


@dataclass
class StrategyConfig:
    name: str = "trend_follow"
    params: dict = field(default_factory=dict)


@dataclass
class CostsConfig:
    fee_pct: float = 0.035
    slippage_pct: float = 0.01
    spread_mode: str = "estimated"
    spread_fixed_pct: float = 0.01
    spread_factor: float = 0.1


@dataclass
class MLConfig:
    # Training
    train_pairs: list[str] = field(default_factory=lambda: ["ETH", "BTC", "SOL", "DOGE", "XRP", "AVAX", "LINK", "ADA"])
    train_timeframe: str = "5m"
    training_data_path: str = "data/ml_training_data.csv"
    model_name: str = "rsi_filter_v1.json"
    # Labeling
    label_window: int = 100
    label_stoploss: float = -0.5
    label_take_profit: float = 1.0
    label_max_candles: int = 20
    # XGBoost
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 20
    # Split
    train_ratio: float = 0.67
    val_ratio: float = 0.17
    # Device
    device: str = "cuda"
    # Inference
    ml_threshold: float = 0.60


@dataclass
class LoggingConfig:
    verbosity: int = 1
    logfile: str | None = None


@dataclass
class BotConfig:
    mode: str = "paper"  # paper | live | backtest
    timeframe: str = "5m"
    pairs: list = field(default_factory=list)  # list[TradingPair]
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} with environment variable value. Warns if missing."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        resolved = os.environ.get(var_name)
        if resolved is None:
            import logging
            logging.getLogger("ownbot.config").warning(
                "%s not set — this may limit functionality", var_name
            )
            return ""
        return resolved
    return value


def load_config(config_path: str = "config.toml") -> BotConfig:
    """Load config from TOML file and resolve env vars for secrets."""
    # Load .env file if it exists
    env_path = Path(config_path).parent / ".env"
    load_dotenv(env_path)

    # Read TOML
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Build exchange config with env var resolution
    exchange_raw = raw.get("exchange", {})
    exchange = ExchangeConfig(
        name=exchange_raw.get("name", "hyperliquid"),
        sandbox=exchange_raw.get("sandbox", True),
        key=_resolve_env_vars(exchange_raw.get("key", "")),
        secret=_resolve_env_vars(exchange_raw.get("secret", "")),
    )

    # Build risk config
    risk_raw = raw.get("risk", {})
    risk = RiskConfig(**{k: risk_raw[k] for k in risk_raw if hasattr(RiskConfig, k)})

    # Build strategy config
    strat_raw = raw.get("strategy", {})
    strategy_name = strat_raw.get("name", "trend_follow")

    # Load strategy-specific TOML if it exists
    strategy_toml_path = Path(config_path).parent / "strategy" / "configs" / f"{strategy_name}.toml"
    strategy_params = {}
    if strategy_toml_path.exists():
        with open(strategy_toml_path, "rb") as f:
            strategy_params = tomllib.load(f)

    # CLI/inline params override strategy TOML
    inline_params = strat_raw.get("params", {})
    strategy_params.update(inline_params)

    strategy = StrategyConfig(
        name=strategy_name,
        params=strategy_params,
    )

    # Build costs config
    costs_raw = raw.get("costs", {})
    costs = CostsConfig(
        fee_pct=costs_raw.get("fee_pct", 0.035),
        slippage_pct=costs_raw.get("slippage_pct", 0.01),
        spread_mode=costs_raw.get("spread_mode", "estimated"),
        spread_fixed_pct=costs_raw.get("spread_fixed_pct", 0.01),
        spread_factor=costs_raw.get("spread_factor", 0.1),
    )

    # Build ML config
    ml_raw = raw.get("ml", {})
    ml = MLConfig(
        train_pairs=ml_raw.get("train_pairs", ["ETH", "BTC", "SOL", "DOGE", "XRP", "AVAX", "LINK", "ADA"]),
        train_timeframe=ml_raw.get("train_timeframe", "5m"),
        training_data_path=ml_raw.get("training_data_path", "data/ml_training_data.csv"),
        model_name=ml_raw.get("model_name", "rsi_filter_v1.json"),
        label_window=ml_raw.get("label_window", 100),
        label_stoploss=ml_raw.get("label_stoploss", -0.5),
        label_take_profit=ml_raw.get("label_take_profit", 1.0),
        label_max_candles=ml_raw.get("label_max_candles", 20),
        n_estimators=ml_raw.get("n_estimators", 200),
        max_depth=ml_raw.get("max_depth", 4),
        learning_rate=ml_raw.get("learning_rate", 0.05),
        min_child_weight=ml_raw.get("min_child_weight", 10),
        subsample=ml_raw.get("subsample", 0.8),
        colsample_bytree=ml_raw.get("colsample_bytree", 0.8),
        reg_alpha=ml_raw.get("reg_alpha", 0.1),
        reg_lambda=ml_raw.get("reg_lambda", 1.0),
        early_stopping_rounds=ml_raw.get("early_stopping_rounds", 20),
        train_ratio=ml_raw.get("train_ratio", 0.67),
        val_ratio=ml_raw.get("val_ratio", 0.17),
        device=ml_raw.get("device", "cuda"),
        ml_threshold=ml_raw.get("ml_threshold", 0.60),
    )

    # Build logging config
    log_raw = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        verbosity=log_raw.get("verbosity", 1),
        logfile=log_raw.get("logfile"),
    )

    # Parse and validate pairs
    from shared.pairs import parse_pairs, validate_pairs
    trading_mode = raw.get("trading_mode", "futures")
    settle = raw.get("settle", "USDT")
    pairs = parse_pairs(raw.get("pairs", ["ETH/USDT", "BTC/USDT"]), trading_mode, settle)
    validate_pairs(pairs)

    return BotConfig(
        mode=raw.get("mode", "paper"),
        timeframe=raw.get("timeframe", "5m"),
        pairs=pairs,
        exchange=exchange,
        risk=risk,
        strategy=strategy,
        costs=costs,
        ml=ml,
        logging=logging_cfg,
    )
