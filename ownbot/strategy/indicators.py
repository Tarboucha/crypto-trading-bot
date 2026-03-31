"""Thin wrappers around pandas-ta for a consistent API.

All functions: DataFrame in → DataFrame out. No side effects.
"""
import pandas as pd
import pandas_ta as ta


def sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.DataFrame:
    """Simple Moving Average."""
    df[f"sma_{period}"] = ta.sma(df[column], length=period)
    return df


def ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.DataFrame:
    """Exponential Moving Average."""
    df[f"ema_{period}"] = ta.ema(df[column], length=period)
    return df


def rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
    """Relative Strength Index."""
    df[f"rsi_{period}"] = ta.rsi(df[column], length=period)
    return df


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence)."""
    macd_df = ta.macd(df[column], fast=fast, slow=slow, signal=signal)
    if macd_df is not None:
        df["macd"] = macd_df.iloc[:, 0]
        df["macd_signal"] = macd_df.iloc[:, 1]
        df["macd_hist"] = macd_df.iloc[:, 2]
    return df


def bbands(
    df: pd.DataFrame, period: int = 20, std: float = 2.0, column: str = "close"
) -> pd.DataFrame:
    """Bollinger Bands."""
    bb = ta.bbands(df[column], length=period, std=std)
    if bb is not None:
        df["bb_lower"] = bb.iloc[:, 0]
        df["bb_middle"] = bb.iloc[:, 1]
        df["bb_upper"] = bb.iloc[:, 2]
    return df


def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range."""
    df[f"atr_{period}"] = ta.atr(df["high"], df["low"], df["close"], length=period)
    return df


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume Simple Moving Average."""
    df[f"vol_sma_{period}"] = ta.sma(df["volume"], length=period)
    return df
