import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ownbot.loggers.rich_handler import OwnBotRichHandler
from ownbot.loggers.buffering_handler import OwnBotBufferingHandler


# Shared buffering handler so other modules can access log records
buffering_handler = OwnBotBufferingHandler(capacity=1000)

# Third-party loggers to keep quiet
NOISY_LOGGERS = [
    "ccxt",
    "urllib3",
    "requests",
    "asyncio",
    "httpcore",
    "httpx",
    "telegram",
    "websockets",
    "huggingface_hub",
]


def setup_logging(
    verbosity: int = 0,
    logfile: str | None = None,
    logfile_max_mb: int = 10,
    logfile_backups: int = 5,
) -> None:
    """Configure logging for OwnBot.

    Args:
        verbosity: 0=WARNING, 1=INFO, 2+=DEBUG
        logfile: Optional path to a log file (always DEBUG level, rotating)
        logfile_max_mb: Max log file size in MB before rotation
        logfile_backups: Number of old log files to keep
    """
    # Map verbosity to level
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # always DEBUG at root, handlers filter

    # Clear existing handlers (avoid duplicates on reload)
    root.handlers.clear()

    # Rich console handler → stderr (so backtester stdout stays clean)
    rich_handler = OwnBotRichHandler(level=level)
    rich_handler.stream = sys.stderr
    root.addHandler(rich_handler)

    # Buffering handler (for future API)
    buffering_handler.setLevel(logging.DEBUG)
    root.addHandler(buffering_handler)

    # Optional file handler (always DEBUG, rotating)
    if logfile:
        log_path = Path(logfile)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            logfile,
            maxBytes=logfile_max_mb * 1_000_000,
            backupCount=logfile_backups,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
