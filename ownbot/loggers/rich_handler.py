import logging
from datetime import datetime

from rich.text import Text

from ownbot.ui.console import error_console


LEVEL_STYLES = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
}


class OwnBotRichHandler(logging.Handler):
    """Custom Rich-based log handler with styled output."""

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.console = error_console

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Timestamp
            timestamp = datetime.fromtimestamp(record.created).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            ts_text = Text(timestamp, style="dim")

            # Logger name
            name_text = Text(f" {record.name:<30s}", style="violet")

            # Level
            level_style = LEVEL_STYLES.get(record.levelname, "")
            level_text = Text(f" {record.levelname:<8s}", style=level_style)

            # Message
            msg = self.format(record)
            msg_text = Text(f" {msg}")

            # Combine and print
            line = Text()
            line.append_text(ts_text)
            line.append_text(name_text)
            line.append_text(level_text)
            line.append_text(msg_text)

            self.console.print(line)
        except Exception:
            self.handleError(record)
