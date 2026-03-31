import logging
from logging.handlers import BufferingHandler


class OwnBotBufferingHandler(BufferingHandler):
    """In-memory log buffer for future API endpoints.

    On flush, keeps half the buffer instead of clearing it,
    so the /logs endpoint never returns empty.
    """

    def flush(self) -> None:
        self.acquire()
        try:
            half = len(self.buffer) // 2
            self.buffer = self.buffer[half:]
        finally:
            self.release()

    def get_records(self) -> list[logging.LogRecord]:
        """Return a copy of buffered records."""
        self.acquire()
        try:
            return list(self.buffer)
        finally:
            self.release()
