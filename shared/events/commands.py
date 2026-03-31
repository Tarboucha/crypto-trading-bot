"""Command events — external requests to the engine (REST API, CLI)."""
from dataclasses import dataclass

from shared.events.base import Event


@dataclass(frozen=True)
class ForceCloseCommand(Event):
    """Emergency close a position."""
    pair: str = ""
    reason: str = "force_close"

    @property
    def topic(self) -> str:
        return "command.force_close"


@dataclass(frozen=True)
class PauseCommand(Event):
    """Pause trading — no new entries, keep existing positions."""
    reason: str = "user"

    @property
    def topic(self) -> str:
        return "command.pause"


@dataclass(frozen=True)
class ResumeCommand(Event):
    """Resume trading after pause."""

    @property
    def topic(self) -> str:
        return "command.resume"


@dataclass(frozen=True)
class StopCommand(Event):
    """Graceful shutdown."""
    reason: str = "user"

    @property
    def topic(self) -> str:
        return "command.stop"
