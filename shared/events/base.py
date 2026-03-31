"""Event base class — all events are immutable dataclasses with a topic."""
from dataclasses import dataclass
from time import time


@dataclass(frozen=True)
class Event:
    """Base event. All events must be immutable (frozen=True).

    The topic field determines routing in the EventBus.
    Convention: "namespace.action" (e.g. "position.opened", "order.filled")
    """
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            object.__setattr__(self, "timestamp", time())

    @property
    def topic(self) -> str:
        """Derive topic from class name: PositionOpened → position.opened"""
        name = type(self).__name__
        # Split CamelCase: PositionOpened → ["Position", "Opened"]
        parts = []
        current = []
        for char in name:
            if char.isupper() and current:
                parts.append("".join(current))
                current = [char.lower()]
            else:
                current.append(char.lower())
        if current:
            parts.append("".join(current))

        if len(parts) >= 2:
            return f"{parts[0]}.{'_'.join(parts[1:])}"
        return parts[0]
