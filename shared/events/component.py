"""Component base class — auto-subscribes on_* methods to event bus.

Inspired by aat's handler naming convention.

Example:
    class MyTracker(Component):
        async def on_position_opened(self, event: PositionOpened):
            ...
        async def on_position_closed(self, event: PositionClosed):
            ...

    tracker = MyTracker()
    tracker.register(bus)
    # Auto-subscribed to "position.opened" and "position.closed"

Naming convention:
    on_position_opened  → subscribes to "position.opened"
    on_order_filled     → subscribes to "order.filled"
    on_tick_start       → subscribes to "tick.start"
    on_event            → subscribes to "*" (catch-all)
"""
import inspect
import logging

from shared.events.bus import EventBus

logger = logging.getLogger(__name__)


class Component:
    """Base class for event-driven components."""

    def register(self, bus: EventBus) -> None:
        """Auto-subscribe all on_* methods to matching topics."""
        registered = []

        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith("on_"):
                continue

            if name == "on_event":
                bus.subscribe("*", method)
                registered.append("*")
                continue

            # on_position_opened → "position.opened"
            # on_tick_start → "tick.start"
            # on_stoploss_hit → "stoploss.hit"
            parts = name[3:]  # remove "on_"
            # Find the split point: first underscore that separates namespace from action
            # position_opened → position.opened
            # order_partially_filled → order.partially_filled
            # tick_start → tick.start
            underscore_pos = parts.find("_")
            if underscore_pos > 0:
                topic = parts[:underscore_pos] + "." + parts[underscore_pos + 1:]
            else:
                topic = parts

            bus.subscribe(topic, method)
            registered.append(topic)

        if registered:
            logger.info(
                "%s registered: %s",
                type(self).__name__,
                ", ".join(registered),
            )
