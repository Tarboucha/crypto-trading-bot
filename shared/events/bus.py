"""Async event bus — publish/subscribe with topic namespaces and wildcards.

Inspired by NautilusTrader MessageBus and pymitter.

Supports:
    - Exact topic:  "position.opened"
    - Wildcard:     "position.*" (all position events)
    - Catch-all:    "*" (every event)

All handlers are async. Handlers that raise are logged but don't break the bus.
"""
import asyncio
import logging
from collections import defaultdict
from typing import Callable, Awaitable

from shared.events.base import Event

logger = logging.getLogger(__name__)

Handler = Callable[[Event], Awaitable[None]]


class EventBus:

    def __init__(self):
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._event_count: int = 0

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Subscribe a handler to a topic.

        Args:
            topic: Event topic. Supports wildcards:
                   "position.opened" — exact match
                   "position.*" — all position events
                   "*" — all events
            handler: Async callable that receives an Event
        """
        self._subscribers[topic].append(handler)
        logger.debug("Subscribed %s to '%s'", handler.__qualname__, topic)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        """Remove a handler from a topic."""
        handlers = self._subscribers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to all matching subscribers.

        Matching order: exact → wildcard (namespace.*) → catch-all (*)
        Handlers that raise are logged but don't stop other handlers.
        """
        topic = event.topic
        self._event_count += 1
        handlers = []

        # 1. Exact match
        handlers.extend(self._subscribers.get(topic, []))

        # 2. Wildcard: "position.*" matches "position.opened"
        if "." in topic:
            namespace = topic.rsplit(".", 1)[0] + ".*"
            handlers.extend(self._subscribers.get(namespace, []))

        # 3. Catch-all
        handlers.extend(self._subscribers.get("*", []))

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(
                    "Handler %s failed on '%s': %s",
                    handler.__qualname__, topic, e,
                )

    @property
    def subscriber_count(self) -> int:
        return sum(len(h) for h in self._subscribers.values())

    @property
    def event_count(self) -> int:
        return self._event_count

    def topics(self) -> list[str]:
        """List all topics with active subscribers."""
        return [t for t, h in self._subscribers.items() if h]
