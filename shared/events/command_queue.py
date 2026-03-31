"""Thread-safe command queue — bridge between REST API thread and async event loop.

The REST API runs in a separate thread. It can't directly publish to the
async EventBus. Instead, it puts commands in this queue, and the engine
drains the queue on each tick.

Usage:
    # REST API thread:
    cmd_queue.put_from_thread(ForceCloseCommand(pair="ETH"))

    # Engine tick (async):
    await cmd_queue.process(bus)
"""
import asyncio
import logging

from shared.events.base import Event
from shared.events.bus import EventBus

logger = logging.getLogger(__name__)


class CommandQueue:

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind to the running event loop. Call from async context."""
        self._loop = loop

    def put_from_thread(self, event: Event) -> None:
        """Put a command from a non-async thread (e.g., REST API).

        Thread-safe. The event will be processed on the next engine tick.
        """
        if self._loop is None:
            logger.warning("CommandQueue not bound to event loop — dropping %s", event.topic)
            return

        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
        logger.debug("Command queued from thread: %s", event.topic)

    async def process(self, bus: EventBus) -> int:
        """Drain the queue and publish all pending commands to the bus.

        Returns number of commands processed.
        """
        count = 0
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                await bus.publish(event)
                count += 1
            except asyncio.QueueEmpty:
                break
        return count
