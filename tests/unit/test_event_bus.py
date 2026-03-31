"""Tests for EventBus — dispatch, wildcards, error isolation, auto-subscribe."""
import pytest
from shared.events.bus import EventBus
from shared.events.component import Component
from shared.events.trading import PositionOpened, PositionClosed, SignalEntry, OrderFilled
from shared.events.system import TickStart, EngineStarted
from shared.events.commands import StopCommand, ForceCloseCommand


class TestTopicDerivation:

    def test_position_opened(self):
        assert PositionOpened().topic == "position.opened"

    def test_position_closed(self):
        assert PositionClosed().topic == "position.closed"

    def test_signal_entry(self):
        assert SignalEntry().topic == "signal.entry"

    def test_tick_start(self):
        assert TickStart().topic == "tick.start"

    def test_command_explicit_topic(self):
        assert StopCommand().topic == "command.stop"
        assert ForceCloseCommand().topic == "command.force_close"


@pytest.mark.asyncio
class TestDispatch:

    async def test_exact_match(self, bus):
        received = []
        async def handler(event):
            received.append(event)
        bus.subscribe("position.opened", handler)
        await bus.publish(PositionOpened(pair="ETH"))
        assert len(received) == 1
        assert received[0].pair == "ETH"

    async def test_no_match(self, bus):
        received = []
        async def handler(event):
            received.append(event)
        bus.subscribe("position.opened", handler)
        await bus.publish(SignalEntry(pair="ETH"))
        assert len(received) == 0

    async def test_multiple_subscribers(self, bus):
        results = []
        async def handler_a(event):
            results.append("a")
        async def handler_b(event):
            results.append("b")
        bus.subscribe("position.opened", handler_a)
        bus.subscribe("position.opened", handler_b)
        await bus.publish(PositionOpened(pair="ETH"))
        assert results == ["a", "b"]  # FIFO order

    async def test_multiple_events(self, bus):
        received = []
        async def handler(event):
            received.append(event.topic)
        bus.subscribe("position.opened", handler)
        bus.subscribe("position.closed", handler)
        await bus.publish(PositionOpened(pair="ETH"))
        await bus.publish(PositionClosed(pair="ETH"))
        assert received == ["position.opened", "position.closed"]


@pytest.mark.asyncio
class TestWildcard:

    async def test_namespace_wildcard(self, bus):
        received = []
        async def handler(event):
            received.append(event.topic)
        bus.subscribe("position.*", handler)
        await bus.publish(PositionOpened(pair="ETH"))
        await bus.publish(PositionClosed(pair="ETH"))
        await bus.publish(SignalEntry(pair="ETH"))  # should not match
        assert len(received) == 2
        assert "signal.entry" not in received

    async def test_catch_all(self, bus):
        received = []
        async def handler(event):
            received.append(event.topic)
        bus.subscribe("*", handler)
        await bus.publish(PositionOpened(pair="ETH"))
        await bus.publish(SignalEntry(pair="BTC"))
        await bus.publish(TickStart())
        assert len(received) == 3


@pytest.mark.asyncio
class TestErrorIsolation:

    async def test_handler_error_doesnt_break_others(self, bus):
        results = []
        async def bad_handler(event):
            raise ValueError("boom")
        async def good_handler(event):
            results.append("ok")
        bus.subscribe("position.opened", bad_handler)
        bus.subscribe("position.opened", good_handler)
        await bus.publish(PositionOpened(pair="ETH"))
        assert results == ["ok"]  # good handler still ran

    async def test_event_count_increments_on_error(self, bus):
        async def bad_handler(event):
            raise ValueError("boom")
        bus.subscribe("position.opened", bad_handler)
        await bus.publish(PositionOpened(pair="ETH"))
        assert bus.event_count == 1


@pytest.mark.asyncio
class TestComponentAutoSubscribe:

    async def test_auto_subscribe(self, bus):
        class TestComp(Component):
            def __init__(self):
                self.opened = []
                self.closed = []
            async def on_position_opened(self, event):
                self.opened.append(event.pair)
            async def on_position_closed(self, event):
                self.closed.append(event.pair)

        comp = TestComp()
        comp.register(bus)

        await bus.publish(PositionOpened(pair="ETH"))
        await bus.publish(PositionClosed(pair="BTC"))

        assert comp.opened == ["ETH"]
        assert comp.closed == ["BTC"]

    async def test_catch_all_via_on_event(self, bus):
        class CatchAll(Component):
            def __init__(self):
                self.count = 0
            async def on_event(self, event):
                self.count += 1

        comp = CatchAll()
        comp.register(bus)
        await bus.publish(PositionOpened(pair="ETH"))
        await bus.publish(SignalEntry(pair="BTC"))
        assert comp.count == 2


class TestBusMetrics:

    def test_subscriber_count(self, bus):
        async def h(e): pass
        bus.subscribe("a", h)
        bus.subscribe("b", h)
        bus.subscribe("b", h)
        assert bus.subscriber_count == 3

    def test_topics(self, bus):
        async def h(e): pass
        bus.subscribe("position.opened", h)
        bus.subscribe("signal.entry", h)
        assert sorted(bus.topics()) == ["position.opened", "signal.entry"]

    @pytest.mark.asyncio
    async def test_event_count(self, bus):
        async def h(e): pass
        bus.subscribe("position.opened", h)
        await bus.publish(PositionOpened(pair="ETH"))
        await bus.publish(PositionOpened(pair="BTC"))
        assert bus.event_count == 2
