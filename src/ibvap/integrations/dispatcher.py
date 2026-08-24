"""Event dispatcher with store-and-forward.

Sits between the analytics pipeline and every outbound sink. The pipeline hands
it events synchronously from a camera thread; the dispatcher persists them,
queues them and delivers them asynchronously, so a slow or unreachable C2
endpoint can never stall a camera worker.

Store-and-forward is the defining behaviour. At a remote BOP the uplink is
intermittent by *design* - a VSAT terminal shared with voice traffic, or a
radio link that drops in weather. An alert raised during an outage is exactly
the alert that matters, so undelivered events are persisted and replayed when
the link returns rather than logged as a failed POST and forgotten.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Iterable

from ibvap.core.config import Settings
from ibvap.core.errors import IntegrationError
from ibvap.core.logging import get_logger
from ibvap.core.types import Event
from ibvap.events.payload import event_to_payload
from ibvap.integrations.base import Sink
from ibvap.integrations.mqtt import MqttSink
from ibvap.integrations.syslog import SyslogSink
from ibvap.integrations.webhook import WebhookSink
from ibvap.storage.database import Database
from ibvap.storage.repository import EventRepository, OutboxRepository
from ibvap.telemetry.metrics import Metrics

log = get_logger(__name__)

#: Retry schedule in seconds. Front-loaded so a momentary blip recovers in
#: seconds, then stretching out so a multi-hour outage is not retried every
#: few seconds for its whole duration - which would drain a battery-backed
#: node and saturate a link that is already struggling.
RETRY_BACKOFF: tuple[float, ...] = (5.0, 15.0, 60.0, 300.0, 900.0, 3600.0)


def backoff_for(attempts: int) -> float:
    """Delay before the next attempt, given how many have already failed."""
    return RETRY_BACKOFF[min(max(0, attempts), len(RETRY_BACKOFF) - 1)]


class EventDispatcher:
    """Persists events, fans them out to sinks and replays failed deliveries."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        *,
        metrics: Metrics | None = None,
        listeners: Iterable[Callable[[Event], None]] = (),
    ) -> None:
        self.settings = settings
        self.database = database
        self.metrics = metrics
        self.sinks: list[Sink] = []
        #: In-process subscribers (WebSocket broadcaster, desktop console).
        self.listeners: list[Callable[[Event], None]] = list(listeners)

        # The pipeline runs on OS threads while delivery runs on the event
        # loop, so events cross that boundary through a queue rather than by
        # calling into asyncio from a camera thread.
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=2000)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._dropped = 0
        self._delivered = 0

    # -- lifecycle --------------------------------------------------------- #

    def build_sinks(self) -> None:
        """Instantiate sinks from configuration."""
        integrations = self.settings.integrations
        self.sinks = [WebhookSink(w) for w in integrations.webhooks if w.enabled]
        if integrations.mqtt.enabled:
            self.sinks.append(MqttSink(integrations.mqtt))
        if integrations.syslog.enabled:
            self.sinks.append(SyslogSink(integrations.syslog))

    async def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        if not self.sinks:
            self.build_sinks()

        for sink in self.sinks:
            try:
                await sink.start()
            except IntegrationError as exc:
                # A sink that cannot connect at boot is normal at a remote site.
                # It stays configured; the outbox accumulates and the replay
                # loop delivers when the link comes back.
                log.warning("sink_start_failed", sink=sink.name, error=str(exc))

        self._running = True
        self._tasks = [
            asyncio.create_task(self._ingest_loop(), name="dispatcher-ingest"),
            asyncio.create_task(self._replay_loop(), name="dispatcher-replay"),
        ]
        log.info(
            "dispatcher_started",
            sinks=[s.name for s in self.sinks],
            store_and_forward=self.settings.integrations.store_and_forward,
        )

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: B014 - shutdown is best effort
                pass
        self._tasks.clear()
        for sink in self.sinks:
            await sink.close()
        log.info("dispatcher_stopped", delivered=self._delivered, dropped=self._dropped)

    # -- submission -------------------------------------------------------- #

    def submit(self, event: Event) -> None:
        """Accept an event from a camera worker thread. Never blocks.

        Called from the pipeline's threads, so it must not touch the event loop
        directly. ``call_soon_threadsafe`` hands the work across; if the loop is
        not running yet the event is dropped with a counter rather than raising
        into a camera worker.
        """
        for listener in self.listeners:
            try:
                listener(event)
            except Exception as exc:  # pragma: no cover - listener is external
                log.error("event_listener_failed", error=str(exc))

        if self._loop is None or not self._running:
            self._dropped += 1
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue, event)
        except RuntimeError:  # loop closed mid-shutdown
            self._dropped += 1

    def _enqueue(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Backpressure at this depth means the database or every sink is
            # wedged. Dropping the newest keeps the older, already-queued
            # alerts - which are the ones an operator has been waiting on.
            self._dropped += 1
            log.warning("dispatcher_queue_full", dropped_total=self._dropped)

    async def _ingest_loop(self) -> None:
        while self._running:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                await self._handle(event)
            except Exception as exc:  # pragma: no cover - defensive
                log.error("event_handling_failed", event=event.event_id, error=str(exc), exc_info=True)
            finally:
                self._queue.task_done()

    async def _handle(self, event: Event) -> None:
        """Persist an event, then queue or deliver it to each interested sink."""
        payload = self._payload(event)

        async with self.database.session() as session:
            await EventRepository(session, self.settings.site_id).add(event)

            interested = [s for s in self.sinks if s.accepts(event)]
            if self.settings.integrations.store_and_forward and interested:
                # Persist the intent to deliver *before* attempting it, so a
                # crash between the attempt and the record cannot lose an alert.
                outbox = OutboxRepository(session)
                for sink in interested:
                    await outbox.enqueue(event.event_id, sink.name, payload)

        if not self.settings.integrations.store_and_forward:
            await asyncio.gather(
                *(self._deliver_direct(s, payload) for s in self.sinks if s.accepts(event)),
                return_exceptions=True,
            )
        else:
            # Deliver immediately as well, so the common case (link healthy) is
            # not delayed by a replay tick. The outbox row is the safety net.
            await self._flush_all(limit=len(self.sinks) * 4)

    def _payload(self, event: Event) -> dict[str, Any]:
        camera = next((c for c in self.settings.cameras if c.id == event.camera_id), None)
        return event_to_payload(
            event,
            site_id=self.settings.site_id,
            site_name=self.settings.site_name,
            camera_name=camera.name if camera else event.camera_id,
            latitude=camera.latitude if camera else None,
            longitude=camera.longitude if camera else None,
        )

    async def _deliver_direct(self, sink: Sink, payload: dict[str, Any]) -> None:
        started = time.perf_counter()
        try:
            await sink.deliver(payload)
            self._delivered += 1
            if self.metrics:
                self.metrics.delivery(sink.name, "success", time.perf_counter() - started)
        except Exception as exc:
            if self.metrics:
                self.metrics.delivery(sink.name, "failure", time.perf_counter() - started)
            log.warning("delivery_failed", sink=sink.name, error=str(exc))

    # -- replay ------------------------------------------------------------ #

    async def _replay_loop(self) -> None:
        """Periodically drain the outbox for every sink."""
        interval = self.settings.integrations.outbox_flush_interval_seconds
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self._flush_all()
                await self._prune_outbox()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # pragma: no cover - defensive
                log.error("replay_loop_error", error=str(exc), exc_info=True)

    async def _flush_all(self, limit: int = 50) -> int:
        """Attempt delivery of pending outbox rows across all sinks."""
        if not self.settings.integrations.store_and_forward:
            return 0
        total = 0
        for sink in self.sinks:
            if not sink.enabled:
                continue
            total += await self._flush_sink(sink, limit)
        return total

    async def _flush_sink(self, sink: Sink, limit: int) -> int:
        async with self.database.session() as session:
            outbox = OutboxRepository(session)
            pending = await outbox.pending(sink.name, limit=limit)
            depth = await outbox.depth(sink.name)
        if self.metrics:
            self.metrics.set_outbox_depth(sink.name, depth)
        if not pending:
            return 0

        delivered = 0
        for record in pending:
            started = time.perf_counter()
            try:
                await sink.deliver(record.payload)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                if self.metrics:
                    self.metrics.delivery(sink.name, "failure", elapsed)
                async with self.database.session() as session:
                    await OutboxRepository(session).mark_failed(
                        record.id, str(exc), backoff_for(record.attempts)
                    )
                # Stop on the first failure for this sink: the endpoint is
                # down, and hammering it with the rest of the backlog achieves
                # nothing but burning the link and inflating attempt counts.
                break

            elapsed = time.perf_counter() - started
            delivered += 1
            self._delivered += 1
            if self.metrics:
                self.metrics.delivery(sink.name, "success", elapsed)
            async with self.database.session() as session:
                await OutboxRepository(session).mark_delivered(record.id)

        if delivered:
            log.debug("outbox_flushed", sink=sink.name, delivered=delivered)
        return delivered

    async def _prune_outbox(self) -> None:
        async with self.database.session() as session:
            await OutboxRepository(session).prune(
                max_pending=self.settings.integrations.outbox_max_events
            )

    # -- introspection ----------------------------------------------------- #

    async def status(self) -> dict[str, Any]:
        depths: dict[str, int] = {}
        async with self.database.session() as session:
            outbox = OutboxRepository(session)
            for sink in self.sinks:
                depths[sink.name] = await outbox.depth(sink.name)
        return {
            "running": self._running,
            "queue_depth": self._queue.qsize(),
            "delivered": self._delivered,
            "dropped": self._dropped,
            "store_and_forward": self.settings.integrations.store_and_forward,
            "sinks": [{**s.status(), "outbox_depth": depths.get(s.name, 0)} for s in self.sinks],
        }
