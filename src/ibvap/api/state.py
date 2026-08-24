"""Application state and lifecycle.

One object owns every long-lived subsystem - supervisor, database, dispatcher,
evidence store, token service - and wires them together. Routers reach it
through dependency injection rather than module-level globals, so tests can
build an isolated instance and the desktop console can embed the same runtime
in-process without an HTTP server at all.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from ibvap.api.security import (
    LoginThrottle,
    Role,
    TokenService,
    generate_password,
    hash_password,
)
from ibvap.core.config import Settings
from ibvap.core.logging import get_logger
from ibvap.core.types import Event, Frame, Track
from ibvap.events.evidence import EvidenceStore
from ibvap.integrations.dispatcher import EventDispatcher
from ibvap.mlops.registry import ModelRegistry
from ibvap.pipeline.supervisor import Supervisor
from ibvap.storage.database import Database
from ibvap.storage.repository import (
    EventRepository,
    UserRepository,
    WatchlistRepository,
)
from ibvap.telemetry.metrics import Metrics, get_metrics
from ibvap.vision.anpr import PlateWatchlist

log = get_logger(__name__)


class AppState:
    """Owns and sequences every long-lived subsystem on a node."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metrics: Metrics = get_metrics(settings.site_id, "1.0.0")
        self.database = Database(settings.storage)
        self.registry = ModelRegistry.load(settings.models)
        self.evidence = EvidenceStore(
            settings.evidence, settings.privacy, site_id=settings.site_id
        )
        self.tokens = TokenService(settings.security)
        self.throttle = LoginThrottle(
            settings.security.max_login_attempts, settings.security.lockout_seconds
        )

        self.dispatcher: EventDispatcher | None = None
        self.supervisor: Supervisor | None = None
        #: Subscribers for the live WebSocket alert feed.
        self.subscribers: set[Callable[[Event], None]] = set()

        self._started_at = 0.0
        self._janitor: asyncio.Task[None] | None = None
        self._bootstrap_password: str | None = None

    # -- lifecycle --------------------------------------------------------- #

    async def startup(self) -> None:
        """Bring the node up in dependency order."""
        self._started_at = time.time()
        await self.database.connect()
        await self._bootstrap_admin()

        self.dispatcher = EventDispatcher(
            self.settings, self.database, metrics=self.metrics,
            listeners=[self._fan_out],
        )
        await self.dispatcher.start()

        self.supervisor = Supervisor(
            self.settings,
            event_sink=self._on_event,
            metrics=self.metrics,
            registry=self.registry,
            frame_sink=self._on_frame,
        )
        # Watchlists are loaded before cameras start, so the very first vehicle
        # through the gate is checked against them rather than missed during a
        # warm-up window.
        await self._load_watchlists()
        self.supervisor.start()

        self._janitor = asyncio.create_task(self._janitor_loop(), name="janitor")
        log.info(
            "node_started",
            site=self.settings.site_id,
            cameras=len(self.supervisor.workers),
            tier=self.settings.tier,
        )

    async def shutdown(self) -> None:
        """Tear down in reverse order."""
        if self._janitor:
            self._janitor.cancel()
            try:
                await self._janitor
            except (asyncio.CancelledError, Exception):  # noqa: B014
                pass
            self._janitor = None

        if self.supervisor:
            # Stop cameras first so no new events arrive while the dispatcher
            # is draining what it already holds.
            self.supervisor.stop()
            self.supervisor = None
        if self.dispatcher:
            await self.dispatcher.stop()
            self.dispatcher = None

        self.registry.close()
        await self.database.disconnect()
        log.info("node_stopped", site=self.settings.site_id)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at if self._started_at else 0.0

    @property
    def bootstrap_password(self) -> str | None:
        """A generated bootstrap password, available once for the operator."""
        return self._bootstrap_password

    # -- event flow -------------------------------------------------------- #

    def _on_event(self, event: Event) -> None:
        """Receive an event from a camera worker thread.

        Runs on a camera thread, so it must stay non-blocking. Evidence capture
        is the one exception - it needs the frame while it is still in memory,
        and the frame is gone by the time an async task would run.
        """
        worker = self.supervisor.worker(event.camera_id) if self.supervisor else None
        if worker is not None and self.settings.evidence.enabled and event.boxes:
            latest = worker.latest()
            if latest is not None:
                frame, tracks = latest
                self.evidence.capture(
                    event, frame,
                    tracks=tracks,
                    zones=list(worker.analytics.zones.values()),
                    tripwires=list(worker.analytics.tripwires.values()),
                    camera_name=worker.camera.name,
                )
        if self.dispatcher is not None:
            self.dispatcher.submit(event)

    def _on_frame(self, frame: Frame, tracks: list[Track]) -> None:
        """Feed frames into the evidence pre-buffer."""
        self.evidence.observe(frame)

    def _fan_out(self, event: Event) -> None:
        """Push an event to every live subscriber (WebSocket, console)."""
        for subscriber in list(self.subscribers):
            try:
                subscriber(event)
            except Exception as exc:  # pragma: no cover - subscriber is external
                log.debug("subscriber_failed", error=str(exc))

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Register a live-event subscriber; returns an unsubscribe callable."""
        self.subscribers.add(callback)
        return lambda: self.subscribers.discard(callback)

    # -- bootstrap --------------------------------------------------------- #

    async def _bootstrap_admin(self) -> None:
        """Create the first administrator if no users exist.

        A generated password is used unless one is supplied through the
        environment. It is surfaced once, at start-up, and the account is
        flagged to force a change on first login - so a fleet of nodes never
        ends up sharing one well-known default credential.
        """
        async with self.database.session() as session:
            users = UserRepository(session)
            if await users.count() > 0:
                return

            configured = self.settings.security.bootstrap_admin_password
            password = configured or generate_password()
            await users.create(
                self.settings.security.bootstrap_admin_user,
                hash_password(password),
                role=Role.ADMIN.value,
                full_name="Bootstrap Administrator",
                must_change_password=not configured,
            )

        if not configured:
            self._bootstrap_password = password
            # Printed rather than logged: log files are shipped to a central
            # collector, and a credential must not travel with them.
            print(
                "\n"
                "  ================================================================\n"
                "   IBVAP bootstrap administrator created\n"
                f"   username: {self.settings.security.bootstrap_admin_user}\n"
                f"   password: {password}\n"
                "   This password is shown once and must be changed at first login.\n"
                "  ================================================================\n",
                flush=True,
            )
        log.info(
            "bootstrap_admin_created", username=self.settings.security.bootstrap_admin_user
        )

    async def _load_watchlists(self) -> None:
        """Populate in-memory watchlists from the database."""
        if self.supervisor is None:
            return
        async with self.database.session() as session:
            repository = WatchlistRepository(session)

            gallery = self.supervisor.face_gallery
            gallery.clear()
            for record in await repository.list_faces():
                embeddings = WatchlistRepository.decode_embeddings(record)
                for vector in embeddings:
                    gallery.enroll(
                        record.person_id, vector,
                        name=record.name, category=record.category, notes=record.notes,
                    )

            watchlist: PlateWatchlist = self.supervisor.plate_watchlist
            watchlist.clear()
            now = time.time()
            for plate in await repository.list_plates():
                if plate.expires_at and plate.expires_at < now:
                    continue
                watchlist.add(
                    plate.plate, category=plate.category,
                    reason=plate.reason, reference=plate.reference,
                )

        log.info(
            "watchlists_loaded",
            faces=self.supervisor.face_gallery.size,
            plates=self.supervisor.plate_watchlist.size,
        )

    async def reload_watchlists(self) -> None:
        """Re-read watchlists after an API change."""
        await self._load_watchlists()

    # -- housekeeping ------------------------------------------------------ #

    async def _janitor_loop(self) -> None:
        """Periodic retention enforcement.

        Runs hourly rather than on every write: pruning is I/O heavy and an
        edge node's disk is better used serving live analytics than being
        walked continuously.
        """
        while True:
            try:
                await asyncio.sleep(3600)
                await self._run_janitor()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # pragma: no cover - defensive
                log.error("janitor_failed", error=str(exc), exc_info=True)

    async def _run_janitor(self) -> None:
        removed_events = 0
        if self.settings.storage.event_retention_days > 0:
            async with self.database.session() as session:
                removed_events = await EventRepository(session).prune(
                    self.settings.storage.event_retention_days * 86400
                )
        pruned = await asyncio.to_thread(self.evidence.prune)
        log.info(
            "janitor_run",
            events_removed=removed_events,
            evidence_files_removed=pruned["files"],
            evidence_megabytes_freed=round(pruned["bytes"] / 1024**2, 1),
        )

    # -- reporting --------------------------------------------------------- #

    async def health(self) -> dict[str, Any]:
        """Full node health, aggregating every subsystem."""
        base: dict[str, Any] = {
            "status": "starting",
            "version": "1.0.0",
            "site_id": self.settings.site_id,
            "site_name": self.settings.site_name,
            "tier": self.settings.tier,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "cameras_total": 0,
            "cameras_online": 0,
            "models": {},
            "watchlists": {"faces": 0, "plates": 0},
        }
        if self.supervisor is not None:
            base.update(self.supervisor.health())
            base["version"] = "1.0.0"

        base["storage"] = {
            "evidence": self.evidence.usage(),
            "database_url": _redact(self.settings.storage.database_url),
        }
        if self.dispatcher is not None:
            base["integrations"] = await self.dispatcher.status()
        return base


def _redact(url: str) -> str:
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"
