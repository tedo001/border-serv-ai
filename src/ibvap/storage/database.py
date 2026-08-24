"""Async database engine and session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event as sa_event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ibvap.core.config import StorageConfig
from ibvap.core.logging import get_logger
from ibvap.storage.models import Base

log = get_logger(__name__)


class Database:
    """Owns the engine and hands out sessions."""

    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig()
        self._engine: AsyncEngine | None = None
        self._sessions: async_sessionmaker[AsyncSession] | None = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("database is not connected; call connect() first")
        return self._engine

    @property
    def is_sqlite(self) -> bool:
        return self.config.database_url.startswith("sqlite")

    async def connect(self) -> None:
        """Create the engine and ensure the schema exists."""
        url = self.config.database_url
        if self.is_sqlite:
            # Create the parent directory for a file-backed database; a fresh
            # node has no data/ directory yet and SQLite will not make one.
            path = url.split("///", 1)[-1]
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_async_engine(
            url,
            echo=self.config.echo_sql,
            future=True,
            # SQLite's async driver serialises writes anyway; a pool adds
            # nothing but connection churn.
            pool_pre_ping=not self.is_sqlite,
        )

        if self.is_sqlite:
            self._configure_sqlite(self._engine)

        self._sessions = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        log.info("database_connected", url=_redact_url(url))

    @staticmethod
    def _configure_sqlite(engine: AsyncEngine) -> None:
        """Apply the pragmas an embedded write-heavy workload needs.

        A camera node writes events continuously while the API reads them for
        the console. In SQLite's default rollback-journal mode those block each
        other, and the console stutters every time an alert lands. WAL removes
        that contention; ``synchronous=NORMAL`` is the standard companion,
        trading a theoretical loss of the last transaction on power failure for
        an order of magnitude in write throughput - the right trade when the
        authoritative copy of an alert is the one already forwarded to C2.
        """

        @sa_event.listens_for(engine.sync_engine, "connect")
        def _set_pragmas(dbapi_connection, _record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                # Wait rather than fail immediately when another writer holds
                # the lock; a transient conflict must not drop an event.
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

    async def disconnect(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessions = None
            log.info("database_disconnected")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session, committing on success and rolling back on error."""
        if self._sessions is None:
            raise RuntimeError("database is not connected; call connect() first")
        async with self._sessions() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def __aenter__(self) -> Database:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()


def _redact_url(url: str) -> str:
    """Strip credentials from a database URL before logging it."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"
