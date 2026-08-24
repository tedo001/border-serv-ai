"""Structured logging.

Field deployments split into two worlds: a BOP node whose logs a soldier reads
on a console, and a sector node whose logs are shipped to a central collector.
``text`` format serves the first, ``json`` the second, and both are the same
call site.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_file: str | Path | None = None,
) -> None:
    """Install the process-wide logging configuration. Idempotent."""
    global _CONFIGURED

    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Rotate so an edge node with a small SSD cannot fill its disk with logs.
        handlers.append(
            logging.handlers.RotatingFileHandler(
                path, maxBytes=32 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
        )

    logging.basicConfig(format="%(message)s", level=log_level, handlers=handlers, force=True)

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # These libraries are chatty at INFO and drown out operational signal.
    for noisy in ("urllib3", "asyncio", "aiosqlite", "multipart", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str, **initial: Any) -> Any:
    """Return a bound structlog logger, configuring defaults on first use."""
    if not _CONFIGURED:
        configure_logging()
    logger = structlog.get_logger(name)
    return logger.bind(**initial) if initial else logger


def bind_context(**kwargs: Any) -> None:
    """Bind values onto every log line emitted by the current task/thread."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
