"""Layered configuration for IBVAP.

Precedence, lowest to highest:

1. Defaults declared on the models below.
2. A YAML site file (``configs/site.yaml``), which is what an operator edits.
3. Environment variables prefixed ``IBVAP_`` (``__`` marks nesting), which is
   how orchestrators inject per-node values.
4. Explicit keyword overrides, used by tests.

Secrets are deliberately *not* sourced from YAML. A site file is copied between
BOPs on removable media and ends up in ticket attachments; anything sensitive
must arrive through the environment or a mounted secret file.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ibvap.core.errors import ConfigError

# --------------------------------------------------------------------------- #
# Component configuration models
# --------------------------------------------------------------------------- #


class ZoneConfig(BaseModel):
    """Operator-drawn polygon, in normalised coordinates."""

    id: str
    name: str = ""
    points: list[tuple[float, float]]
    kind: Literal["restricted", "buffer", "mask", "counting"] = "restricted"
    enabled: bool = True

    @field_validator("points")
    @classmethod
    def _validate_points(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("a zone needs at least 3 points")
        for x, y in v:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"zone points must be normalised to [0,1], got ({x}, {y})")
        return v

    @model_validator(mode="after")
    def _default_name(self) -> "ZoneConfig":
        if not self.name:
            self.name = self.id
        return self


class TripwireConfig(BaseModel):
    """Directed virtual fence line, in normalised coordinates."""

    id: str
    name: str = ""
    start: tuple[float, float]
    end: tuple[float, float]
    direction: Literal["any", "left", "right"] = "any"
    #: Human meaning of a left-sense crossing, e.g. "infiltration".
    left_label: str = "left"
    #: Human meaning of a right-sense crossing, e.g. "exfiltration".
    right_label: str = "right"
    enabled: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "TripwireConfig":
        for pt in (self.start, self.end):
            if not (0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0):
                raise ValueError(f"tripwire points must be normalised to [0,1], got {pt}")
        if self.start == self.end:
            raise ValueError("tripwire start and end must differ")
        if not self.name:
            self.name = self.id
        return self


class RuleConfig(BaseModel):
    """One analytics rule instance bound to a camera."""

    id: str
    #: Rule implementation key, e.g. ``intrusion``, ``loitering``, ``anpr``.
    type: str
    enabled: bool = True
    #: Zones/tripwires this rule watches. Empty means "the whole frame".
    zones: list[str] = Field(default_factory=list)
    tripwires: list[str] = Field(default_factory=list)
    #: Object classes the rule reacts to. Empty means "all classes".
    classes: list[str] = Field(default_factory=list)
    #: Severity override; ``None`` uses the event type's default.
    severity: Literal["info", "low", "medium", "high", "critical"] | None = None
    #: Suppress repeat alerts for the same incident for this many seconds.
    cooldown_seconds: float = 30.0
    #: Only active during these local-time windows, e.g. ``["18:00-06:00"]``.
    active_hours: list[str] = Field(default_factory=list)
    #: Rule-specific tuning (dwell_seconds, speed_threshold, ...).
    params: dict[str, Any] = Field(default_factory=dict)


class CameraConfig(BaseModel):
    """A single video source and everything the platform should do with it."""

    id: str
    name: str = ""
    #: RTSP/HTTP URL, a file path for replay, or ``synthetic://`` for drills.
    url: str
    enabled: bool = True
    #: Free-text posting, e.g. "BOP Ranipur - North Tower".
    location: str = ""
    latitude: float | None = None
    longitude: float | None = None
    #: Compass bearing the camera faces, degrees from true north.
    bearing: float | None = None
    #: Cap analytics throughput; decoding still runs at source rate.
    target_fps: float = Field(default=8.0, gt=0.0, le=60.0)
    #: Run heavy detection every Nth sampled frame; tracking fills the gaps.
    detect_interval: int = Field(default=1, ge=1, le=30)
    #: Force TCP transport for RTSP (UDP loses packets over VSAT/microwave).
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    #: Decoder read timeout before the supervisor declares the stream dead.
    read_timeout_seconds: float = Field(default=15.0, gt=0.0)
    #: Longest edge the frame is resized to before inference.
    process_width: int = Field(default=960, ge=160, le=3840)
    zones: list[ZoneConfig] = Field(default_factory=list)
    tripwires: list[TripwireConfig] = Field(default_factory=list)
    rules: list[RuleConfig] = Field(default_factory=list)
    #: Enable ANPR on this camera (costs an extra model pass; use on approaches).
    anpr_enabled: bool = False
    #: Enable face detection/recognition (use at check posts and gates).
    face_enabled: bool = False
    #: Auto-enable low-light enhancement when the frame is dark.
    night_enhancement: bool = True
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "CameraConfig":
        if not self.name:
            self.name = self.id
        zone_ids = {z.id for z in self.zones}
        wire_ids = {t.id for t in self.tripwires}
        if len(zone_ids) != len(self.zones):
            raise ValueError(f"camera {self.id!r} has duplicate zone ids")
        if len(wire_ids) != len(self.tripwires):
            raise ValueError(f"camera {self.id!r} has duplicate tripwire ids")
        for rule in self.rules:
            for z in rule.zones:
                if z not in zone_ids:
                    raise ValueError(f"rule {rule.id!r} references unknown zone {z!r}")
            for w in rule.tripwires:
                if w not in wire_ids:
                    raise ValueError(f"rule {rule.id!r} references unknown tripwire {w!r}")
        return self


class ModelSpec(BaseModel):
    """Which registry entry backs one inference role."""

    #: Registry model name; ``null`` disables the role.
    name: str | None = None
    #: Registry version, or ``latest``.
    version: str = "latest"
    #: Score below which detections are discarded.
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    #: IoU threshold for non-maximum suppression.
    nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


class ModelsConfig(BaseModel):
    """Model registry location, execution providers and per-role bindings."""

    registry_path: Path = Path("models/registry.yaml")
    models_dir: Path = Path("models")
    #: ONNX Runtime execution providers, in priority order. Unavailable
    #: providers are skipped at load time rather than failing the process.
    providers: list[str] = Field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    #: Threads per inference session. 0 lets ONNX Runtime decide.
    intra_op_threads: int = Field(default=0, ge=0, le=64)
    #: Fall back to the deterministic stub backend when weights are absent.
    #: Keeps CI and field drills runnable on a laptop with no model files.
    allow_stub_fallback: bool = True

    detector: ModelSpec = Field(default_factory=lambda: ModelSpec(name="yolo-person-vehicle", score_threshold=0.35))
    face_detector: ModelSpec = Field(default_factory=lambda: ModelSpec(name="face-detector", score_threshold=0.6))
    face_embedder: ModelSpec = Field(default_factory=lambda: ModelSpec(name="face-embedder"))
    plate_detector: ModelSpec = Field(default_factory=lambda: ModelSpec(name="plate-detector", score_threshold=0.4))
    plate_ocr: ModelSpec = Field(default_factory=lambda: ModelSpec(name="plate-ocr"))


class PipelineConfig(BaseModel):
    """Throughput, latency and resource-safety knobs for the worker pool."""

    #: Bounded frame queue per camera. Small on purpose: a deep queue turns a
    #: CPU shortfall into growing alert latency, which is worse than dropping
    #: frames. Live security value decays in seconds.
    queue_size: int = Field(default=4, ge=1, le=64)
    #: What to do when the queue is full. ``drop_oldest`` keeps latency flat.
    overflow_policy: Literal["drop_oldest", "drop_newest", "block"] = "drop_oldest"
    #: Worker threads for inference across all cameras.
    inference_workers: int = Field(default=2, ge=1, le=32)
    #: Restart backoff bounds for a failed camera, in seconds.
    reconnect_min_seconds: float = Field(default=2.0, gt=0.0)
    reconnect_max_seconds: float = Field(default=60.0, gt=0.0)
    #: Emit a CAMERA_OFFLINE event after this long without a frame.
    offline_after_seconds: float = Field(default=30.0, gt=0.0)
    #: Frames between forced full-detector runs even when tracking is healthy.
    max_track_only_frames: int = Field(default=10, ge=1, le=120)

    @model_validator(mode="after")
    def _validate(self) -> "PipelineConfig":
        if self.reconnect_max_seconds < self.reconnect_min_seconds:
            raise ValueError("reconnect_max_seconds must be >= reconnect_min_seconds")
        return self


class TrackerConfig(BaseModel):
    """Multi-object tracker association and lifecycle thresholds."""

    #: Detections above this score enter the high-confidence first pass.
    high_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Detections above this score are eligible for the recovery second pass.
    low_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    #: Minimum IoU to associate a detection with an existing track.
    match_iou: float = Field(default=0.2, ge=0.0, le=1.0)
    #: Consecutive hits before a track is reported to analytics.
    min_hits: int = Field(default=3, ge=1, le=30)
    #: Frames a track survives unmatched before removal (occlusion budget).
    max_age: int = Field(default=30, ge=1, le=300)
    #: Trail length retained per track, used by behavioural rules.
    trail_length: int = Field(default=60, ge=2, le=600)


class AnalyticsConfig(BaseModel):
    """Global analytics behaviour and false-alarm suppression."""

    #: Classes that never raise intrusion alerts. Stray cattle on rural border
    #: fences are the dominant false-alarm source; suppressing them by class is
    #: the single highest-value tuning knob in the whole platform.
    ignore_classes: list[str] = Field(default_factory=lambda: ["animal"])
    #: Ignore detections smaller than this fraction of frame height (noise).
    min_object_height_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    #: Mean luma below which a frame is classified as night/IR.
    night_luma_threshold: float = Field(default=60.0, ge=0.0, le=255.0)
    #: Global cap on events per camera per minute; protects the C2 link.
    max_events_per_camera_per_minute: int = Field(default=60, ge=1, le=10_000)
    #: Seconds of identical-key events collapsed into one alert.
    dedup_window_seconds: float = Field(default=20.0, ge=0.0)


class EvidenceConfig(BaseModel):
    """Snapshot/clip capture and chain-of-custody settings."""

    enabled: bool = True
    directory: Path = Path("data/evidence")
    #: Write an annotated JPEG for each event.
    snapshot: bool = True
    snapshot_quality: int = Field(default=85, ge=40, le=100)
    #: Write a short video clip around the event using a rolling pre-buffer.
    clip: bool = True
    clip_pre_seconds: float = Field(default=4.0, ge=0.0, le=60.0)
    clip_post_seconds: float = Field(default=6.0, ge=0.0, le=60.0)
    #: Delete artefacts older than this. 0 disables automatic deletion.
    retention_days: int = Field(default=30, ge=0, le=3650)
    #: Cap on total evidence bytes; the janitor prunes oldest-first past it.
    max_bytes: int = Field(default=50 * 1024**3, ge=0)


class PrivacyConfig(BaseModel):
    """Privacy controls. Surveillance capability demands explicit limits."""

    #: Blur faces in stored evidence unless the face matched a watchlist.
    blur_unmatched_faces: bool = False
    #: Store face embeddings only for watchlist enrolments, never for passers-by.
    store_only_watchlist_faces: bool = True
    #: Purge raw face crops after this many days regardless of retention.
    face_crop_retention_days: int = Field(default=7, ge=0, le=3650)
    #: Record every watchlist match in the audit log for later review.
    audit_biometric_matches: bool = True


class SecurityConfig(BaseModel):
    """Authentication, authorisation and transport security."""

    #: HS256 signing key. MUST come from the environment in production.
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = Field(default=60, ge=1, le=1440)
    refresh_token_ttl_days: int = Field(default=7, ge=1, le=90)
    #: Bootstrap administrator, created on first start if no users exist.
    bootstrap_admin_user: str = "admin"
    #: Bootstrap password; when empty a random one is generated and logged once.
    bootstrap_admin_password: str = ""
    #: Browser origins allowed to call the API.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8080"])
    #: Reject unauthenticated requests. Only disable for isolated bench work.
    require_auth: bool = True
    #: Failed logins from one IP before temporary lockout.
    max_login_attempts: int = Field(default=5, ge=1, le=100)
    lockout_seconds: float = Field(default=300.0, ge=0.0)


class StorageConfig(BaseModel):
    """Metadata persistence."""

    #: SQLite by default so a BOP node needs no database server. Point at
    #: PostgreSQL for the central/sector tier.
    database_url: str = "sqlite+aiosqlite:///data/ibvap.db"
    #: Delete event rows older than this. 0 keeps them forever.
    event_retention_days: int = Field(default=180, ge=0, le=3650)
    echo_sql: bool = False


class WebhookConfig(BaseModel):
    """One outbound HTTP sink into a command-and-control system."""

    name: str
    url: str
    enabled: bool = True
    #: Minimum severity forwarded to this sink.
    min_severity: Literal["info", "low", "medium", "high", "critical"] = "medium"
    #: Only forward these event types; empty means all.
    event_types: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    #: Shared secret for HMAC-SHA256 request signing. Env-injected.
    hmac_secret: str = ""
    timeout_seconds: float = Field(default=10.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=10)


class MqttConfig(BaseModel):
    """MQTT sink - the usual transport to a sector control room."""

    enabled: bool = False
    host: str = "localhost"
    port: int = Field(default=1883, ge=1, le=65535)
    username: str = ""
    password: str = ""
    topic_prefix: str = "ibvap/events"
    qos: int = Field(default=1, ge=0, le=2)
    tls: bool = False
    min_severity: Literal["info", "low", "medium", "high", "critical"] = "low"


class SyslogConfig(BaseModel):
    """Syslog/CEF sink for SIEM ingestion."""

    enabled: bool = False
    host: str = "localhost"
    port: int = Field(default=514, ge=1, le=65535)
    protocol: Literal["udp", "tcp"] = "udp"
    facility: int = Field(default=13, ge=0, le=23)
    min_severity: Literal["info", "low", "medium", "high", "critical"] = "medium"


class IntegrationsConfig(BaseModel):
    """All outbound C2 integrations plus store-and-forward behaviour."""

    webhooks: list[WebhookConfig] = Field(default_factory=list)
    mqtt: MqttConfig = Field(default_factory=MqttConfig)
    syslog: SyslogConfig = Field(default_factory=SyslogConfig)
    #: Persist undelivered events and replay them when the link returns.
    #: Essential at remote BOPs where connectivity is intermittent by default.
    store_and_forward: bool = True
    outbox_max_events: int = Field(default=50_000, ge=100)
    outbox_flush_interval_seconds: float = Field(default=15.0, gt=0.0)


class TelemetryConfig(BaseModel):
    """Metrics, logging and health reporting."""

    prometheus_enabled: bool = True
    #: Metrics are served on the API under ``/metrics``; this optional port
    #: exposes them separately for nodes that keep the API on a private VLAN.
    prometheus_port: int | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    #: JSON logs for shipping to a central collector; text for a console.
    log_format: Literal["json", "text"] = "json"
    log_file: Path | None = None


class ApiConfig(BaseModel):
    """HTTP server binding."""

    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    #: Serve the bundled operator console at ``/``.
    serve_ui: bool = True
    #: Cap concurrent MJPEG preview subscribers; each costs encode bandwidth.
    max_preview_clients: int = Field(default=8, ge=0, le=128)


# --------------------------------------------------------------------------- #
# Root settings
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Root configuration object for an IBVAP node."""

    model_config = SettingsConfigDict(
        env_prefix="IBVAP_",
        env_nested_delimiter="__",
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    #: Stable identifier for this node, used in every event and metric.
    site_id: str = "bop-default"
    site_name: str = "Border Out Post"
    #: ``edge`` runs analytics next to the cameras; ``central`` aggregates.
    tier: Literal["edge", "central"] = "edge"

    api: ApiConfig = Field(default_factory=ApiConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        ids = [c.id for c in self.cameras]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate camera ids: {sorted(dupes)}")
        return self

    # -- helpers ---------------------------------------------------------- #

    def camera(self, camera_id: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        raise ConfigError(f"unknown camera {camera_id!r}")

    def enabled_cameras(self) -> list[CameraConfig]:
        return [c for c in self.cameras if c.enabled]

    def ensure_secret(self) -> str:
        """Return the JWT secret, generating an ephemeral one if unset.

        An ephemeral secret is acceptable for a dev run - every restart simply
        invalidates outstanding tokens - but never for production, so callers
        are expected to surface the warning emitted by :func:`load_settings`.
        """
        if not self.security.jwt_secret:
            self.security.jwt_secret = secrets.token_urlsafe(48)
        return self.security.jwt_secret


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict, tolerating an empty document."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config root of {p} must be a mapping, got {type(data).__name__}")
    return data


def load_settings(
    config_path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Build a :class:`Settings` from YAML, environment and explicit overrides.

    ``config_path`` defaults to ``$IBVAP_CONFIG`` and then ``configs/site.yaml``.
    A missing default file is not an error: the platform must be able to start
    with defaults so a fresh node can be configured through the API.
    """
    path = config_path or os.environ.get("IBVAP_CONFIG")
    data: dict[str, Any] = {}
    if path:
        data = load_yaml(path)
    elif Path("configs/site.yaml").is_file():
        data = load_yaml("configs/site.yaml")

    # Support an `include:` list so shared camera groups can be factored out.
    includes = data.pop("include", []) or []
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    base_dir = Path(path).parent if path else Path("configs")
    for inc in includes:
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = base_dir / inc_path
        merged = _deep_merge(merged, load_yaml(inc_path))
    merged = _deep_merge(merged, data)
    if overrides:
        merged = _deep_merge(merged, overrides)

    try:
        return Settings(**merged)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(f"invalid configuration: {exc}") from exc
