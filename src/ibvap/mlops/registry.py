"""Versioned model registry.

Every model the platform runs is declared in ``models/registry.yaml`` with an
explicit version, a SHA-256 checksum, its output layout and its class list.
Nothing is inferred from a filename.

This is not ceremony. A surveillance platform whose output can put a person in
front of an armed response must be able to answer, months later, *exactly which
artefact produced this alert*. Three properties follow:

* **Integrity.** A checksum mismatch is fatal, never a warning. An artefact
  that was truncated by a failed sync over a VSAT link will still load in ONNX
  Runtime and still return tensors - they are simply wrong.
* **Pinning.** Sites run a named version, not "whatever is in the directory".
  A sector node and a BOP node must be able to prove they agree.
* **Declared layout.** The decoder is told whether a graph is YOLOv5, YOLOv8 or
  an end-to-end YOLO26 head. Guessing from tensor shape works until the day it
  does not, and the failure is silent: garbled boxes, not an exception.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ibvap.core.config import ModelsConfig, ModelSpec
from ibvap.core.errors import (
    ChecksumMismatchError,
    ConfigError,
    ModelError,
    ModelNotFoundError,
)
from ibvap.core.logging import get_logger
from ibvap.vision.backends import InferenceBackend, OnnxBackend

log = get_logger(__name__)

#: Roles the platform binds models to.
MODEL_ROLES: tuple[str, ...] = (
    "detector",
    "face_detector",
    "face_embedder",
    "plate_detector",
    "plate_ocr",
    "classifier",
)


@dataclass(slots=True)
class ModelVersion:
    """One immutable artefact of one model."""

    version: str
    #: Path to the artefact, relative to the registry's models directory.
    file: str
    #: Lowercase hex SHA-256. Empty disables verification (development only).
    sha256: str = ""
    #: Output layout for detector-family models, e.g. ``yolo26``.
    layout: str = "auto"
    #: Model input as ``(width, height)``; ``None`` reads it from the graph.
    input_size: tuple[int, int] | None = None
    #: Ordered class names the model emits.
    classes: list[str] = field(default_factory=list)
    #: Published accuracy figures, for the model card and drift comparison.
    metrics: dict[str, float] = field(default_factory=dict)
    #: Provenance: training dataset, licence, export command, author.
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    #: Set false to keep an artefact declared but block it from loading.
    enabled: bool = True


@dataclass(slots=True)
class ModelEntry:
    """A named model and all of its versions."""

    name: str
    role: str
    versions: dict[str, ModelVersion] = field(default_factory=dict)
    default_version: str = ""
    description: str = ""

    def resolve(self, version: str = "latest") -> ModelVersion:
        """Return a specific version, or the default when asked for ``latest``."""
        if version in ("latest", "", "default"):
            if self.default_version and self.default_version in self.versions:
                return self.versions[self.default_version]
            if not self.versions:
                raise ModelNotFoundError(f"model {self.name!r} declares no versions")
            # Highest version string wins when no default is declared. Sorting
            # is lexicographic on dotted integers, which is correct for the
            # semver-ish scheme used throughout.
            newest = max(self.versions, key=_version_key)
            return self.versions[newest]

        entry = self.versions.get(version)
        if entry is None:
            raise ModelNotFoundError(
                f"model {self.name!r} has no version {version!r}; "
                f"available: {sorted(self.versions)}"
            )
        return entry


def _version_key(version: str) -> tuple[int, ...]:
    """Sort key for dotted numeric versions; non-numeric parts sort as 0."""
    parts: list[int] = []
    for chunk in version.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256. Chunked so a 200 MB model is not resident."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistry:
    """Loads registry metadata and constructs verified inference backends."""

    def __init__(self, config: ModelsConfig | None = None) -> None:
        self.config = config or ModelsConfig()
        self.entries: dict[str, ModelEntry] = {}
        self._backends: dict[str, InferenceBackend] = {}
        self._verified: set[str] = set()
        self.loaded_from: Path | None = None

    # -- loading ----------------------------------------------------------- #

    @classmethod
    def load(cls, config: ModelsConfig | None = None) -> ModelRegistry:
        registry = cls(config)
        registry.reload()
        return registry

    def reload(self) -> None:
        """Re-read the registry file. A missing file yields an empty registry.

        Absence is not an error: a fresh node legitimately has no artefacts yet
        and must still start, fall back to classical detection and report that
        state through ``/health``.
        """
        path = Path(self.config.registry_path)
        self.entries.clear()
        if not path.is_file():
            log.warning("model_registry_missing", path=str(path))
            return

        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid model registry {path}: {exc}") from exc
        if not isinstance(document, dict):
            raise ConfigError(f"model registry {path} must be a mapping")

        for name, raw in (document.get("models") or {}).items():
            if not isinstance(raw, dict):
                raise ConfigError(f"model {name!r} must be a mapping")
            role = raw.get("role", "")
            if role and role not in MODEL_ROLES:
                raise ConfigError(
                    f"model {name!r} has unknown role {role!r}; expected one of {list(MODEL_ROLES)}"
                )

            versions: dict[str, ModelVersion] = {}
            for version, spec in (raw.get("versions") or {}).items():
                if not isinstance(spec, dict):
                    raise ConfigError(f"model {name!r} version {version!r} must be a mapping")
                size = spec.get("input_size")
                versions[str(version)] = ModelVersion(
                    version=str(version),
                    file=str(spec.get("file", "")),
                    sha256=str(spec.get("sha256", "")).lower(),
                    layout=str(spec.get("layout", "auto")),
                    input_size=(int(size[0]), int(size[1])) if size else None,
                    classes=list(spec.get("classes") or []),
                    metrics={k: float(v) for k, v in (spec.get("metrics") or {}).items()},
                    provenance=dict(spec.get("provenance") or {}),
                    notes=str(spec.get("notes", "")),
                    enabled=bool(spec.get("enabled", True)),
                )

            self.entries[name] = ModelEntry(
                name=name,
                role=role,
                versions=versions,
                default_version=str(raw.get("default", "")),
                description=str(raw.get("description", "")),
            )

        self.loaded_from = path
        log.info("model_registry_loaded", path=str(path), models=len(self.entries))

    # -- resolution -------------------------------------------------------- #

    def get(self, name: str) -> ModelEntry:
        entry = self.entries.get(name)
        if entry is None:
            raise ModelNotFoundError(
                f"model {name!r} is not in the registry; available: {sorted(self.entries)}"
            )
        return entry

    def artefact_path(self, version: ModelVersion) -> Path:
        path = Path(version.file)
        return path if path.is_absolute() else Path(self.config.models_dir) / path

    def verify(self, name: str, version: str = "latest") -> ModelVersion:
        """Resolve a version and verify its artefact on disk.

        Verification runs once per artefact per process: hashing a 200 MB file
        on every camera start would add seconds to bring-up for no benefit,
        since the file cannot change underneath a running node without a
        restart.
        """
        entry = self.get(name)
        resolved = entry.resolve(version)
        if not resolved.enabled:
            raise ModelNotFoundError(f"model {name}:{resolved.version} is disabled in the registry")

        path = self.artefact_path(resolved)
        if not path.is_file():
            raise ModelNotFoundError(f"artefact for {name}:{resolved.version} not found at {path}")

        cache_key = f"{name}:{resolved.version}"
        if cache_key in self._verified:
            return resolved

        if resolved.sha256:
            actual = sha256_file(path)
            if actual != resolved.sha256:
                # Fatal by design: never run inference on an artefact whose
                # provenance cannot be established.
                raise ChecksumMismatchError(
                    f"checksum mismatch for {name}:{resolved.version} at {path}: "
                    f"expected {resolved.sha256}, got {actual}"
                )
        else:
            log.warning(
                "model_checksum_absent",
                model=name, version=resolved.version,
                detail="artefact integrity is unverified; declare sha256 before deployment",
            )

        self._verified.add(cache_key)
        return resolved

    # -- backends ---------------------------------------------------------- #

    def load_backend(self, spec: ModelSpec) -> tuple[InferenceBackend, ModelVersion] | None:
        """Build a verified backend for a role binding.

        Returns ``None`` when the model is unconfigured or its artefact is
        absent and stub fallback is permitted - the caller then runs in
        degraded mode. A checksum failure always raises: a corrupt artefact is
        a different situation from a missing one and must not be papered over.
        """
        if not spec.name:
            return None

        try:
            version = self.verify(spec.name, spec.version)
        except ChecksumMismatchError:
            raise
        except (ModelNotFoundError, ConfigError) as exc:
            if self.config.allow_stub_fallback:
                log.warning("model_unavailable_degrading", model=spec.name, error=str(exc))
                return None
            raise

        cache_key = f"{spec.name}:{version.version}"
        cached = self._backends.get(cache_key)
        if cached is not None:
            return cached, version

        path = self.artefact_path(version)
        backend = OnnxBackend(
            path,
            providers=self.config.providers,
            intra_op_threads=self.config.intra_op_threads,
        )
        backend.warmup()
        self._backends[cache_key] = backend
        log.info(
            "model_backend_ready",
            model=spec.name, version=version.version, layout=version.layout,
            path=str(path), providers=backend.providers,
        )
        return backend, version

    def close(self) -> None:
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()

    # -- introspection ----------------------------------------------------- #

    def describe(self) -> list[dict[str, Any]]:
        """Registry contents for the API and operator console."""
        out: list[dict[str, Any]] = []
        for entry in self.entries.values():
            for version in entry.versions.values():
                path = self.artefact_path(version)
                out.append({
                    "name": entry.name,
                    "role": entry.role,
                    "version": version.version,
                    "layout": version.layout,
                    "classes": len(version.classes),
                    "metrics": version.metrics,
                    "enabled": version.enabled,
                    "present": path.is_file(),
                    "verified": f"{entry.name}:{version.version}" in self._verified,
                    "checksum_declared": bool(version.sha256),
                    "is_default": version.version == entry.default_version,
                    "notes": version.notes,
                })
        return out


# ------------------------------------------------------------------------- #
# Registry authoring
# ------------------------------------------------------------------------- #



def register_model(
    registry_path: str | Path,
    name: str,
    version: str,
    metadata: dict[str, Any],
    *,
    role: str = "detector",
    make_default: bool = False,
    description: str = "",
    metrics: dict[str, float] | None = None,
) -> None:
    """Add or update an entry in ``models/registry.yaml``.

    Refuses to overwrite an existing version with different content. An
    artefact that has been deployed must never change under a version another
    site is pinning; publish a new version instead.
    """
    path = Path(registry_path)
    document: dict[str, Any] = {"version": 1, "models": {}}
    if path.is_file():
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or document
    document.setdefault("models", {})

    entry = document["models"].setdefault(
        name, {"role": role, "description": description, "versions": {}}
    )
    if description:
        entry["description"] = description
    entry["role"] = role

    existing = entry["versions"].get(version)
    if existing and existing.get("sha256") != metadata.get("sha256"):
        raise ModelError(
            f"{name}:{version} is already registered with a different checksum. "
            "Publish a new version rather than mutating a deployed one."
        )

    models_dir = path.parent
    artefact = Path(metadata["file"])
    try:
        relative = artefact.resolve().relative_to(models_dir.resolve())
        file_field = str(relative)
    except ValueError:
        file_field = str(artefact)

    entry["versions"][version] = {
        "file": file_field,
        "sha256": metadata["sha256"],
        "layout": metadata.get("layout", "auto"),
        "input_size": metadata.get("input_size"),
        "classes": metadata.get("classes", []),
        "metrics": metrics or metadata.get("metrics", {}),
        "provenance": metadata.get("provenance", {}),
        "notes": metadata.get("notes", ""),
        "enabled": True,
    }
    if make_default or not entry.get("default"):
        entry["default"] = version

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False, width=100), encoding="utf-8")
    log.info(
        "model_registered",
        name=name, version=version, role=role,
        layout=metadata.get("layout"), default=entry.get("default"),
    )


def write_model_card(
    output: str | Path, name: str, version: str, metadata: dict[str, Any],
    *, metrics: dict[str, float] | None = None, intended_use: str = "",
    limitations: str = "",
) -> Path:
    """Write a model card documenting provenance, performance and limits.

    A model card is not paperwork here. This platform can put a person in front
    of an armed response; whoever authorises a deployment needs to know what
    the model was trained on, how it was measured, and where it is known to
    fail - particularly at night, at range, and on classes it confuses.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = metadata.get("provenance", {})
    metrics = metrics or metadata.get("metrics", {})

    lines = [
        f"# Model card — {name}:{version}", "",
        "## Identity", "",
        f"- **Name**: `{name}`", f"- **Version**: `{version}`",
        f"- **Artefact**: `{Path(metadata.get('file', '')).name}`",
        f"- **SHA-256**: `{metadata.get('sha256', '')}`",
        f"- **Output layout**: `{metadata.get('layout', 'auto')}`",
        f"- **Input size**: {metadata.get('input_size')}",
        f"- **Size**: {metadata.get('size_mb', '?')} MB",
        f"- **Classes** ({len(metadata.get('classes', []))}): "
        f"{', '.join(metadata.get('classes', [])[:20])}"
        + (" …" if len(metadata.get("classes", [])) > 20 else ""),
        "", "## Provenance", "",
    ]
    lines += [f"- **{k.replace('_', ' ')}**: {v}" for k, v in provenance.items()] or ["- not recorded"]

    lines += ["", "## Measured performance", ""]
    if metrics:
        lines += ["| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in metrics.items()]
    else:
        lines.append(
            "**Not yet evaluated.** Run `ibvap evaluate` against a labelled set "
            "from a representative site before deploying this version."
        )

    lines += [
        "", "## Intended use", "",
        intended_use or (
            "Detection of people and vehicles in fixed CCTV views at border out "
            "posts, check posts and border roads, as input to the IBVAP analytics "
            "rules. Not validated for any other purpose."
        ),
        "", "## Known limitations", "",
        limitations or (
            "- Accuracy degrades at long range; objects below roughly 20 px in "
            "height are unreliable.\n"
            "- Night-time and IR performance depends on illumination and is "
            "materially worse than daytime unless the model was trained with IR "
            "imagery.\n"
            "- Livestock is a common source of false person detections on rural "
            "fence lines; the `animal` class exists so it can be suppressed by "
            "rule rather than by threshold.\n"
            "- Heavy occlusion, adverse weather (fog, heavy rain) and camera "
            "tampering all reduce recall.\n"
            "- The model reflects the distribution of its training data; "
            "performance at a given site should be re-measured after deployment."
        ),
        "", "## Operational guidance", "",
        "- Verify the checksum above matches the deployed artefact before use.",
        "- Re-evaluate after any quantisation; INT8 conversion changes accuracy.",
        "- Monitor `ibvap_detections_total` and the drift report for a shift in "
        "the score distribution, which usually precedes a measurable accuracy loss.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("model_card_written", path=str(path))
    return path
