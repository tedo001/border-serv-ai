"""Evidence capture with chain of custody.

An alert that cannot be substantiated later is of limited use to a border
force: incidents are reviewed days afterwards, and the reviewer needs to see
what the camera saw and be able to establish that it has not been altered
since.

Three mechanisms provide that:

* **Snapshot and clip.** An annotated still plus, where configured, a short
  clip assembled from a rolling pre-event buffer - so the seconds *before* the
  trigger are preserved, which is usually where the useful context is.
* **Per-artefact hashing.** SHA-256 over the stored bytes, recorded in a
  manifest written alongside the artefact.
* **A hash chain across manifests.** Each manifest carries the digest of the
  previous one for that day. Altering or deleting an entry breaks the chain
  from that point on, so removal is detectable, not merely alteration.

The chain is deliberately simple - it is tamper-*evident*, not tamper-proof,
and makes no cryptographic claim beyond that. Anyone with write access to the
directory could rebuild it; defeating that requires an append-only store or
external notarisation, which is a deployment decision rather than a platform
one, and is documented as such.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ibvap.core.config import EvidenceConfig, PrivacyConfig
from ibvap.core.errors import StorageError
from ibvap.core.geometry import Tripwire, Zone
from ibvap.core.logging import get_logger
from ibvap.core.timeutils import to_iso
from ibvap.core.types import Event, Frame, Track
from ibvap.events.annotate import annotate_frame

log = get_logger(__name__)

#: Written into every manifest so a reader knows how to interpret it.
MANIFEST_VERSION = 1


@dataclass(slots=True)
class EvidenceRecord:
    """What was stored for one event."""

    event_id: str
    snapshot_path: str | None = None
    clip_path: str | None = None
    manifest_path: str | None = None
    snapshot_sha256: str | None = None
    clip_sha256: str | None = None
    #: Digest of this manifest, forming the next link in the chain.
    chain_hash: str | None = None
    bytes_written: int = 0


class FrameBuffer:
    """Rolling pre-event frame buffer for one camera.

    Sized in seconds rather than frames so the memory cost is predictable
    regardless of the camera's analytics rate. At 8 fps, 4 seconds of 960x540
    BGR is roughly 50 MB per camera - which is exactly why the default is
    modest and the cap is enforced rather than advisory.
    """

    def __init__(self, seconds: float = 4.0, fps: float = 8.0, max_frames: int = 240) -> None:
        capacity = max(1, min(max_frames, int(seconds * max(1.0, fps)) + 1))
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, frame: Frame) -> None:
        with self._lock:
            self._frames.append((frame.monotonic, frame.image))

    def snapshot(self) -> list[tuple[float, np.ndarray]]:
        with self._lock:
            return list(self._frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


class EvidenceStore:
    """Writes, hashes and prunes evidence artefacts."""

    def __init__(
        self,
        config: EvidenceConfig | None = None,
        privacy: PrivacyConfig | None = None,
        *,
        site_id: str = "",
    ) -> None:
        self.config = config or EvidenceConfig()
        self.privacy = privacy or PrivacyConfig()
        self.site_id = site_id
        self.root = Path(self.config.directory)
        self._lock = threading.Lock()
        #: Per-day tip of the hash chain: ``date -> digest``.
        self._chain_tips: dict[str, str] = {}
        self._buffers: dict[str, FrameBuffer] = {}

        if self.config.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # -- pre-event buffering ---------------------------------------------- #

    def buffer_for(self, camera_id: str, fps: float = 8.0) -> FrameBuffer:
        buffer = self._buffers.get(camera_id)
        if buffer is None:
            buffer = FrameBuffer(self.config.clip_pre_seconds, fps)
            self._buffers[camera_id] = buffer
        return buffer

    def observe(self, frame: Frame) -> None:
        """Feed a frame into that camera's pre-event buffer."""
        if not self.config.enabled or not self.config.clip:
            return
        self.buffer_for(frame.camera_id, frame.fps or 8.0).add(frame)

    # -- capture ----------------------------------------------------------- #

    def capture(
        self,
        event: Event,
        frame: Frame,
        *,
        tracks: list[Track] | None = None,
        zones: list[Zone] | None = None,
        tripwires: list[Tripwire] | None = None,
        camera_name: str = "",
        face_boxes_to_blur: list[Any] | None = None,
    ) -> EvidenceRecord:
        """Store evidence for ``event`` and return what was written."""
        record = EvidenceRecord(event_id=event.event_id)
        if not self.config.enabled:
            return record

        try:
            directory = self._directory_for(event)
            directory.mkdir(parents=True, exist_ok=True)

            if self.config.snapshot:
                self._write_snapshot(
                    event, frame, record, directory,
                    tracks=tracks, zones=zones, tripwires=tripwires,
                    camera_name=camera_name, face_boxes_to_blur=face_boxes_to_blur,
                )
            if self.config.clip:
                self._write_clip(event, frame, record, directory)

            self._write_manifest(event, record, directory)

            event.snapshot_path = record.snapshot_path
            event.clip_path = record.clip_path
            event.evidence_hash = record.snapshot_sha256
        except OSError as exc:
            # Evidence failure must never lose the alert itself: an operator
            # being told about an intrusion without a picture beats not being
            # told at all.
            log.error(
                "evidence_write_failed",
                event=event.event_id, camera=event.camera_id, error=str(exc),
            )
        return record

    def _directory_for(self, event: Event) -> Path:
        day = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        return self.root / day / event.camera_id

    def _write_snapshot(
        self,
        event: Event,
        frame: Frame,
        record: EvidenceRecord,
        directory: Path,
        **kwargs: Any,
    ) -> None:
        face_boxes = kwargs.pop("face_boxes_to_blur", None)
        image = frame.image

        # Privacy: blur uninvolved faces before annotation, so the redaction is
        # baked into the stored bytes rather than applied at view time - the
        # artefact itself must be safe to hand to a wider audience.
        if self.privacy.blur_unmatched_faces and face_boxes:
            from ibvap.vision.face import blur_face

            image = image.copy()
            for box in face_boxes:
                image = blur_face(image, box)

        annotated = annotate_frame(
            image,
            tracks=kwargs.get("tracks"),
            zones=kwargs.get("zones"),
            tripwires=kwargs.get("tripwires"),
            event=event,
            camera_name=kwargs.get("camera_name") or event.camera_id,
            timestamp=event.timestamp,
        )

        path = directory / f"{event.event_id}.jpg"
        ok, encoded = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.snapshot_quality]
        )
        if not ok:
            raise StorageError(f"failed to encode snapshot for event {event.event_id}")

        data = encoded.tobytes()
        _atomic_write(path, data)
        record.snapshot_path = str(path)
        record.snapshot_sha256 = hashlib.sha256(data).hexdigest()
        record.bytes_written += len(data)

    def _write_clip(
        self, event: Event, frame: Frame, record: EvidenceRecord, directory: Path
    ) -> None:
        """Assemble a clip from the pre-event buffer plus the trigger frame."""
        buffer = self._buffers.get(event.camera_id)
        frames = buffer.snapshot() if buffer else []
        if not frames:
            return

        images = [img for _, img in frames]
        images.append(frame.image)
        height, width = images[0].shape[:2]
        # Guard against a mid-stream resolution change: the writer silently
        # produces a corrupt file if frame sizes vary.
        images = [
            img if img.shape[:2] == (height, width) else cv2.resize(img, (width, height))
            for img in images
        ]

        path = directory / f"{event.event_id}.mp4"
        fps = max(1.0, frame.fps or 8.0)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            log.warning("clip_writer_unavailable", event=event.event_id)
            return
        try:
            for img in images:
                writer.write(img)
        finally:
            writer.release()

        if path.is_file():
            record.clip_path = str(path)
            record.clip_sha256 = _sha256_path(path)
            record.bytes_written += path.stat().st_size

    def _write_manifest(self, event: Event, record: EvidenceRecord, directory: Path) -> None:
        """Write the manifest and extend that day's hash chain."""
        day = directory.parent.name
        with self._lock:
            previous = self._chain_tips.get(day) or self._load_chain_tip(directory.parent)

            manifest: dict[str, Any] = {
                "manifest_version": MANIFEST_VERSION,
                "event_id": event.event_id,
                "site_id": self.site_id,
                "camera_id": event.camera_id,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "message": event.message,
                "rule_id": event.rule_id,
                "zone_id": event.zone_id,
                "track_ids": event.track_ids,
                "confidence": event.confidence,
                "attributes": _json_safe(event.attributes),
                "event_time": to_iso(event.timestamp),
                "frame_index": event.frame_index,
                "captured_at": to_iso(time.time()),
                "artefacts": {
                    "snapshot": Path(record.snapshot_path).name if record.snapshot_path else None,
                    "snapshot_sha256": record.snapshot_sha256,
                    "clip": Path(record.clip_path).name if record.clip_path else None,
                    "clip_sha256": record.clip_sha256,
                },
                "previous_manifest_sha256": previous,
            }

            payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(payload).hexdigest()
            manifest["manifest_sha256"] = digest

            path = directory / f"{event.event_id}.json"
            _atomic_write(path, json.dumps(manifest, indent=2, sort_keys=True).encode())

            self._chain_tips[day] = digest
            _atomic_write(directory.parent / ".chain", digest.encode())

            record.manifest_path = str(path)
            record.chain_hash = digest
            record.bytes_written += path.stat().st_size

    @staticmethod
    def _load_chain_tip(day_directory: Path) -> str:
        marker = day_directory / ".chain"
        if marker.is_file():
            try:
                return marker.read_text(encoding="utf-8").strip()
            except OSError:
                return ""
        return ""

    # -- verification ------------------------------------------------------ #

    def verify(self, manifest_path: str | Path) -> dict[str, Any]:
        """Verify one manifest against its artefacts and its own digest."""
        path = Path(manifest_path)
        result: dict[str, Any] = {
            "manifest": str(path), "valid": False, "issues": [],
        }
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result["issues"].append(f"unreadable manifest: {exc}")
            return result

        declared = manifest.pop("manifest_sha256", "")
        recomputed = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if declared != recomputed:
            result["issues"].append("manifest digest mismatch: metadata was altered")

        artefacts = manifest.get("artefacts") or {}
        for kind in ("snapshot", "clip"):
            name = artefacts.get(kind)
            expected = artefacts.get(f"{kind}_sha256")
            if not name or not expected:
                continue
            artefact_path = path.parent / name
            if not artefact_path.is_file():
                result["issues"].append(f"{kind} missing: {name}")
                continue
            if _sha256_path(artefact_path) != expected:
                result["issues"].append(f"{kind} digest mismatch: {name} was altered")

        result["valid"] = not result["issues"]
        result["event_id"] = manifest.get("event_id")
        return result

    def verify_chain(self, day: str) -> dict[str, Any]:
        """Verify the hash chain for one day across all cameras.

        Detects deletion as well as alteration: a removed manifest leaves a
        successor whose recorded predecessor digest no longer resolves.
        """
        day_directory = self.root / day
        manifests: list[Path] = sorted(day_directory.glob("*/*.json"))
        report: dict[str, Any] = {
            "day": day, "manifests": len(manifests), "valid": True, "issues": [],
        }
        if not manifests:
            return report

        by_digest: dict[str, dict[str, Any]] = {}
        entries: list[dict[str, Any]] = []
        for manifest_path in manifests:
            check = self.verify(manifest_path)
            if not check["valid"]:
                report["issues"].extend(f"{manifest_path.name}: {i}" for i in check["issues"])
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            entries.append(data)
            if digest := data.get("manifest_sha256"):
                by_digest[digest] = data

        # Every declared predecessor must exist, except the chain's genesis.
        for entry in entries:
            previous = entry.get("previous_manifest_sha256")
            if previous and previous not in by_digest:
                report["issues"].append(
                    f"{entry.get('event_id')}: predecessor {previous[:12]}... not found; "
                    "an earlier manifest was deleted or altered"
                )

        report["valid"] = not report["issues"]
        return report

    # -- retention --------------------------------------------------------- #

    def prune(self) -> dict[str, int]:
        """Enforce the retention policy. Returns what was removed."""
        removed_files = 0
        removed_bytes = 0
        if not self.config.enabled or not self.root.exists():
            return {"files": 0, "bytes": 0}

        now = time.time()
        entries: list[tuple[float, Path, int]] = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.name == ".chain":
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, path, stat.st_size))

        # 1. Age-based deletion.
        if self.config.retention_days > 0:
            cutoff = now - self.config.retention_days * 86400
            for mtime, path, size in list(entries):
                if mtime < cutoff and _safe_unlink(path):
                    removed_files += 1
                    removed_bytes += size
                    entries.remove((mtime, path, size))

        # 2. Size-based deletion, oldest first. A disk that fills stops the node
        #    writing *anything*, so a hard cap matters more than keeping the
        #    full retention window on a small edge SSD.
        if self.config.max_bytes > 0:
            total = sum(size for _, _, size in entries)
            if total > self.config.max_bytes:
                # sorted() orders by mtime first, so this walks oldest-first.
                for _mtime, path, size in sorted(entries):
                    if total <= self.config.max_bytes:
                        break
                    if _safe_unlink(path):
                        removed_files += 1
                        removed_bytes += size
                        total -= size

        _remove_empty_directories(self.root)
        if removed_files:
            log.info(
                "evidence_pruned",
                files=removed_files, megabytes=round(removed_bytes / 1024**2, 1),
            )
        return {"files": removed_files, "bytes": removed_bytes}

    def usage(self) -> dict[str, Any]:
        """Current evidence footprint, for health reporting."""
        if not self.root.exists():
            return {"files": 0, "bytes": 0, "max_bytes": self.config.max_bytes}
        total = 0
        count = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                    count += 1
                except OSError:
                    continue
        return {
            "files": count,
            "bytes": total,
            "megabytes": round(total / 1024**2, 1),
            "max_bytes": self.config.max_bytes,
            "utilisation": round(total / self.config.max_bytes, 4) if self.config.max_bytes else 0.0,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via a temporary file and rename.

    A node losing power mid-write - routine at a BOP on generator supply -
    must not leave a half-written manifest that fails verification forever.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except OSError:
        return False


def _remove_empty_directories(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                next(path.iterdir())
            except StopIteration:
                path.rmdir()
            except OSError:
                continue


def _json_safe(value: Any) -> Any:
    """Coerce a value into something ``json.dumps`` accepts."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
