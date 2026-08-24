# Deployment

## Choosing a tier

| | Edge (BOP node) | Central (sector) |
|---|---|---|
| Runs | Beside the cameras | In a control room / data centre |
| Cameras | 4–16 typically | None |
| Database | SQLite (no server) | PostgreSQL |
| Deploy with | Docker Compose or systemd | Kubernetes |
| Survives uplink loss | Yes — outbox replays | N/A |

The edge tier is deliberately **not** deployed on Kubernetes. A BOP node gains
nothing from an orchestrator sitting on the far side of the link that just
failed.

## Sizing a node

Benchmark on the actual hardware before committing cameras to it:

```bash
ibvap benchmark --resolution 1920x1080
```

The report gives a camera count at 8 fps with 30% headroom, sized on **p95**
latency rather than the median. Exceeding it does not break the node — frames
are dropped rather than queued, so alert latency stays flat — but coverage falls.

Rules of thumb:

- Detection dominates cost; tracking is ~1/15th of it.
- Raising `detect_interval` to 2 nearly halves load with little accuracy cost
  at 8 fps.
- `process_width: 960` is a good default; 1280 costs ~1.8x for marginal gain on
  perimeter work.

## Docker (recommended for a BOP)

```bash
cp configs/site.example.yaml configs/site.yaml   # then edit
ibvap validate configs/site.yaml

ibvap secret > deploy/docker/.env                # JWT signing key
docker compose -f deploy/docker/compose.yaml up -d
docker compose -f deploy/docker/compose.yaml logs -f node
```

With local metrics retention:

```bash
docker compose -f deploy/docker/compose.yaml --profile monitoring up -d
```

## Bare metal (systemd)

See [`deploy/systemd/README.md`](../deploy/systemd/README.md).

## Secrets

Secrets never go in `site.yaml`. That file is copied between posts on removable
media and gets attached to tickets. Inject through the environment:

| Variable | Purpose |
|---|---|
| `IBVAP_SECURITY__JWT_SECRET` | Token signing key (min 32 bytes; `ibvap secret`) |
| `IBVAP_SECURITY__BOOTSTRAP_ADMIN_PASSWORD` | First admin password; omit to auto-generate |
| `IBVAP_INTEGRATIONS__MQTT__PASSWORD` | MQTT broker credential |

Any setting can be overridden this way — `IBVAP_` prefix, `__` for nesting.

## Installing models

A node runs classical fallback detection until artefacts are installed, and
reports itself `degraded` throughout.

```bash
pip install "ibvap[export]"

ibvap models export yolo26s.pt \
    -o models/detector/yolo26s-border-1.0.0.onnx \
    --family yolo26 --imgsz 640 \
    --register yolo26s-border:1.0.0 \
    --card docs/model-cards/yolo26s-border.md

ibvap models verify        # confirm checksums before restarting
```

Copy `models/` and `registry.yaml` together to each node. A checksum mismatch
is **fatal** by design — an artefact truncated by a failed sync still loads and
still returns tensors, they are simply wrong.

## Commissioning checklist

1. `ibvap probe <rtsp-url>` for every camera — confirms frames actually arrive.
2. Draw zones and tripwires in the desktop console's Zone editor.
3. `ibvap validate configs/site.yaml`.
4. `ibvap models verify`.
5. Start the node; change the bootstrap admin password at first login.
6. Watch for a shift with `analytics.ignore_classes: [animal]` set, then tune
   thresholds against what actually fired.
7. Capture drift reference profiles once the camera is performing acceptably.
8. Confirm alerts reach the C2 system end to end.

## Upgrading

```bash
docker compose -f deploy/docker/compose.yaml pull
docker compose -f deploy/docker/compose.yaml up -d
```

The database schema is created on start. Evidence and events survive restarts.
Roll back by pinning the previous image tag.
