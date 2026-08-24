# IBVAP architecture

## The problem this solves

Border forces already have CCTV at out posts, check posts and along border
roads. What they lack is anything that *watches* it: conventional systems
record and display, leaving a human to notice things. Adding FRS or ANPR
normally means buying dedicated appliances or smart cameras — per camera, per
site, on hardware that has to be carried to places reachable only by road.

IBVAP is the software that turns those existing cameras into an intelligent
sensor network. It ingests standard RTSP, runs the analytics itself, and raises
alerts. No camera is replaced and no appliance is added.

## Shape of the system

```
   IP cameras (RTSP)
          │
          ▼
   ┌──────────────┐   decode at source rate, enqueue at analytics rate
   │  StreamReader│   bounded queue, drop-oldest, reconnect w/ backoff
   └──────┬───────┘
          ▼
   ┌──────────────┐   detector (strided) → tracker (every frame)
   │ CameraWorker │   → ANPR / face (once per track) → rules
   └──────┬───────┘
          ▼
   ┌──────────────┐   dedup · cooldown · rate limit
   │AnalyticsEngine│
   └──────┬───────┘
          ▼
   ┌──────────────┐        ┌───────────────┐
   │EvidenceStore │◄───────┤   Event       ├────────► WebSocket (consoles)
   │ snapshot+clip│        └───────┬───────┘
   │ hash chain   │                ▼
   └──────────────┘        ┌───────────────┐
                           │  Dispatcher   │  store-and-forward outbox
                           └───────┬───────┘
                                   ▼
                     webhook · MQTT · syslog/CEF  → command and control
```

One `CameraWorker` per camera, isolated from the others. A `Supervisor` owns
them all and is the single object the API, CLI and desktop console talk to.

## The decisions that shape everything else

### Drop frames, never latency

The frame queue is small (default 4) and overflows **oldest-first**. A deep
queue would preserve every frame at the cost of alerting on an intrusion a
minute after it happened — which is not an alert, it is a historical record.
Live security value decays in seconds, so when analytics falls behind, old
frames are discarded and latency stays flat.

Frames are still *decoded* at source rate, because an unread FFmpeg buffer
grows until the frames it returns are already seconds stale. Decode
everything; enqueue a sample.

### Degrade loudly, never silently

Missing model artefacts do not stop a node. The detector falls back to
classical background subtraction with shape heuristics, face detection falls
back to a Haar cascade where one is available, and plate localisation falls
back to morphology. These are materially worse — and the platform says so, in
`/health`, in the operator console, and in the `ibvap_model_info` metric.

A control room must never be left believing it has face recognition when the
node has no embedder.

### Normalised coordinates everywhere

Zones and tripwires are stored in `[0, 1]` space, never pixels. Cameras get
re-profiled routinely (1080p main stream by day, 720p sub-stream when the link
degrades); pixel geometry would silently shift under every operator-drawn fence.

### Suppression is a first-class feature

An unfiltered rule set across 32 cameras produces thousands of events an hour,
and a control room that receives thousands of alerts an hour stops reading them
within a shift. Three independent mechanisms narrow the flow — deduplication,
per-rule cooldown, and a hard per-camera rate limit — and every suppressed
event is *counted* so over-aggressive tuning shows up on a dashboard instead of
being mistaken for a quiet night.

The single highest-value tuning knob is `analytics.ignore_classes: [animal]`.
Stray cattle on rural fence lines are the dominant false-alarm source in this
domain.

### Evidence must be defensible

Every alert with a bounding box gets an annotated snapshot, optionally a clip
assembled from a rolling pre-event buffer, a SHA-256 per artefact, and a
manifest chained to the previous one for that day. Alteration *and deletion*
are both detectable.

This is tamper-**evident**, not tamper-proof: anyone with write access to the
directory could rebuild the chain. Defeating that needs an append-only store or
external notarisation, which is a deployment decision.

## Module map

| Package | Responsibility |
|---|---|
| `core/` | Domain types, layered config, geometry, logging, time windows |
| `ingest/` | Video sources, resilient reader, bounded queue |
| `vision/` | ONNX backends, detector, tracker, ANPR, face |
| `analytics/` | Rules and the per-camera engine with its suppression gate |
| `events/` | Annotation, evidence store, canonical payload |
| `pipeline/` | Model bundle, camera worker, supervisor |
| `storage/` | ORM, async engine, repositories |
| `integrations/` | Webhook, MQTT, syslog sinks; store-and-forward dispatcher |
| `api/` | FastAPI app, auth/RBAC, routers |
| `ui/` | Browser operator console (no build step, no CDN) |
| `desktop/` | PyQt6 native console |
| `mlops/` | Registry, export, benchmark, evaluation, drift |
| `telemetry/` | Prometheus instrumentation |

## Data flow for one alert

1. `StreamReader` decodes a frame and enqueues it if the sampling interval has
   elapsed.
2. `CameraWorker` runs the detector (every `detect_interval` frames) and the
   tracker (every frame; it coasts on Kalman prediction in between).
3. Confirmed tracks reach `AnalyticsEngine`, which runs each configured rule.
4. Rules emit events; the gate decides which survive.
5. `AppState` captures evidence while the frame is still in memory.
6. `EventDispatcher` persists the event, queues it per sink, and delivers.
7. Consoles receive it over WebSocket; C2 receives it over its configured sink.

Median end-to-end latency on a 4-core CPU node is single-digit milliseconds for
the analytics stage; the dominant cost is detection.

## Tiers

**Edge** (`tier: edge`) runs beside the cameras at a BOP. SQLite, no external
dependencies, survives losing its uplink indefinitely via the outbox.

**Central** (`tier: central`) aggregates at sector level. PostgreSQL, multiple
replicas, no cameras of its own.

The edge tier is deliberately *not* deployed on Kubernetes: a BOP node gains
nothing from an orchestrator that sits on the far side of the link that just
failed.
