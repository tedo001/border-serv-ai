# IBVAP architecture

## The problem

Border forces already have CCTV at out posts, check posts and along border
roads. What they lack is anything that *watches* it: conventional systems
record and display, leaving a human to notice things. Adding FRS or ANPR
normally means buying dedicated appliances or smart cameras — per camera, per
site, on hardware that has to be carried to places reachable only by road.

IBVAP is the software that turns those existing cameras into an intelligent
sensor network. It ingests standard RTSP, runs the analytics itself, and raises
alerts. No camera is replaced and no appliance is added.

---

## The pipeline

One `CameraWorker` owns one camera end to end. Frames flow left to right; each
stage narrows the data and adds meaning.

```
 ┌─────────────┐
 │   SOURCE    │  RTSP · HTTP · file · sim://  (scenario simulator)
 └──────┬──────┘
        │  decode at source rate
        ▼
 ┌─────────────┐   Decode everything, enqueue a sample. An unread decoder
 │  INGEST     │   buffer grows until its frames are seconds stale, so the
 │ StreamReader│   reader pulls at full rate and samples into a small
 └──────┬──────┘   bounded queue that overflows OLDEST-FIRST.
        │  8 fps typical, 4-frame queue
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  DETECTION            where is something, and roughly what   │
 │                                                              │
 │   RT-DETR ─┐   set predictor, NMS-free, normalised cxcywh    │
 │   YOLO26 ──┤   end-to-end head, NMS folded into the graph    │
 │   YOLOv8 ──┤   transposed head, platform runs NMS            │
 │   YOLOv5 ──┤   objectness × class score                      │
 │   MOG2 ────┘   classical fallback when no artefact exists    │
 └──────┬───────────────────────────────────────────────────────┘
        │  boxes + coarse class + score
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  CLASSIFICATION       what is it really                      │
 │  MobileNetV3 / ImageNet-1k over each new track's crop        │
 │                                                              │
 │  Runs ONCE PER TRACK, not per frame. Demotes and reclassifies │
 │  only — never promotes to PERSON, because ImageNet-1k has no │
 │  person class. Its value is the ~400 animal classes.         │
 └──────┬───────────────────────────────────────────────────────┘
        │  corrected class
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  TRACKING             which ones are the same object         │
 │  ByteTrack-style: Kalman prediction + Hungarian association  │
 │  Three passes — high-score IoU, low-score IoU recovery,      │
 │  distance rescue for fast movers.                            │
 └──────┬───────────────────────────────────────────────────────┘
        │  stable track ids, trails, velocity
        ├──────────────────────────┬──────────────────────────┐
        ▼                          ▼                          ▼
 ┌─────────────┐          ┌─────────────┐          ┌─────────────────┐
 │ ANPR        │          │ FACE        │          │  ANALYTICS      │
 │ locate →    │          │ detect →    │          │  10 rules over  │
 │ deskew →    │          │ align →     │          │  zones, wires,  │
 │ CRNN+CTC →  │          │ embed →     │          │  dwell, speed,  │
 │ IND grammar │          │ gallery     │          │  schedule       │
 └──────┬──────┘          └──────┬──────┘          └────────┬────────┘
        └────────────────────────┴──────────────────────────┘
                                 │  Events
                                 ▼
                    ┌────────────────────────┐
                    │  EVENT GATE            │
                    │  dedup · cooldown ·    │
                    │  per-camera rate limit │
                    └────────┬───────────────┘
                             ▼
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
 ┌──────────────┐                        ┌─────────────────┐
 │  EVIDENCE    │                        │   DISPATCHER    │
 │  snapshot +  │                        │  store-and-     │
 │  clip, SHA-  │                        │  forward outbox │
 │  256, daily  │                        └────────┬────────┘
 │  hash chain  │                                 │
 └──────────────┘                                 ▼
                                       webhook (HMAC signed)  →  C2
                                                  │
                              WebSocket ──────────┴──→ browser + desktop consoles
```

---

## Model roles

Every model is declared in `models/registry.yaml` with a version, a SHA-256
checksum, its output layout and its class list. Nothing is inferred from a
filename, and a checksum mismatch is fatal.

| Role | Default | Job | Absent → |
|---|---|---|---|
| `detector` | YOLO26-S | Locate people, vehicles, animals, bags | MOG2 background subtraction |
| `classifier` | MobileNetV3 (ImageNet-1k) | Refine the coarse class per track | Detector class used as-is |
| `plate_detector` | single-class YOLO | Find plates inside vehicle crops | Morphological plate search |
| `plate_ocr` | CRNN + CTC | Read plate glyphs | ANPR disabled |
| `face_detector` | single-class YOLO | Find faces inside person crops | Haar cascade if available |
| `face_embedder` | ArcFace-style 512-d | Embed for watchlist matching | Face recognition disabled |

### Two runtimes, one pipeline

Detection runs on one of two runtimes, and which one is a property of the
registry entry rather than of the code:

| Runtime | What it is | Where it belongs |
|---|---|---|
| `ultralytics` | A published Torch checkpoint - RT-DETR or YOLO - run directly | Development, evaluation, tuning |
| `onnx` | A verified, checksummed ONNX artefact on ONNX Runtime | A border post |

The point of keeping both is that the first produces the second:

```
ibvap models fetch rtdetr-l
  download rtdetr-l.pt      ->  models/weights/
  export ONNX               ->  models/detector/rtdetr-l-coco.onnx
  hash and register         ->  models/registry.yaml
```

A post then binds the ONNX entry and never installs Torch. Both carry the same
weights, so a threshold tuned in the analyst console means the same thing in
the field - and that agreement is checked rather than assumed. On the same
photograph the two runtimes return the same five objects, with scores within
0.006 and boxes within two pixels.

This matters more than it sounds. Before the Torch runtime existed there was
no way to run a real RT-DETR graph through this repository, and the RT-DETR
ONNX decoder had never been executed against one. It was wrong: it assumed the
reference head's `(queries, 4 + nc)` per-class scores, while Ultralytics
exports `(300, 6)` of `[cx, cy, w, h, confidence, class_id]`. Reading the
second as the first put a class *index* of up to 79 through a sigmoid, and 300
phantom detections came back at score 0.51 - `sigmoid(0)` - every one labelled
class 0. Both forms are now decoded and both are pinned by tests.

### Detector families

| Layout | Output | Notes |
|---|---|---|
| `rtdetr` | `(queries, 4+nc)` | Set predictor. Boxes **normalised** cxcywh — scaled to input pixels before the letterbox inverse. No NMS. |
| `yolo26` | `(N, 6)` xyxy | End-to-end head; NMS already applied inside the graph, so the platform skips it. |
| `yolov8` | `(4+nc, anchors)` | v8/v9/v10/v11. Class scores, no objectness. Platform runs class-aware NMS. |
| `yolov5` | `(anchors, 5+nc)` | Objectness × class score. |
| `nms_xyxy` | `(N, 6)` xyxy | Any export with NMS folded in. |

Two families need care and both fail *silently* if mishandled, which is why the
layout is declared rather than guessed:

- **RT-DETR boxes are normalised to `[0,1]`.** Decoded as YOLO pixels they
  cluster in the top-left corner — plausible in a list, nonsense on screen.
- **End-to-end heads have already suppressed duplicates.** Running NMS again
  can only delete true positives, such as two people standing shoulder to
  shoulder.

### Why MobileNet earns a stage

The detector gives a coarse class; MobileNet checks it. The case that pays for
it is livestock.

Stray cattle on a rural fence line are the largest single source of false
intrusion alarms in this domain. `analytics.ignore_classes: [animal]` exists to
suppress them — but suppression only works if the object is *classified* as an
animal. The classical fallback classifies by bounding-box aspect ratio alone,
so a cow (wide) reads as a car and is never suppressed.

Measured on the `cattle` simulation scenario:

| Configuration | Result |
|---|---|
| Fallback detector, `ignore_classes: [animal]` | **False intrusion alarm** — cow read as `car` |
| Fallback + MobileNet classifier stage | **No alarm** — 27 tracks reclassified to `animal` |
| Same settings, `intrusion` scenario (a person) | **Alarms correctly** |

ImageNet-1k is a good fit here specifically because roughly 400 of its 1000
classes are animals, in detail — ox, water buffalo, bison, ram, ibex.

It is a poor fit for one thing, and the code enforces it: **ImageNet-1k has no
person class**, so the stage may demote or reclassify but never promote to
`PERSON`.

---

## The decisions that shape everything else

**Drop frames, never latency.** The queue is small and overflows oldest-first.
A deep queue preserves every frame at the cost of alerting a minute late —
which is not an alert, it is a historical record.

**Degrade loudly, never silently.** Missing artefacts do not stop a node; it
falls back and *says so* in `/health`, in both consoles, and in the
`ibvap_model_info` metric. A control room must never believe it has face
recognition when the node has no embedder.

**Normalised coordinates everywhere.** Zones and tripwires are stored in
`[0,1]`, never pixels. Cameras get re-profiled routinely (1080p by day, 720p
sub-stream when the link degrades); pixel geometry would silently shift under
every operator-drawn fence.

**Suppression is a feature.** An unfiltered rule set across 32 cameras produces
thousands of events an hour, and a control room receiving thousands of alerts
an hour stops reading them within a shift. Deduplication, per-rule cooldown and
a hard per-camera rate limit narrow the flow, and every suppressed event is
*counted* so over-tuning shows on a dashboard rather than looking like a quiet
night.

**Evidence must be defensible.** SHA-256 per artefact plus a per-day hash chain
across manifests, so alteration *and deletion* are both detectable. This is
tamper-evident, not tamper-proof — anyone with write access could rebuild the
chain; defeating that needs an append-only store, which is a deployment choice.

---

## Simulation

`sim://<scenario>` renders scripted border footage in-process: terrain with a
fence line and approach road, actors that walk, drive and graze, day/night/IR
palettes, haze and sensor noise, with perspective scaling so distant objects
are smaller.

| Scenario | Exercises |
|---|---|
| `patrol` | Routine activity — should raise nothing |
| `intrusion` | Zone entry and fence crossing |
| `cattle` | Livestock suppression, with a person for contrast |
| `vehicle` | Approach-road traffic, ANPR staging |
| `loiter` | Dwell-time logic |
| `abandoned` | Unattended-object logic |
| `crowd` | Density and clustering |
| `night` | IR palette, night-movement rules |

It exists because the real footage — people crossing fences at night, cattle in
restricted zones — is operationally sensitive, rarely shareable, and never
available when a specific case is needed. Scenarios are deterministic by seed,
so a test asserting "this raises an intrusion alert" gives the same answer on
every machine.

**It is not a substitute for real footage when judging accuracy.** These are
geometric figures on synthetic terrain; a detector's score here says nothing
about its score on a real IR frame at 40 m.

---

## Consoles

Three surfaces read the same platform, and the split is by *job*, not by
technology.

| Console | Job | Needs a node? |
|---|---|---|
| Browser (`ui/`) | Duty operator: video wall, alert triage, evidence review | yes |
| Desktop (`desktop/main_window.py`) | Supervisor: the above plus the zone editor | yes |
| Live analysis (`desktop/analyst.py`) | Analyst: one source, tuned by hand | no |

The live analysis console is the odd one out and deliberately so. It builds a
`CameraWorker` itself — same detector, same tracker, same rules, same event
gate — and subscribes to its `frame_sink` and `event_sink`. It does not talk
to the API, does not authenticate and does not write evidence, because it is
not a second node: it is a bench for deciding what the node should be told to
do.

That choice has two consequences worth stating:

* **What the analyst tunes is what the node runs.** A confidence threshold, a
  fence line or a livestock setting that behaves one way in this window
  behaves the same way on the post, because there is only one implementation
  underneath. A console with its own inference path would drift from the node
  within a release or two, and the first argument about a missed alert would
  be unwinnable.
* **Frames are coalesced, not queued.** The worker produces faster than a GUI
  paints. The bridge keeps only the newest unpainted frame and counts the
  rest as dropped, so the window can fall behind in *detail* but never in
  *time* — the same trade-off the ingest queue makes for the same reason.

Its risk slider filters the alert *view* rather than the pipeline: every event
stays in the JSON export, so raising the threshold to quieten the screen can
never quietly discard the record that something happened.

---

## Module map

| Package | Responsibility |
|---|---|
| `core/` | Domain types, layered config, geometry, logging, time windows |
| `ingest/` | Video sources, scenario simulator, resilient reader, bounded queue |
| `vision/` | ONNX and Torch runtimes, detectors, classifier, tracker, ANPR, face, supervision interchange |
| `analytics/` | Rules and the per-camera engine with its suppression gate |
| `events/` | Annotation, evidence store, canonical payload |
| `pipeline/` | Model bundle, camera worker, supervisor |
| `storage/` | ORM, async engine, repositories |
| `integrations/` | Signed webhook sink and the store-and-forward dispatcher |
| `api/` | FastAPI app, auth/RBAC, routers |
| `ui/` | Browser operator console (no build step, no CDN) |
| `desktop/` | PyQt6 control panel, operator console (zone editor) and live analysis console |
| `mlops/` | Registry and authoring, benchmark, evaluation, drift |
| `telemetry/` | Prometheus instrumentation |

---

## Tiers

**Edge** (`tier: edge`) runs beside the cameras at a BOP. SQLite, no external
dependencies, survives losing its uplink indefinitely via the outbox.

**Central** (`tier: central`) aggregates at sector level. PostgreSQL, multiple
replicas, no cameras of its own.

The edge tier is deliberately *not* deployed on an orchestrator: a BOP node
gains nothing from a control plane sitting on the far side of the link that
just failed.
