# Operations

## Daily checks

```bash
curl -s http://localhost:8080/health | jq '{status, cameras_online, cameras_total}'
```

| Status | Meaning |
|---|---|
| `healthy` | All cameras delivering, neural models loaded |
| `degraded` | All cameras delivering, but running fallback detection |
| `partial` | Some cameras offline |
| `down` | No cameras delivering |
| `idle` | No cameras configured |

## Tuning false alarms

Work in this order — the first item resolves most of them:

1. **Livestock.** `analytics.ignore_classes: [animal]`. The dominant false-alarm
   source on rural fence lines.
2. **Vegetation and shadows.** Raise `params.confirm_frames` on the rule (3–5),
   and `analytics.min_object_height_fraction` to 0.03.
3. **A zone that is too large.** Restrict the polygon to the ground that
   matters. Zone membership is tested at the object's *feet*, not its centroid.
4. **Repeat alerts on one subject.** Raise the rule's `cooldown_seconds`.
5. **Alert storms.** `analytics.max_events_per_camera_per_minute` is a hard
   ceiling that protects the C2 link regardless of what the rules decide.

Check what is being suppressed rather than guessing:

```
ibvap_events_suppressed_total   # by camera and reason
```

If suppression is high and operators report missing alerts, the tuning has gone
too far.

## Missing detections

1. Is the node `degraded`? Fallback detection is materially weaker — install
   model artefacts.
2. Are frames being dropped? `ibvap_frames_dropped_total{reason="queue_full"}`
   means the node is oversubscribed. Lower `target_fps`, raise
   `detect_interval`, or move cameras to another node.
3. Is the object too small? Below ~20 px tall, detection is unreliable at any
   setting. This is a camera placement problem, not a tuning one.
4. Are tracks fragmenting? Raise `tracker.max_age` for scenes with heavy
   occlusion.

## Key metrics

| Metric | Watch for |
|---|---|
| `ibvap_camera_up` | 0 = offline |
| `ibvap_camera_fps` | Well below `target_fps` = node overloaded |
| `ibvap_frames_dropped_total` | Sustained `queue_full` = oversubscribed |
| `ibvap_pipeline_latency_seconds` | p95 above ~0.5 s = struggling |
| `ibvap_outbox_depth` | Growing = C2 link down |
| `ibvap_events_suppressed_total` | High = over-tuned |
| `ibvap_model_info{backend="motion_fallback"}` | Running degraded |

Alerting rules ship in [`deploy/docker/alerts.yml`](../deploy/docker/alerts.yml).

## Drift

Field accuracy cannot be measured without ground truth, so the platform
compares live behaviour against a reference captured at commissioning — once an
operator has confirmed the camera is performing acceptably.

A drift alert is a prompt to **look**, never an automatic action. Common causes:

| Signal | Usual cause |
|---|---|
| Detection rate collapsed | Lens obstructed, camera re-aimed, illuminator failed |
| Score distribution shifted | Degraded image quality, changed view |
| Object size changed | Camera moved or zoomed — **its zones are now wrong** |
| Class mix changed | Scene traffic pattern changed, or the model has |

## Evidence

```bash
ibvap evidence verify-chain 2026-08-24   # chain of custody for one day
ibvap evidence prune                     # apply retention now
```

Retention runs hourly on its own. Both age (`retention_days`) and total size
(`max_bytes`) are enforced — the size cap matters more on a small edge SSD,
because a full disk stops the node recording anything at all.

## Camera tamper

`camera_tamper` alerts mean the stream is still up but the camera has been
blinded — covered lens, defocus, or repositioning. A plain "is the stream up?"
health check reports green throughout, which is exactly why this rule exists.

Treat it as a physical security event, not a fault.

## Backup

Back up, in order of importance:

1. `configs/site.yaml` — zones, tripwires and rules represent real survey work.
2. The database — events, users, watchlists, audit trail.
3. `models/` plus `registry.yaml` — must travel together.
4. Evidence — largest, and usually subject to a retention policy anyway.

---

## Detector runtimes

A node picks its detector from the registry entry bound in `models.detector`,
and the entry's `runtime` decides how it loads. Three are tried in order, most
trustworthy first, and every fall is logged:

| Order | Runtime | Reported as | Needs |
|---|---|---|---|
| 1 | `ultralytics` | `ultralytics:rtdetr` / `ultralytics:yolo` | the `torch` extra |
| 2 | `onnx` | `neural` | a verified artefact in `models/` |
| 3 | — | `motion_fallback` | nothing |

`/health`, both consoles, the control panel and the `ibvap_model_info` metric
all carry the mode, and `degraded` is true for anything below a real model. A
node that quietly ran the weakest option would report healthy while missing
people, so it never does.

### Putting real weights on a node

```bash
# development and evaluation - Torch, on the machine doing the tuning
pip install -e '.[torch]'
#   models.detector.name: rtdetr-l

# the post - ONNX Runtime alone, no Torch anywhere on the node
ibvap models fetch rtdetr-l
#   models.detector.name: rtdetr-l-onnx
```

`fetch` downloads the checkpoint, exports the ONNX graph, hashes it and writes
the registry entry. The checksum it records is of *your* export, so run it on
the build host and ship `models/` with the site build; a mismatch at load time
is fatal by design.

### Air-gapped posts

Nothing here reaches the network at run time except the first checkpoint
download. To prepare a post that cannot reach one, copy the file into
`models/weights/` (Torch) or `models/detector/` (ONNX) before first start; the
registry entry names exactly what it expects.

---

## ANPR: commissioning a plate reader

ANPR is a chain, and each link degrades separately. `/health` reports each one
under `models`, so check there first when reads are poor.

| Link | Model | Absent means |
|---|---|---|
| Vehicle detection | `detector` | no vehicles, so no ANPR at all |
| Plate localisation | `plate_detector` | morphological search: clean approaches only |
| Recognition | `plate_ocr` | plates are located but never read |

### 1. Train the plate detector

There is no published licence-plate model worth shipping: no COCO class covers
it, and plate shape, colour and mounting differ by country and often by site.
Collect vehicle crops from the posts this will serve, label the plate box, and
fine-tune YOLOv8 single-class. A few thousand crops that include **night, rain
and oblique angles** are worth more than tens of thousands of clean daylight
frontals — localisation degrades far faster with angle than vehicle detection
does, and the failure is silent: no box, no read, no alert, no log line saying
a plate was there.

Put the weights at `models/weights/yolov8-plate.pt` and set `enabled: true` on
the `plate-yolov8` entry in `models/registry.yaml`.

### 2. Register it for the field

```bash
# ONNX: ships in the site build, runs on ONNX Runtime alone
ibvap models fetch models/weights/yolov8-plate.pt \
    --role plate_detector --name plate-yolov8 --format onnx

# TensorRT: built on the node, at commissioning
ibvap models fetch models/weights/yolov8-plate.pt \
    --role plate_detector --name plate-yolov8 --format engine --half
```

Then bind it:

```yaml
models:
  plate_detector: { name: plate-yolov8-onnx, score_threshold: 0.4 }
```

### 3. Tune the vote, not the threshold

A plate is decided by agreement across frames, not by the best single read.
Two knobs control it:

```yaml
analytics:
  plate_min_reads: 3        # frames that must agree
  plate_vote_margin: 1.5    # how far the leader must lead the runner-up
```

Raise `plate_min_reads` where vehicles are slow and well lit and you want fewer
wrong reads; lower it at a fast approach where a vehicle is only in shot for a
few frames and you would rather have a read than none. Raising the margin is
the right response to *confusions* — two candidates one character apart — which
is a different problem from *no reads*, and lowering the OCR threshold will not
fix it.

Every `plate_read` event carries `reads`, `total_reads`, `vote_margin` and
`runner_up`. If reads are being disputed, those four numbers say whether the
platform was confident or lucky.

### Reading the failure reasons

The `ibvap_plate_reads_total` metric is labelled by outcome:

| Label | Meaning |
|---|---|
| `accepted` | read and grammatical |
| `rejected_format` | read but not a valid plate; it votes, but never alone decides |
| `rejected_position` | the plate box jumped off the plate; the read was refused |
| `no_plate_located` | localisation found nothing — usually angle or light |
| `no_text_read` | a plate was located but OCR returned nothing |
| `low_confidence` | text came back below the OCR threshold |
| `no_read_predicted` | the Kalman-predicted box also yielded nothing |
| `ocr_unavailable` | no OCR artefact on this node |
| `empty_crop` | the vehicle box fell outside the frame |

A high `rejected_position` rate means the plate detector is firing on
headlights or reflective strips: retrain with hard negatives rather than
raising its threshold, which will cost real plates too.
