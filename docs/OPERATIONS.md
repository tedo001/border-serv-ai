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
