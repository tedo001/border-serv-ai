# IBVAP — Intelligent Border Video Analytics Platform

Software-defined AI surveillance that turns **existing IP CCTV infrastructure**
into an intelligent sensor network — no dedicated FRS or ANPR appliances, no
smart-camera hardware, no per-camera licence.

Built for Problem Statement **26187** — Ministry of Home Affairs, Sashastra
Seema Bal (SSB), Police II Division.

---

## What it does

| Capability | Implementation |
|---|---|
| Human detection and tracking | RT-DETR / YOLO26 / YOLOv8 / v5 detector + ByteTrack-style tracker with Kalman filtering |
| Object classification | MobileNetV3 (ImageNet-1k) secondary stage — reclassifies coarse detections, notably livestock |
| Vehicle detection and classification | Same detector, taxonomy-mapped to car / truck / bus / motorcycle / bicycle |
| Face detection and recognition | Detector → ArcFace-style embedding → watchlist gallery with margin gating |
| ANPR | Plate localisation → CRNN + CTC → **Indian number-plate grammar correction** |
| Virtual fence intrusion | Polygon zones and directed tripwires in normalised coordinates |
| Suspicious activity | Loitering, abandoned object, crowd gathering, rapid and wrong-direction movement |
| Night-time movement | Schedule windows that wrap midnight + measured-darkness gating + CLAHE enhancement |
| Camera tamper | Lens obstruction, defocus and repositioning — detected while the stream stays up |
| Real-time alerts and logging | WebSocket feed, evidence with chain of custody, audit trail |
| C2 integration | Signed HTTP webhooks with store-and-forward |

## Quick start

```bash
git clone https://github.com/tedo001/border-serv-ai
cd border-serv-ai
python app.py
```

That is the whole thing. `app.py` is standard-library only, so it runs on a bare
checkout: it creates the virtual environment, installs the project, writes a
working configuration with synthetic cameras, generates a signing key, starts
the node and opens the console. A control panel window appears where a graphical
toolkit is available, and a numbered text menu where it is not.

Individual actions: `python app.py start | desktop | status | logs | test | stop`.

<details>
<summary>Or set it up by hand</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,desktop]"

cp configs/site.example.yaml configs/site.yaml     # then edit
ibvap validate configs/site.yaml

export IBVAP_SECURITY__JWT_SECRET=$(ibvap secret | cut -d= -f2)
ibvap serve --config configs/site.yaml
```

</details>

Open <http://127.0.0.1:8080/>. The bootstrap admin password is printed once at
startup.

No cameras to hand? Use a synthetic source — it generates video in-process:

```yaml
cameras:
  - id: cam-north
    url: "synthetic://?width=640&height=480&fps=20"
    target_fps: 8
    rules:
      - { id: presence, type: presence }
```

## Real detection in two commands

Out of the box the platform runs on classical motion detection and says so.
To give it real weights:

```bash
pip install -e '.[torch,sv]'      # RT-DETR + the live-view annotators
ibvap analyst                     # pick rtdetr-l, press Start Analysis
```

The checkpoint downloads into `models/weights` on first use. For a border post,
export the same weights to ONNX so the node needs no Torch at all:

```bash
ibvap models fetch rtdetr-l                    # -> ONNX, for a CPU post
ibvap models fetch rtdetr-l --format engine --half   # -> TensorRT, on a GPU node
```

A TensorRT plan is built by the GPU that will run it and refuses to load
anywhere else, so it is a commissioning step on the node — not something the
site build carries.

Both runtimes carry the same weights and agree to within 0.006 of score and two
pixels of box on the same frame, so a threshold tuned in the console means the
same thing in the field.

**Running in PyCharm?** Follow [`docs/PYCHARM.md`](docs/PYCHARM.md) — a
step-by-step guide from clone to live alerts.

## Three consoles and a control panel

**Control panel** (`python app.py`) — one window for the whole platform: set up
the environment, start and stop the node, open any console, run the tests,
benchmark the machine. It shows the node's live status, its detector mode and
every camera while it runs. On a bare checkout it falls back to a plain window
so it can install the environment that the real one needs.

**Live analysis** (`ibvap analyst`) — the quickest way to see the platform
work. One window: pick a source (an RTSP camera, a video file, or a built-in
simulated border scenario that needs no hardware at all), pick a detector,
drag the confidence, IoU and risk thresholds, draw a fence line straight onto
the video, and watch the annotated result with live counters and an alert log
you can export as JSON. It needs no node, no database and no login — it builds
the same camera worker the node runs, so a threshold that looks right here
behaves the same way in the field.

```bash
ibvap analyst
```

**Browser** (bundled, served by the node) — video wall, alert triage with
evidence review, watchlists, system dashboard. Dependency-free: no build step
and no CDN, because a BOP node is often air-gapped or on a metered link.

**Desktop** (`ibvap console`) — everything the browser console does, plus an
interactive **zone editor**: draw fences and tripwires directly onto a live
frame from the camera they belong to, and save them back to the node.

```bash
ibvap console --node http://127.0.0.1:8080
```

## Designed for where it runs

Remote border posts are not data centres, and the design reflects that:

- **Works without model artefacts.** Falls back to classical detection and says
  so, in `/health`, the console and metrics — never silently.
- **Survives losing the uplink.** Alerts raised during an outage are persisted
  and replayed when the link returns.
- **Drops frames, never latency.** Under load, old frames are discarded so
  alerts stay live rather than becoming a historical record.
- **Suppresses the false alarms that matter.** Livestock on a rural fence line
  is the dominant false-alarm source; it is suppressible by class.
- **Reads a plate by agreement, not by luck.** A Kalman filter follows the
  plate box through the frames the detector loses it, and the read is decided
  by a weighted vote across the vehicle's whole passage — with the evidence
  attached to the event, because `HR26DK8337` and `HR26DK8837` are not the
  same vehicle.
- **Evidence you can defend.** SHA-256 per artefact and a per-day hash chain, so
  both alteration and deletion are detectable months later.
- **Runs unprivileged** on commodity hardware, from a fanless mini-PC upward.

## Documentation

| Document | Contents |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | How it fits together, and why |
| [PyCharm guide](docs/PYCHARM.md) | Step-by-step IDE setup |
| [Deployment](docs/DEPLOYMENT.md) | Sizing, Docker, systemd, secrets |
| [Security](docs/SECURITY.md) | Auth, RBAC, audit, privacy, evidence integrity |
| [Operations](docs/OPERATIONS.md) | Tuning, metrics, drift, troubleshooting |

## Command line

```bash
ibvap serve      --config configs/site.yaml   # run a node
ibvap console    --node http://…              # desktop operator console
ibvap analyst                                 # live analysis console
ibvap models fetch rtdetr-l                   # real weights -> ONNX -> registry
ibvap validate   configs/site.yaml            # check a site file
ibvap probe      rtsp://…                     # test a camera before configuring it
ibvap benchmark  --resolution 1920x1080       # how many cameras this box can carry
ibvap models     list | verify | register     # model registry
ibvap evidence   verify-chain 2026-08-24      # chain of custody
ibvap secret                                  # generate a signing key
```

## Development

```bash
pytest tests/ -q                    # 247 tests
pytest tests/unit -q                # fast subset (~5 s)
ruff check src tests
```

## Licence

Apache-2.0.
