# Running IBVAP in PyCharm

A step-by-step guide from a fresh clone to a running node with live alerts,
plus the desktop console and the test suite — all driven from the IDE.

Works with PyCharm Community or Professional, 2023.1 or newer.

---

## Step 1 — Open the project

1. **File → Open…**
2. Select the `border-serv-ai` folder (the one containing `pyproject.toml`).
3. Choose **Trust Project** when prompted.

> Open the *repository root*, not `src/`. PyCharm reads `pyproject.toml` from
> the root to understand the package layout.

---

## Step 2 — Create the interpreter

1. **File → Settings** (Windows/Linux) or **PyCharm → Settings** (macOS).
2. Go to **Project: border-serv-ai → Python Interpreter**.
3. Click the gear icon → **Add…**
4. Select **Virtualenv Environment → New environment**:
   - **Location**: `<project>/.venv`
   - **Base interpreter**: Python **3.10, 3.11 or 3.12**
5. Click **OK** and wait for indexing to finish.

---

## Step 3 — Install the project

Open the PyCharm terminal (**View → Tool Windows → Terminal**, or `Alt+F12`).
The prompt should already show `(.venv)`.

```bash
pip install --upgrade pip
pip install -e ".[dev,desktop]"
```

This installs IBVAP in editable mode plus the test tools and PyQt6. It takes a
few minutes — ONNX Runtime and OpenCV are large.

Verify:

```bash
ibvap --version
```

You should see `IBVAP 1.0.0`.

### If OpenCV fails to import on Linux

```bash
sudo apt-get install -y libgl1 libglib2.0-0 ffmpeg
```

Windows and macOS need nothing extra.

---

## Step 4 — Mark the sources root

So PyCharm resolves imports and stops underlining them in red:

1. In the Project tree, right-click the **`src`** folder.
2. **Mark Directory as → Sources Root**.

The folder turns blue. `import ibvap` now resolves.

---

## Step 5 — Create your site configuration

In the terminal:

```bash
cp configs/site.example.yaml configs/site.yaml
```

Open `configs/site.yaml` in the editor.

**To try it immediately with no cameras**, replace the whole `cameras:` block
with synthetic sources — these generate video in-process, so nothing external
is required:

```yaml
cameras:
  - id: cam-north
    name: North Tower
    url: "synthetic://?width=640&height=480&fps=20&seed=1"
    target_fps: 8
    zones:
      - id: fence-strip
        name: Fence Strip
        points: [[0.35, 0.30], [1.00, 0.30], [1.00, 1.00], [0.35, 1.00]]
    tripwires:
      - id: fence-line
        name: Fence Line
        start: [0.50, 0.10]
        end:   [0.50, 0.95]
        direction: any
        left_label: exfiltration
        right_label: infiltration
    rules:
      - id: fence-intrusion
        type: intrusion
        zones: [fence-strip]
        params: { confirm_frames: 2 }
        cooldown_seconds: 10
      - id: fence-crossing
        type: line_crossing
        tripwires: [fence-line]
        cooldown_seconds: 10

  - id: cam-gate
    name: Main Gate
    url: "synthetic://?width=640&height=480&fps=20&seed=5"
    target_fps: 8
    rules:
      - id: gate-presence
        type: presence
```

**For real cameras**, set `url` to your RTSP address instead:

```yaml
    url: rtsp://10.20.0.11:554/Streaming/Channels/101
```

Test the camera *before* committing it to configuration:

```bash
ibvap probe rtsp://10.20.0.11:554/Streaming/Channels/101
```

That reports resolution, measured frame rate, whether the frame is IR, and
whether the stream actually delivers frames.

Then validate the whole file — it catches rules pointing at zones that do not
exist, zones written in pixels instead of normalised coordinates, and similar
mistakes:

```bash
ibvap validate configs/site.yaml
```

---

## Step 6 — Generate the signing secret

```bash
ibvap secret
```

Copy the line it prints. It looks like:

```
IBVAP_SECURITY__JWT_SECRET=xTq9...long-random-string
```

You will paste this into the run configuration in the next step.

---

## Step 7 — Run configuration for the node

1. **Run → Edit Configurations… → `+` → Python**
2. Fill in:

| Field | Value |
|---|---|
| **Name** | `IBVAP Node` |
| **Run** | select **Module name** (not Script path) |
| **Module name** | `ibvap.cli` |
| **Parameters** | `serve --config configs/site.yaml` |
| **Working directory** | the project root |
| **Environment variables** | see below |

3. Click the **Environment variables** field and add:

| Name | Value |
|---|---|
| `IBVAP_SECURITY__JWT_SECRET` | *(the value from Step 6)* |
| `IBVAP_SECURITY__BOOTSTRAP_ADMIN_PASSWORD` | `ChangeThisPassphrase1!` |
| `PYTHONUNBUFFERED` | `1` |

> Setting the bootstrap password explicitly keeps it stable across restarts
> while you are developing. Leave it unset and the node generates a random one
> and prints it once at startup.

4. **Apply → OK**.

Now press **Run** (`Shift+F10`). The console shows:

```
IBVAP 1.0.0 — BOP Ranipur (bop-ranipur)
  tier      edge
  cameras   2 enabled
  console   http://0.0.0.0:8080/
  API docs  http://0.0.0.0:8080/api/docs
```

> A warning that no model artefacts were found is **expected** on a fresh
> checkout. The node falls back to classical motion detection and reports
> itself as `degraded`. It still detects, tracks and raises alerts.

---

## Step 8 — Open the operator console

Ctrl-click (or Cmd-click) `http://0.0.0.0:8080/` in the run console, or browse
to <http://127.0.0.1:8080/>.

Sign in with:

- **Username**: `admin`
- **Password**: `ChangeThisPassphrase1!`

You should see:

- **Video wall** — live tiles with zone overlays, tripwire direction arrows,
  tracked figures and motion trails
- **Alerts** — intrusion and line-crossing events appearing in real time, with
  annotated evidence and an integrity-verification button
- **Watchlists** — add a vehicle registration such as `MH12AB1234`
- **System** — node health, per-camera throughput, model status

Interactive API documentation is at <http://127.0.0.1:8080/api/docs>.

---

## Step 9 — Run configuration for the desktop console

1. **Run → Edit Configurations… → `+` → Python**
2. Fill in:

| Field | Value |
|---|---|
| **Name** | `IBVAP Desktop Console` |
| **Run** | **Module name** |
| **Module name** | `ibvap.desktop.app` |
| **Parameters** | `--node http://127.0.0.1:8080` |
| **Working directory** | the project root |

3. **Apply → OK**.

With `IBVAP Node` already running, start this configuration. A native window
opens; sign in with the same credentials.

The desktop console adds a **Zone editor** the browser does not have: pick a
camera, click **Draw zone**, left-click to place points, right-click to close
the polygon, name it, then **Save to camera**. The geometry is written back to
the node and the camera restarts with it.

### On a headless Linux machine

Add an environment variable `QT_QPA_PLATFORM=offscreen` to render without a
display (useful for smoke tests), or run it on a desktop session.

---

## Step 10 — Run the tests

The simplest route is the gutter: open any file under `tests/` and click the
green ▶ beside a test or class.

For a run configuration:

1. **Run → Edit Configurations… → `+` → pytest**
2. Fill in:

| Field | Value |
|---|---|
| **Name** | `All tests` |
| **Target** | **Script path** → `tests` |
| **Working directory** | the project root |

3. **Apply → OK**.

Expect **247 passed** in about a minute.

To run only the fast unit tests, set the target to `tests/unit` (about five
seconds).

### Coverage

With PyCharm Professional, use **Run → Run 'All tests' with Coverage**. On
Community, use the terminal:

```bash
pytest tests/ --cov=ibvap --cov-report=html
```

Then open `htmlcov/index.html`.

---

## Step 11 — Debugging

Breakpoints work normally throughout. The most useful places to start:

| What you want to understand | Where to break |
|---|---|
| Why a rule did or did not fire | `src/ibvap/analytics/rules.py`, inside the rule's `evaluate` |
| What the detector returned | `src/ibvap/pipeline/worker.py`, in `process_frame` after `_detect` |
| Why a track lost its identity | `src/ibvap/vision/tracker.py`, in `ByteTracker.update` |
| Why an alert was suppressed | `src/ibvap/analytics/engine.py`, in `EventGate.admit` |
| An ANPR misread | `src/ibvap/vision/anpr.py`, in `normalise_plate` |

Run the `IBVAP Node` configuration in **Debug** mode (`Shift+F9`) and the
breakpoints hit on live synthetic traffic.

> Camera work happens on background threads. In the Debugger panel, use the
> **Frames** dropdown to switch to the `worker-cam-north` thread.

---

## Optional — useful extra run configurations

Same recipe (**`+` → Python**, **Module name** `ibvap.cli`), varying only the
parameters:

| Name | Parameters | Purpose |
|---|---|---|
| Validate config | `validate configs/site.yaml` | Check a site file before deploying it |
| Probe camera | `probe rtsp://…` | Test a camera URL |
| Benchmark | `benchmark --resolution 1920x1080` | How many cameras this machine can carry |
| List models | `models list` | Inspect the model registry |
| Verify evidence | `evidence verify-chain 2026-08-24` | Check a day's chain of custody |

---

## Optional — enable the neural detector

Out of the box the node runs classical fallback detection. To use YOLO26:

```bash
pip install -e ".[export]"          # brings in torch + ultralytics

ibvap models export yolo26s.pt \
    -o models/detector/yolo26s-border-1.0.0.onnx \
    --family yolo26 --imgsz 640 \
    --register yolo26s-border:1.0.0 \
    --card docs/model-cards/yolo26s-border.md
```

That exports to ONNX, verifies the graph's actual output layout, writes the
checksum into `models/registry.yaml`, and produces a model card.

Restart the node. `/health` should now report `detector_mode: neural` and
`degraded: false`.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: ibvap` | Step 3 not run, or `src` not marked as Sources Root (Step 4) |
| `ImportError: libGL.so.1` | `sudo apt-get install libgl1 libglib2.0-0` |
| `Address already in use` | Another node is running; stop it, or add `--port 8081` |
| `security.jwt_secret must be at least 32 bytes` | The env var in Step 7 is missing or truncated |
| Console shows `degraded` | Expected without model artefacts — see the optional step above |
| Video tiles stay black | The camera is offline. Check the **System** tab, and run `ibvap probe <url>` |
| PyQt window will not open on Linux | Install `libegl1 libxkbcommon-x11-0`, or set `QT_QPA_PLATFORM=offscreen` |
| `401` on every API call | The token expired (60 min default). Sign in again |
