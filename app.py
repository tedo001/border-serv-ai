#!/usr/bin/env python3
"""IBVAP launcher - one-click control panel for the whole platform.

Run this file and everything else follows: it builds a virtual environment,
installs the project, writes a working configuration, generates a signing key,
starts the analytics node and opens the operator console.

    python app.py

It is deliberately dependency-free and standard-library only, because the very
first thing it has to do is create the environment that holds the dependencies.
It therefore runs on a bare checkout, straight after `git clone`, with nothing
installed.

Three ways to drive it, all equivalent:

* **Click it.** Double-click the file, or press Run in PyCharm or Antigravity.
  A small window opens with a button per action. Where the system Python has no
  Tk (common on minimal Linux), it falls back to a numbered text menu.
* **From a terminal.** `python app.py start`, `python app.py test`, and so on.
* **From an IDE run configuration.** Point it at this file and pass the action
  as a parameter. See `docs/PYCHARM.md`.

Nothing here is required to run IBVAP - `ibvap serve` remains the supported
production entry point, and this launcher only ever calls the same CLI. It
exists so that a reviewer who has just cloned the repository can get to a
running system without reading the documentation first.
"""

from __future__ import annotations

import json
import os
import platform
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
CONFIG = ROOT / "configs" / "site.yaml"
EXAMPLE_CONFIG = ROOT / "configs" / "site.example.yaml"
ENV_FILE = ROOT / ".env"
VAR = ROOT / "var"
LOG_FILE = VAR / "node.log"
PID_FILE = VAR / "node.pid"

HOST = "127.0.0.1"
PORT = int(os.environ.get("IBVAP_PORT", "8080"))
BASE_URL = f"http://{HOST}:{PORT}"

IS_WINDOWS = os.name == "nt"

#: Demo password for the bootstrap administrator. Only ever used for the
#: local demonstration configuration this launcher writes; a real deployment
#: injects its own and is told to change it at first login.
DEMO_PASSWORD = "ChangeThisPassphrase1!"


# --------------------------------------------------------------------------- #
# Small output helpers
# --------------------------------------------------------------------------- #

class Reporter:
    """Sends progress somewhere sensible.

    The GUI swaps in a callback that appends to its log pane; on the terminal
    it just prints. Keeping this behind one object means every action below is
    written once and works in both.
    """

    def __init__(self) -> None:
        self._sink = None

    def bind(self, sink) -> None:
        self._sink = sink

    def __call__(self, message: str = "") -> None:
        if self._sink is not None:
            self._sink(message)
        else:
            print(message, flush=True)

    def step(self, message: str) -> None:
        self(f"\n=== {message} ===")

    def ok(self, message: str) -> None:
        self(f"  [ok] {message}")

    def warn(self, message: str) -> None:
        self(f"  [!] {message}")

    def fail(self, message: str) -> None:
        self(f"  [x] {message}")


say = Reporter()


# --------------------------------------------------------------------------- #
# Environment bootstrap
# --------------------------------------------------------------------------- #

def venv_python() -> Path:
    """Interpreter inside the project virtual environment."""
    return VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def venv_script(name: str) -> Path:
    """A console script installed into the virtual environment."""
    suffix = ".exe" if IS_WINDOWS else ""
    return VENV / ("Scripts" if IS_WINDOWS else "bin") / f"{name}{suffix}"


def venv_ready() -> bool:
    return venv_python().is_file()


def project_installed() -> bool:
    """Whether ibvap is importable from the virtual environment."""
    if not venv_ready():
        return False
    result = subprocess.run(
        [str(venv_python()), "-c", "import ibvap"],
        capture_output=True, cwd=ROOT,
    )
    return result.returncode == 0


def run(
    command: list[str], *, check: bool = True, quiet: bool = False,
    env: dict[str, str] | None = None,
) -> int:
    """Run a command, streaming its output through the reporter."""
    if not quiet:
        say(f"  $ {' '.join(str(c) for c in command)}")
    process = subprocess.Popen(
        [str(c) for c in command], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line and not quiet:
            say(f"    {line}")
    code = process.wait()
    if check and code != 0:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}")
    return code


def create_venv() -> None:
    """Create the virtual environment if it is missing."""
    if venv_ready():
        say.ok(f"virtual environment present at {VENV.name}/")
        return
    say(f"  creating a virtual environment at {VENV.name}/ (this is a one-off)")
    run([sys.executable, "-m", "venv", str(VENV)])
    say.ok("virtual environment created")


def install_project(extras: str = "dev,desktop") -> None:
    """Install IBVAP in editable mode with the requested extras."""
    python = str(venv_python())
    run([python, "-m", "pip", "install", "--upgrade", "--quiet", "pip"], quiet=True)

    say(f"  installing the project with extras [{extras}] - this takes a few minutes "
        "the first time")
    try:
        run([python, "-m", "pip", "install", "--quiet", "-e", f".[{extras}]"])
    except RuntimeError:
        # PyQt6 needs system graphics libraries that a headless server may not
        # have. The desktop console is optional, so fall back rather than
        # leaving the whole installation broken.
        say.warn("full install failed - retrying without the desktop console")
        run([python, "-m", "pip", "install", "--quiet", "-e", ".[dev]"])
        say.warn("desktop console unavailable; the browser console still works")
    say.ok("project installed")


def python_version_ok() -> bool:
    return sys.version_info >= (3, 10)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEMO_CAMERAS = """
# --- Demonstration cameras -------------------------------------------------
# `synthetic://` sources generate video inside the process, so this file runs
# with no cameras, no network and no hardware. Replace the URLs with your own
# RTSP addresses when you have them - test one first with:
#
#     ibvap probe rtsp://10.20.0.11:554/Streaming/Channels/101
#
cameras:
  - id: cam-north
    name: North Tower
    url: "synthetic://?width=640&height=480&fps=20&seed=1"
    location: BOP Ranipur - North Tower
    latitude: 26.9124
    longitude: 88.4270
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
      - id: tamper
        type: camera_tamper
        params: { confirm_frames: 15, warmup_frames: 30 }

  - id: cam-gate
    name: Main Gate
    url: "synthetic://?width=640&height=480&fps=20&seed=5"
    location: BOP Ranipur - Main Gate
    target_fps: 8
    rules:
      - id: gate-presence
        type: presence

  - id: cam-road
    name: Border Road KM-12
    url: "synthetic://?width=640&height=480&fps=20&seed=9&night=1"
    location: Border Road KM-12
    target_fps: 8
    rules:
      - id: night-watch
        type: night_movement
        params: { min_speed: 0.01, confirm_frames: 3, require_dark: true }
"""


def ensure_config() -> None:
    """Write a runnable site configuration if none exists."""
    if CONFIG.is_file():
        say.ok(f"configuration present at {CONFIG.relative_to(ROOT)}")
        return
    if not EXAMPLE_CONFIG.is_file():
        raise RuntimeError(f"missing {EXAMPLE_CONFIG} - is this a complete checkout?")

    text = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    # Replace the example's RTSP cameras with synthetic ones so the demo runs
    # with no hardware. Everything above `cameras:` is kept as written.
    head, _, _ = text.partition("\ncameras:")
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(head.rstrip() + "\n" + DEMO_CAMERAS, encoding="utf-8")
    say.ok(f"wrote {CONFIG.relative_to(ROOT)} with three synthetic cameras")
    say("      (edit it to point at real RTSP cameras when you have them)")


def ensure_secret() -> dict[str, str]:
    """Return the environment for the node, creating secrets on first run.

    Secrets live in `.env`, which is git-ignored. They are never written into
    the site YAML, because that file gets copied between posts and attached to
    tickets.
    """
    values: dict[str, str] = {}
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()

    if "IBVAP_SECURITY__JWT_SECRET" not in values:
        values["IBVAP_SECURITY__JWT_SECRET"] = secrets.token_urlsafe(48)
        values.setdefault("IBVAP_SECURITY__BOOTSTRAP_ADMIN_PASSWORD", DEMO_PASSWORD)
        ENV_FILE.write_text(
            "# IBVAP local secrets. Git-ignored - never commit this file.\n"
            "# Regenerate the signing key at any time with: ibvap secret\n"
            + "".join(f"{k}={v}\n" for k, v in values.items()),
            encoding="utf-8",
        )
        say.ok(f"generated signing key and wrote {ENV_FILE.name}")
    else:
        say.ok("signing key loaded from .env")

    environment = os.environ.copy()
    environment.update(values)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


# --------------------------------------------------------------------------- #
# Node lifecycle
# --------------------------------------------------------------------------- #

def port_in_use() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((HOST, PORT)) == 0


def health() -> dict | None:
    """Fetch node health, or None when it is not answering."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=4) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def running_pid() -> int | None:
    """PID of a node this launcher started, if it is still alive."""
    if not PID_FILE.is_file():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    try:
        if IS_WINDOWS:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True
            )
            return pid if str(pid) in result.stdout else None
        os.kill(pid, 0)  # signal 0 only checks for existence
        return pid
    except (OSError, subprocess.SubprocessError):
        return None


def start_node(open_browser: bool = True) -> bool:
    """Start the analytics node in the background and wait for it to be ready."""
    if health() is not None:
        say.ok(f"a node is already serving on {BASE_URL}")
        if open_browser:
            webbrowser.open(BASE_URL)
        return True
    if port_in_use():
        say.fail(f"port {PORT} is busy but is not answering as IBVAP")
        say("      stop whatever is using it, or set IBVAP_PORT to another port")
        return False

    # The invariant lives here rather than in the callers: *every* path that
    # starts a node needs the project installed, and putting the check in one
    # action meant the others launched a node into an empty environment and
    # died with "No module named 'ibvap'" - an error that looks like a broken
    # product rather than an unfinished setup step.
    if not project_installed():
        say.warn("the project is not installed in .venv yet - setting up first")
        if not action_setup():
            return False

    ensure_config()
    environment = ensure_secret()
    VAR.mkdir(parents=True, exist_ok=True)

    say(f"  starting the node; log -> {LOG_FILE.relative_to(ROOT)}")
    with LOG_FILE.open("w", encoding="utf-8") as log:
        # Detached from this process group so the node keeps running if the
        # launcher window is closed.
        creation = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
        # --host/--port are passed explicitly rather than left to the site
        # file: IBVAP_PORT has to move the port the node actually binds, not
        # just the one this launcher polls, or the two silently disagree and
        # the node looks like it failed to start when it is running fine.
        process = subprocess.Popen(
            [
                str(venv_python()), "-m", "ibvap.cli", "serve",
                "--config", str(CONFIG), "--host", HOST, "--port", str(PORT),
            ],
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=not IS_WINDOWS, creationflags=creation,
        )
    PID_FILE.write_text(str(process.pid), encoding="utf-8")

    say("  waiting for the node to report ready")
    for _attempt in range(60):
        if process.poll() is not None:
            say.fail(f"the node exited immediately (code {process.returncode})")
            tail = LOG_FILE.read_text(encoding="utf-8").splitlines()[-12:]
            if any("No module named 'ibvap'" in line for line in tail):
                say("      the virtual environment exists but the project is not")
                say("      installed into it. Run 'Set up', or from a terminal:")
                say(f"        {venv_python()} -m pip install -e .")
            say("      last lines of the log:")
            for line in tail:
                say(f"        {line}")
            return False
        snapshot = health()
        if snapshot is not None:
            say.ok(
                f"node ready: {snapshot.get('status')} - "
                f"{snapshot.get('cameras_online')}/{snapshot.get('cameras_total')} cameras online"
            )
            models = snapshot.get("models", {})
            if models.get("degraded"):
                say("      running classical fallback detection (no model artefacts "
                    "installed); accuracy is reduced but analytics work")
            say("")
            say(f"      console   {BASE_URL}/")
            say(f"      API docs  {BASE_URL}/api/docs")
            say(f"      sign in   admin / {DEMO_PASSWORD}")
            if open_browser:
                webbrowser.open(BASE_URL)
            return True
        time.sleep(1)

    say.fail("the node did not become ready within 60 seconds")
    return False


def stop_node() -> bool:
    """Stop a node started by this launcher."""
    pid = running_pid()
    if pid is None:
        if health() is not None:
            say.warn("a node is serving but was not started by this launcher; "
                     "stop it where you started it")
            return False
        say.ok("no node is running")
        return True

    say(f"  stopping process {pid}")
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError as exc:
        say.fail(f"could not stop it: {exc}")
        return False

    for _ in range(15):
        if running_pid() is None:
            break
        time.sleep(0.5)
    PID_FILE.unlink(missing_ok=True)
    say.ok("node stopped")
    return True


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #

def action_setup() -> bool:
    """Prepare everything needed to run, without starting anything."""
    say.step("Setting up")
    if not python_version_ok():
        say.fail(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old; "
                 "3.10 or newer is required")
        return False
    say.ok(f"Python {platform.python_version()} on {platform.system()}")
    create_venv()
    if not project_installed():
        install_project()
    else:
        say.ok("project already installed")
    ensure_config()
    ensure_secret()
    say("\n  Ready. Choose 'Start node' to bring the platform up.")
    return True


def action_start() -> bool:
    """Set up if needed, then start the node and open the console."""
    say.step("Starting the node")
    return start_node()


def action_stop() -> bool:
    say.step("Stopping the node")
    return stop_node()


def action_status() -> bool:
    say.step("Status")
    say.ok(f"virtual environment : {'present' if venv_ready() else 'missing'}")
    say.ok(f"project installed   : {'yes' if project_installed() else 'no'}")
    say.ok(f"configuration       : "
           f"{CONFIG.relative_to(ROOT) if CONFIG.is_file() else 'not written yet'}")

    snapshot = health()
    if snapshot is None:
        say.ok("node                : not running")
        return True

    say.ok(f"node                : {snapshot.get('status')} on {BASE_URL}")
    say(f"      site       {snapshot.get('site_name')} ({snapshot.get('site_id')})")
    say(f"      cameras    {snapshot.get('cameras_online')}/{snapshot.get('cameras_total')} online")
    say(f"      uptime     {snapshot.get('uptime_seconds', 0) / 60:.1f} minutes")
    models = snapshot.get("models", {})
    say(f"      detector   {models.get('detector_mode')} "
        f"({'degraded' if models.get('degraded') else 'neural'})")
    storage = (snapshot.get("storage") or {}).get("evidence", {})
    say(f"      evidence   {storage.get('megabytes', 0)} MB in {storage.get('files', 0)} files")
    return True


def action_console() -> bool:
    """Open the browser operator console."""
    say.step("Opening the browser console")
    if health() is None:
        say.warn("no node is running - starting one first")
        if not start_node(open_browser=False):
            return False
    webbrowser.open(BASE_URL)
    say.ok(f"opened {BASE_URL}")
    say(f"      sign in with admin / {DEMO_PASSWORD}")
    return True


def action_desktop() -> bool:
    """Launch the native PyQt desktop console."""
    say.step("Launching the desktop console")
    check = subprocess.run(
        [str(venv_python()), "-c", "import PyQt6"], capture_output=True, cwd=ROOT
    )
    if check.returncode != 0:
        say.fail("PyQt6 is not installed in this environment")
        say("      install it with:  .venv/bin/pip install -e '.[desktop]'")
        say("      on Linux you may also need:  sudo apt-get install libegl1 libgl1 "
            "libxkbcommon-x11-0")
        return False

    if health() is None:
        say.warn("no node is running - starting one first")
        if not start_node(open_browser=False):
            return False

    say(f"  connecting the desktop console to {BASE_URL}")
    subprocess.Popen(
        [str(venv_python()), "-m", "ibvap.desktop.app", "--node", BASE_URL],
        cwd=ROOT, env=ensure_secret(),
    )
    say.ok("desktop console launched in a separate window")
    say(f"      sign in with admin / {DEMO_PASSWORD}")
    return True


def action_test() -> bool:
    """Run the full test suite."""
    say.step("Running the test suite")
    if not project_installed():
        say.warn("project not installed - setting up first")
        if not action_setup():
            return False
    code = run([str(venv_python()), "-m", "pytest", "tests/", "-q"], check=False)
    if code == 0:
        say.ok("all tests passed")
    else:
        say.fail(f"tests failed (exit code {code})")
    return code == 0


def action_validate() -> bool:
    """Validate the site configuration."""
    say.step("Validating the configuration")
    ensure_config()
    # Pass the same environment the node runs with: without it the CLI reports
    # a missing signing key, which is confusing when .env already holds one.
    code = run(
        [str(venv_python()), "-m", "ibvap.cli", "validate", str(CONFIG)],
        check=False, env=ensure_secret(),
    )
    return code == 0


def action_benchmark() -> bool:
    """Measure how many cameras this machine can carry."""
    say.step("Benchmarking this machine")
    code = run(
        [str(venv_python()), "-m", "ibvap.cli", "benchmark", "--resolution", "1280x720"],
        check=False, env=ensure_secret(),
    )
    return code == 0


def action_logs() -> bool:
    """Show the tail of the node log."""
    say.step("Node log")
    if not LOG_FILE.is_file():
        say.warn("no log yet - start the node first")
        return True
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines()[-40:]:
        say(f"  {line}")
    return True


def action_clean() -> bool:
    """Remove generated runtime data, keeping configuration and secrets."""
    say.step("Cleaning runtime data")
    stop_node()
    removed = []
    for target in (ROOT / "data", VAR):
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            removed.append(target.name + "/")
    say.ok(f"removed {', '.join(removed)}" if removed else "nothing to remove")
    say("      configuration and .env were kept")
    return True


#: Ordered action table, shared by the GUI, the text menu and the CLI.
ACTIONS: list[tuple[str, str, object]] = [
    ("setup",     "Set up (create environment, install)", action_setup),
    ("start",     "Start node and open console",          action_start),
    ("console",   "Open browser console",                 action_console),
    ("desktop",   "Open desktop console",                 action_desktop),
    ("status",    "Status",                               action_status),
    ("logs",      "Show node log",                        action_logs),
    ("stop",      "Stop node",                            action_stop),
    ("test",      "Run tests",                            action_test),
    ("validate",  "Validate configuration",               action_validate),
    ("benchmark", "Benchmark this machine",               action_benchmark),
    ("clean",     "Clean runtime data",                   action_clean),
]

BY_NAME = {name: function for name, _, function in ACTIONS}


# --------------------------------------------------------------------------- #
# Graphical control panel
# --------------------------------------------------------------------------- #

def run_gui() -> int:
    """A small Tk control panel. Returns 1 when Tk is unavailable."""
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except ImportError:
        return 1

    import queue
    import threading

    window = tk.Tk()
    window.title("IBVAP - Intelligent Border Video Analytics Platform")
    window.geometry("880x600")
    window.configure(bg="#0d1117")

    header = tk.Frame(window, bg="#161b22", height=54)
    header.pack(fill="x")
    header.pack_propagate(False)
    tk.Label(header, text="IBVAP", bg="#161b22", fg="#2f81f7",
             font=("Helvetica", 15, "bold")).pack(side="left", padx=(16, 8))
    tk.Label(header, text="Intelligent Border Video Analytics Platform",
             bg="#161b22", fg="#8b949e", font=("Helvetica", 10)).pack(side="left")
    state_label = tk.Label(header, text="checking...", bg="#161b22", fg="#8b949e",
                           font=("Helvetica", 9, "bold"))
    state_label.pack(side="right", padx=16)

    body = tk.Frame(window, bg="#0d1117")
    body.pack(fill="both", expand=True, padx=12, pady=12)

    buttons = tk.Frame(body, bg="#0d1117")
    buttons.pack(side="left", fill="y", padx=(0, 12))

    log = scrolledtext.ScrolledText(
        body, bg="#0a0e14", fg="#e6edf3", insertbackground="#e6edf3",
        font=("Courier", 9), relief="flat", wrap="word",
    )
    log.pack(side="right", fill="both", expand=True)
    log.insert("end", "Press 'Start node and open console' to bring the platform up.\n"
                      "First run creates a virtual environment and installs "
                      "dependencies, which takes a few minutes.\n")

    messages: queue.Queue[str] = queue.Queue()
    say.bind(messages.put)

    def drain() -> None:
        while True:
            try:
                line = messages.get_nowait()
            except queue.Empty:
                break
            log.insert("end", line + "\n")
            log.see("end")
        window.after(120, drain)

    widgets: list[tk.Button] = []

    def launch(function) -> None:
        for widget in widgets:
            widget.configure(state="disabled")

        def worker() -> None:
            try:
                function()
            except Exception as exc:  # a failed action must not kill the panel
                messages.put(f"  [x] {type(exc).__name__}: {exc}")
            finally:
                window.after(0, lambda: [w.configure(state="normal") for w in widgets])

        threading.Thread(target=worker, daemon=True).start()

    for name, label, function in ACTIONS:
        primary = name == "start"
        button = tk.Button(
            buttons, text=label, width=30, anchor="w",
            bg="#2f81f7" if primary else "#21262d",
            fg="#ffffff" if primary else "#e6edf3",
            activebackground="#388bfd", relief="flat", padx=10, pady=7,
            font=("Helvetica", 10, "bold" if primary else "normal"),
            command=lambda f=function: launch(f),
        )
        button.pack(fill="x", pady=2)
        widgets.append(button)

    def poll_state() -> None:
        snapshot = health()
        if snapshot is None:
            state_label.configure(text="NODE STOPPED", fg="#8b949e")
        else:
            status = str(snapshot.get("status", "?")).upper()
            colour = {"HEALTHY": "#3fb950", "DEGRADED": "#d29922",
                      "PARTIAL": "#d29922"}.get(status, "#f85149")
            state_label.configure(
                text=f"{status}  {snapshot.get('cameras_online')}/"
                     f"{snapshot.get('cameras_total')} cameras",
                fg=colour,
            )
        window.after(5000, poll_state)

    drain()
    poll_state()
    window.mainloop()
    return 0


# --------------------------------------------------------------------------- #
# Text menu
# --------------------------------------------------------------------------- #

def run_menu() -> int:
    """Numbered menu, for terminals and machines without Tk."""
    banner = textwrap.dedent(f"""
        ============================================================
          IBVAP - Intelligent Border Video Analytics Platform
          {ROOT}
        ============================================================
    """).strip()
    print(banner)

    while True:
        snapshot = health()
        state = (
            f"{snapshot.get('status')} - {snapshot.get('cameras_online')}/"
            f"{snapshot.get('cameras_total')} cameras online"
            if snapshot else "stopped"
        )
        print(f"\n  Node: {state}\n")
        for index, (_, label, _) in enumerate(ACTIONS, start=1):
            print(f"   {index:2d}. {label}")
        print("    0. Quit")

        try:
            choice = input("\n  Choose: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice in ("0", "q", "quit", "exit"):
            return 0
        if not choice.isdigit() or not (1 <= int(choice) <= len(ACTIONS)):
            print("  Not a valid choice.")
            continue

        try:
            ACTIONS[int(choice) - 1][2]()
        except Exception as exc:
            say.fail(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

USAGE = f"""IBVAP launcher

  python app.py              open the control panel (window, or menu without Tk)
  python app.py <action>     run one action and exit
  python app.py --menu       force the text menu
  python app.py --help       this message

Actions:
{chr(10).join(f'  {name:<11} {label}' for name, label, _ in ACTIONS)}

The node listens on {BASE_URL}. Override with the IBVAP_PORT environment variable.
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    os.chdir(ROOT)

    if argv and argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    if argv and argv[0] in BY_NAME:
        return 0 if BY_NAME[argv[0]]() else 1

    if argv and argv[0] not in ("--menu", "--gui"):
        print(f"Unknown action {argv[0]!r}.\n")
        print(USAGE)
        return 2

    force_menu = bool(argv) and argv[0] == "--menu"
    if not force_menu and run_gui() == 0:
        return 0
    if not force_menu:
        print("(no graphical toolkit available - using the text menu)\n")
    return run_menu()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130) from None
