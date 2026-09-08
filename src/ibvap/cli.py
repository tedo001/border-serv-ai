"""IBVAP command-line interface.

Covers the operations a deployment actually needs: run a node, validate a site
file before it is copied to a post, manage models, benchmark hardware to size a
site, verify evidence, and probe a camera before committing it to configuration.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path
from typing import Any

from ibvap import __version__


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    try:
        return int(args.handler(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        if getattr(args, "traceback", False):
            raise
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ibvap",
        description="IBVAP - Intelligent Border Video Analytics Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ibvap serve --config configs/site.yaml\n"
            "  ibvap validate configs/site.yaml\n"
            "  ibvap probe rtsp://10.0.0.5:554/Streaming/Channels/101\n"
            "  ibvap benchmark --resolution 1920x1080\n"
            "  ibvap models list\n"
            "  ibvap evidence verify-chain 2026-08-24\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"IBVAP {__version__}")
    parser.add_argument("--traceback", action="store_true", help="show full tracebacks")
    subparsers = parser.add_subparsers(dest="command")

    # -- serve --
    serve = subparsers.add_parser("serve", help="run an analytics node with its API")
    serve.add_argument("--config", "-c", help="path to the site YAML file")
    serve.add_argument("--host", help="override the API bind address")
    serve.add_argument("--port", type=int, help="override the API port")
    serve.add_argument("--reload", action="store_true", help="auto-reload on code change (development)")
    serve.set_defaults(handler=_serve)

    # -- console --
    console = subparsers.add_parser("console", help="launch the desktop operator console")
    console.add_argument("--node", "-n", default="http://127.0.0.1:8080", help="node address")
    console.set_defaults(handler=_console)

    # -- analyst --
    analyst = subparsers.add_parser(
        "analyst", help="launch the live analysis console (one source, no node)"
    )
    analyst.set_defaults(handler=_analyst)

    # -- validate --
    validate = subparsers.add_parser("validate", help="validate a site configuration file")
    validate.add_argument("config", help="path to the site YAML file")
    validate.set_defaults(handler=_validate)

    # -- probe --
    probe = subparsers.add_parser("probe", help="test a camera URL before configuring it")
    probe.add_argument("url", help="RTSP/HTTP URL, file path, or synthetic://")
    probe.add_argument("--frames", type=int, default=30, help="frames to read")
    probe.add_argument("--transport", default="tcp", choices=["tcp", "udp"])
    probe.add_argument("--save", help="write an annotated sample frame to this path")
    probe.set_defaults(handler=_probe)

    # -- benchmark --
    benchmark = subparsers.add_parser("benchmark", help="measure throughput and size a site")
    benchmark.add_argument("--config", "-c", help="site file, to benchmark the configured models")
    benchmark.add_argument("--resolution", default="1280x720", help="frame size, e.g. 1920x1080")
    benchmark.add_argument("--iterations", type=int, default=50)
    benchmark.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    benchmark.set_defaults(handler=_benchmark)

    # -- secret --
    secret = subparsers.add_parser("secret", help="generate a JWT signing key")
    secret.set_defaults(handler=_secret)

    # -- models --
    models = subparsers.add_parser("models", help="inspect and manage the model registry")
    model_actions = models.add_subparsers(dest="action")

    model_list = model_actions.add_parser("list", help="list registered models")
    model_list.add_argument("--config", "-c")
    model_list.set_defaults(handler=_models_list)

    model_verify = model_actions.add_parser("verify", help="verify artefact checksums")
    model_verify.add_argument("--config", "-c")
    model_verify.set_defaults(handler=_models_verify)

    model_fetch = model_actions.add_parser(
        "fetch",
        help="download a pretrained checkpoint, export ONNX and register it",
    )
    model_fetch.add_argument(
        "checkpoint",
        help="Ultralytics checkpoint, e.g. rtdetr-l.pt, rtdetr-x.pt, yolo11s.pt",
    )
    model_fetch.add_argument("--name", help="registry name (default: the checkpoint stem)")
    model_fetch.add_argument("--version", default="coco", help="registry version label")
    model_fetch.add_argument("--imgsz", type=int, default=640, help="export input size")
    model_fetch.add_argument("--opset", type=int, default=17, help="ONNX opset")
    model_fetch.add_argument("--registry", default="models/registry.yaml")
    model_fetch.add_argument("--models-dir", default="models")
    model_fetch.add_argument(
        "--no-export", action="store_true",
        help="download the checkpoint only; do not export or register ONNX",
    )
    model_fetch.set_defaults(handler=_models_fetch)

    model_register = model_actions.add_parser(
        "register", help="add an ONNX artefact to the registry"
    )
    model_register.add_argument("artefact", help="path to the .onnx file")
    model_register.add_argument("--name", required=True, help="registry model name")
    model_register.add_argument("--version", required=True, help="version, e.g. 1.0.0")
    model_register.add_argument("--role", default="detector",
                                help="detector, classifier, plate_detector, plate_ocr, "
                                     "face_detector or face_embedder")
    model_register.add_argument("--layout", default="auto",
                                help="output layout: rtdetr, yolo26, yolov8, yolov5, nms_xyxy")
    model_register.add_argument("--imgsz", type=int, default=640, help="model input size")
    model_register.add_argument("--classes", default="",
                                help="comma-separated class names, in model order")
    model_register.add_argument("--registry", default="models/registry.yaml")
    model_register.add_argument("--card", help="write a model card to this path")
    model_register.set_defaults(handler=_models_register)

    # -- evidence --
    evidence = subparsers.add_parser("evidence", help="verify and manage stored evidence")
    evidence_actions = evidence.add_subparsers(dest="action")

    evidence_verify = evidence_actions.add_parser("verify-chain", help="verify one day's hash chain")
    evidence_verify.add_argument("day", help="date as YYYY-MM-DD")
    evidence_verify.add_argument("--config", "-c")
    evidence_verify.set_defaults(handler=_evidence_verify)

    evidence_prune = evidence_actions.add_parser("prune", help="apply the retention policy now")
    evidence_prune.add_argument("--config", "-c")
    evidence_prune.set_defaults(handler=_evidence_prune)

    return parser


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def _load(args: Any):
    from ibvap.core.config import load_settings

    return load_settings(getattr(args, "config", None))


def _serve(args: Any) -> int:
    import uvicorn

    from ibvap.api.app import create_app

    settings = _load(args)
    if args.host:
        settings.api.host = args.host
    if args.port:
        settings.api.port = args.port

    print(
        f"IBVAP {__version__} — {settings.site_name} ({settings.site_id})\n"
        f"  tier      {settings.tier}\n"
        f"  cameras   {len(settings.enabled_cameras())} enabled\n"
        f"  console   http://{settings.api.host}:{settings.api.port}/\n"
        f"  API docs  http://{settings.api.host}:{settings.api.port}/api/docs\n"
    )
    uvicorn.run(
        create_app(settings),
        host=settings.api.host,
        port=settings.api.port,
        log_config=None,
        access_log=False,
    )
    return 0


def _console(args: Any) -> int:
    try:
        from ibvap.desktop.app import main as console_main
    except ImportError as exc:
        print(
            f"the desktop console requires PyQt6: pip install 'ibvap[desktop]'\n({exc})",
            file=sys.stderr,
        )
        return 1
    return console_main(["ibvap-console", "--node", args.node])


def _analyst(_args: Any) -> int:
    try:
        from ibvap.desktop.analyst import main as analyst_main
    except ImportError as exc:
        print(
            f"the live analysis console requires PyQt6: pip install 'ibvap[desktop]'\n({exc})",
            file=sys.stderr,
        )
        return 1
    return analyst_main()


def _validate(args: Any) -> int:
    from ibvap.analytics import available_rules
    from ibvap.core.config import load_settings

    settings = load_settings(args.config)
    known_rules = set(available_rules())

    print(f"✓ {args.config} parsed")
    print(f"  site     {settings.site_name} ({settings.site_id}), tier {settings.tier}")
    print(f"  cameras  {len(settings.cameras)} configured, {len(settings.enabled_cameras())} enabled")

    warnings: list[str] = []
    for camera in settings.cameras:
        for rule in camera.rules:
            if rule.type not in known_rules:
                warnings.append(
                    f"camera {camera.id}: rule {rule.id!r} has unknown type {rule.type!r}"
                )
        if camera.anpr_enabled and not settings.models.plate_ocr.name:
            warnings.append(f"camera {camera.id}: ANPR is enabled but no OCR model is configured")
        if camera.face_enabled and not settings.models.face_embedder.name:
            warnings.append(
                f"camera {camera.id}: face recognition is enabled but no embedder is configured"
            )
        if not camera.rules:
            warnings.append(f"camera {camera.id}: no analytics rules; it will only record")
        if camera.url.startswith("rtsp://") and "@" in camera.url:
            warnings.append(
                f"camera {camera.id}: credentials are embedded in the URL; "
                "prefer an environment variable so the site file can be shared safely"
            )

    if not settings.security.jwt_secret:
        warnings.append(
            "security.jwt_secret is not set; a random key will be generated at each "
            "start and all sessions will be invalidated on restart"
        )

    for camera in settings.cameras:
        for zone in camera.zones:
            import numpy as np

            from ibvap.core.geometry import polygon_area

            area = polygon_area(np.array(zone.points))
            if area < 0.005:
                warnings.append(
                    f"camera {camera.id}: zone {zone.id!r} covers only "
                    f"{area * 100:.2f}% of the frame and may be too small to trigger"
                )

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    else:
        print("\nno warnings")
    return 0


def _probe(args: Any) -> int:
    import time

    import cv2

    from ibvap.ingest.source import build_source
    from ibvap.vision.preprocess import is_infrared, is_night_frame, mean_luma, sharpness

    source = build_source(args.url, rtsp_transport=args.transport)
    print(f"opening {args.url} …")
    started = time.perf_counter()
    source.open()
    open_ms = (time.perf_counter() - started) * 1000

    info = source.info
    print(f"✓ opened in {open_ms:.0f} ms")
    print(f"  resolution  {info.width}x{info.height}")
    print(f"  source fps  {info.fps:.1f}")
    print(f"  backend     {info.backend}")

    frames = 0
    first = None
    started = time.perf_counter()
    try:
        for _ in range(args.frames):
            frame = source.read()
            if first is None:
                first = frame
            frames += 1
    except Exception as exc:
        print(f"  ⚠ stream ended after {frames} frames: {exc}")
    finally:
        source.close()

    elapsed = time.perf_counter() - started
    if frames:
        print(f"  measured    {frames / max(elapsed, 1e-6):.1f} fps over {frames} frames")
    if first is not None:
        luma = mean_luma(first)
        print(f"  luminance   {luma:.0f} ({'night/IR' if is_night_frame(first) else 'daylight'})")
        print(f"  monochrome  {'yes (IR)' if is_infrared(first) else 'no'}")
        print(f"  sharpness   {sharpness(first):.0f}")
        if args.save:
            cv2.imwrite(args.save, first)
            print(f"  sample      written to {args.save}")

    if frames == 0:
        print("\n✗ the stream opened but delivered no frames")
        return 1
    return 0


def _benchmark(args: Any) -> int:
    from ibvap.mlops.benchmark import (
        benchmark_pipeline_stages,
        format_report,
        system_profile,
    )
    from ibvap.vision.tracker import ByteTracker

    try:
        width, height = (int(v) for v in args.resolution.lower().split("x"))
    except ValueError:
        print(f"invalid resolution {args.resolution!r}; expected WIDTHxHEIGHT", file=sys.stderr)
        return 1

    settings = _load(args)
    from ibvap.pipeline.models import build_model_bundle

    bundle = build_model_bundle(settings)
    if bundle.detector is None:
        print("no detector is available", file=sys.stderr)
        return 1

    print(f"benchmarking {bundle.detector_mode} detector at {width}x{height} …\n")
    results = benchmark_pipeline_stages(
        bundle.detector, ByteTracker(),
        resolution=(width, height), iterations=args.iterations,
    )
    if args.json:
        print(json.dumps(
            {
                "profile": system_profile(),
                "detector_mode": bundle.detector_mode,
                "results": {k: v.as_dict() for k, v in results.items()},
            },
            indent=2,
        ))
    else:
        print(format_report(results, system_profile()))
    bundle.close()
    return 0


def _secret(_args: Any) -> int:
    key = secrets.token_urlsafe(48)
    print(f"IBVAP_SECURITY__JWT_SECRET={key}")
    print(
        "\nSet this in the node's environment (not in the site YAML - that file "
        "gets copied between posts and attached to tickets).",
        file=sys.stderr,
    )
    return 0


def _models_list(args: Any) -> int:
    from ibvap.mlops.registry import ModelRegistry

    registry = ModelRegistry.load(_load(args).models)
    entries = registry.describe()
    if not entries:
        print("no models registered")
        print("register one with:  ibvap models export <weights.pt> -o <model.onnx> "
              "--register name:version")
        return 0

    header = f"{'name':<24}{'role':<16}{'version':<10}{'layout':<12}{'present':<9}{'default'}"
    print(header)
    print("-" * len(header))
    for entry in entries:
        print(
            f"{entry['name']:<24}{entry['role']:<16}{entry['version']:<10}"
            f"{entry['layout']:<12}{'yes' if entry['present'] else 'NO':<9}"
            f"{'*' if entry['is_default'] else ''}"
        )
        if entry["metrics"]:
            print(f"{'':<24}metrics: {entry['metrics']}")
        if not entry["checksum_declared"]:
            print(f"{'':<24}⚠ no checksum declared - integrity is unverified")
    return 0


def _models_verify(args: Any) -> int:
    from ibvap.core.errors import ModelError
    from ibvap.mlops.registry import ModelRegistry

    registry = ModelRegistry.load(_load(args).models)
    failures = 0
    for entry in registry.describe():
        name, version = entry["name"], entry["version"]
        try:
            registry.verify(name, version)
            print(f"✓ {name}:{version}")
        except ModelError as exc:
            failures += 1
            print(f"✗ {name}:{version} — {exc}")
    if failures:
        print(f"\n{failures} artefact(s) failed verification", file=sys.stderr)
    return 1 if failures else 0


def _models_fetch(args: Any) -> int:
    """Turn a published checkpoint into a registered ONNX artefact.

    This is the whole model lifecycle in one command, and it is the step that
    makes the platform's detection real rather than declared:

        download  ->  run it (Torch)  ->  export ONNX  ->  hash  ->  register

    The Torch runtime is for development and evaluation; the ONNX artefact it
    exports is what goes to a post, on a node that never needs Torch. Both
    carry the same weights, so a threshold tuned in the analyst console means
    the same thing in the field.
    """
    from ibvap.mlops.registry import register_model, sha256_file
    from ibvap.vision.ultralytics_detector import (
        UltralyticsDetector,
        checkpoint_classes,
        export_onnx,
        is_rtdetr,
    )

    checkpoint = args.checkpoint if args.checkpoint.endswith(".pt") else f"{args.checkpoint}.pt"
    name = args.name or Path(checkpoint).stem
    models_dir = Path(args.models_dir)

    print(f"fetching {checkpoint} …")
    resolved = UltralyticsDetector._resolve(checkpoint, models_dir)
    size_mb = round(resolved.stat().st_size / 1024**2, 2)
    print(f"  checkpoint  {resolved}  ({size_mb} MB)")

    classes = checkpoint_classes(checkpoint, models_dir=models_dir)
    print(f"  classes     {len(classes)}")

    if args.no_export:
        print("\n--no-export: the Torch runtime can use this checkpoint now.")
        print(f"Point a camera at it with:  models.detector.name: {name}")
        return 0

    layout = "rtdetr" if is_rtdetr(checkpoint) else "yolov8"
    destination = models_dir / "detector" / f"{Path(checkpoint).stem}-{args.version}.onnx"
    print(f"exporting ONNX (imgsz={args.imgsz}, opset={args.opset}) …")
    export_onnx(
        checkpoint, destination,
        imgsz=args.imgsz, opset=args.opset, models_dir=models_dir,
    )

    metadata = {
        "file": str(destination.relative_to(models_dir)),
        "sha256": sha256_file(destination),
        "layout": layout,
        "runtime": "onnx",
        "input_size": [args.imgsz, args.imgsz],
        "classes": classes,
        "size_mb": round(destination.stat().st_size / 1024**2, 2),
        "provenance": {
            "exported_from": Path(checkpoint).name,
            "exporter": "ultralytics",
            "opset": args.opset,
        },
    }
    register_model(
        args.registry, f"{name}-onnx", args.version, metadata,
        role="detector", make_default=True,
    )
    print(f"  artefact    {destination}  ({metadata['size_mb']} MB)")
    print(f"  sha256      {metadata['sha256']}")
    print(f"\nregistered {name}-onnx:{args.version} in {args.registry}")
    print(
        "\nThis artefact runs on ONNX Runtime alone - a post needs no Torch. "
        "Evaluate it on labelled footage from a representative site before "
        "commissioning it.",
        file=sys.stderr,
    )
    return 0


def _models_register(args: Any) -> int:
    """Register an already-exported ONNX artefact.

    Producing the .onnx is left to whatever framework trained the model -
    `yolo export`, `torch.onnx.export`, or a vendor toolchain. This command
    does the part that belongs to the platform: hash the artefact, record its
    layout and class list, and refuse to overwrite a version that is already
    published under a different checksum.
    """
    from ibvap.mlops.registry import register_model, sha256_file, write_model_card

    artefact = Path(args.artefact)
    if not artefact.is_file():
        print(f"artefact not found: {artefact}", file=sys.stderr)
        return 1

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    metadata = {
        "file": str(artefact),
        "sha256": sha256_file(artefact),
        "layout": args.layout,
        "input_size": [args.imgsz, args.imgsz],
        "classes": classes,
        "size_mb": round(artefact.stat().st_size / 1024**2, 2),
        "provenance": {"registered_from": artefact.name},
    }
    register_model(
        args.registry, args.name, args.version, metadata,
        role=args.role, make_default=True,
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "classes"}, indent=2))
    print(f"\nregistered {args.name}:{args.version} in {args.registry}")

    if args.card:
        write_model_card(args.card, args.name, args.version, metadata)
        print(f"model card written to {args.card}")
    print(
        "\nEvaluate accuracy on labelled data from a representative site before "
        "commissioning this version.",
        file=sys.stderr,
    )
    return 0



def _evidence_verify(args: Any) -> int:
    from ibvap.events.evidence import EvidenceStore

    settings = _load(args)
    store = EvidenceStore(settings.evidence, settings.privacy, site_id=settings.site_id)
    report = store.verify_chain(args.day)

    print(f"{report['day']}: {report['manifests']} manifest(s)")
    if report["valid"]:
        print("✓ chain intact — no evidence has been altered or removed")
        return 0
    print(f"✗ {len(report['issues'])} issue(s):")
    for issue in report["issues"]:
        print(f"  • {issue}")
    return 1


def _evidence_prune(args: Any) -> int:
    from ibvap.events.evidence import EvidenceStore

    settings = _load(args)
    store = EvidenceStore(settings.evidence, settings.privacy, site_id=settings.site_id)
    before = store.usage()
    result = store.prune()
    after = store.usage()
    print(
        f"removed {result['files']} artefact(s), freed "
        f"{result['bytes'] / 1024**2:.1f} MB\n"
        f"  before  {before['megabytes']} MB in {before['files']} files\n"
        f"  after   {after['megabytes']} MB in {after['files']} files"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
