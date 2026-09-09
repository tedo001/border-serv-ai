#!/usr/bin/env python3
"""Run the test suite the way CI does: only the `dev` extra, nothing optional.

Two CI failures in a row came from tests that passed locally because a
developer machine had `torch`, `supervision` or `PyQt6` installed and CI does
not. The tests were not wrong about the code; they were reading a different
environment. This blocks every optional package at the import system, so the
suite here sees exactly what a CI runner sees.

    python scripts/ci_like.py tests/unit -q
    python scripts/ci_like.py tests/integration -q

Run it before pushing anything that touches an optional dependency.
"""

import sys

#: Everything outside `pip install -e ".[dev]"`, which is all CI installs.
BLOCKED = {
    "torch", "torchvision", "ultralytics", "ultralytics_platform",
    "supervision", "PyQt6", "tensorrt", "onnx", "onnxslim",
}


class _Blocker:
    """Refuses the optional packages, exactly as an absent module would."""

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in BLOCKED:
            # The error has to look like a genuinely missing module or
            # pytest.importorskip re-raises it instead of skipping.
            raise ModuleNotFoundError(f"No module named {root!r}", name=root)
        # None means "not my business"; the real finders take it from here.
        return None  # noqa: RET501


def main(argv: list[str]) -> int:
    for imported in list(sys.modules):
        if imported.split(".")[0] in BLOCKED:
            del sys.modules[imported]
    sys.meta_path.insert(0, _Blocker())

    import pytest

    return pytest.main(argv or ["tests/unit", "-q"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
