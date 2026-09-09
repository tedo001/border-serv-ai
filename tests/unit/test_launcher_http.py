"""What the control panel is allowed to fetch.

The panel polls a node built from ``IBVAP_PORT``, an environment variable it
does not control. ``urllib.request.urlopen`` honours whatever scheme the URL
carries, so a ``file://`` string reaching it reads a local file and hands the
contents back as if the node had served them. These pin the restriction down.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6", reason="the desktop extra is not installed")

from ibvap.desktop.launcher import (  # noqa: E402
    ALLOWED_SCHEMES,
    _get_json,
    _http_opener,
)


class TestSchemeRestriction:
    @pytest.mark.parametrize("scheme", ["file", "ftp", "data", "gopher", "jar"])
    def test_only_http_is_allowed(self, scheme: str) -> None:
        assert scheme not in ALLOWED_SCHEMES

    def test_http_and_https_are_allowed(self) -> None:
        assert sorted(ALLOWED_SCHEMES) == ["http", "https"]

    def test_a_local_file_is_never_read(self, workspace) -> None:
        """The failure this guards against, as an assertion.

        A ``file://`` URL reaching urlopen returns the file's contents, and the
        caller cannot tell them from a node's reply.
        """
        secret = workspace / "secret.json"
        secret.write_text(json.dumps({"token": "should-never-be-read"}))

        assert _get_json(f"file://{secret}") is None

    @pytest.mark.parametrize(
        "url",
        ["ftp://example.invalid/x", "data:application/json,{}", "gopher://x/1"],
    )
    def test_other_schemes_return_nothing_rather_than_raising(self, url: str) -> None:
        """A refusal must not crash the poll thread it runs on."""
        assert _get_json(url) is None

    def test_an_unreachable_node_is_indistinguishable_from_a_stopped_one(self) -> None:
        # Port 9 is discard; nothing listens, which is the state the panel
        # spends most of its life in.
        assert _get_json("http://127.0.0.1:9/health", timeout=0.5) is None


class TestOpener:
    def test_the_opener_carries_no_file_or_ftp_handler(self) -> None:
        """`build_opener` adds the default handlers on top of what it is given.

        Restricting the opener that way passes a scanner and changes nothing,
        so the director is built bare and the handlers added by hand.
        """
        import urllib.request

        handlers = _http_opener().handlers
        assert not any(isinstance(h, urllib.request.FileHandler) for h in handlers)
        assert not any(isinstance(h, urllib.request.FTPHandler) for h in handlers)
        assert any(isinstance(h, urllib.request.HTTPHandler) for h in handlers)

    def test_an_unhandled_scheme_raises_rather_than_returning_none(self) -> None:
        """Without an UnknownHandler the director returns None.

        The caller then calls __enter__ on it and the poll thread dies with a
        TypeError instead of quietly reporting the node as unreachable.
        """
        import urllib.error
        import urllib.request

        request = urllib.request.Request("gopher://example.invalid/1")
        with pytest.raises(urllib.error.URLError):
            _http_opener().open(request, timeout=0.5)
