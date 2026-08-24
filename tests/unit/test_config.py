"""Configuration loading, validation and the object taxonomy."""

from __future__ import annotations

from pathlib import Path

import pytest

from ibvap.core.config import (
    CameraConfig,
    RuleConfig,
    Settings,
    ZoneConfig,
    load_settings,
)
from ibvap.core.errors import ConfigError
from ibvap.core.timeutils import humanise_duration, in_any_window, parse_window
from ibvap.core.types import ObjectCategory, ObjectClass, Severity


class TestTaxonomy:
    def test_canonical_values_round_trip(self) -> None:
        """Without this, a config value like "animal" resolves to UNKNOWN and
        every class-based filter silently stops matching."""
        for member in ObjectClass:
            assert ObjectClass.coerce(member.value) is member

    def test_foreign_labels_map_onto_the_taxonomy(self) -> None:
        assert ObjectClass.coerce("cow") is ObjectClass.ANIMAL
        assert ObjectClass.coerce("lorry") is ObjectClass.TRUCK
        assert ObjectClass.coerce("backpack") is ObjectClass.BAG

    def test_unknown_labels_degrade_safely(self) -> None:
        """Swapping in a model with a different class list must not crash a
        running pipeline."""
        assert ObjectClass.coerce("wombat") is ObjectClass.UNKNOWN

    def test_categories(self) -> None:
        assert ObjectClass.PERSON.category is ObjectCategory.HUMAN
        assert ObjectClass.TRUCK.category is ObjectCategory.VEHICLE
        assert ObjectClass.ANIMAL.category is ObjectCategory.ANIMAL

    def test_severity_ordering(self) -> None:
        assert Severity.INFO < Severity.LOW < Severity.MEDIUM
        assert Severity.HIGH < Severity.CRITICAL


class TestTimeWindows:
    def test_midnight_wrapping(self) -> None:
        """A curfew window is the common case, and the one naive
        implementations get wrong."""
        import datetime

        window = parse_window("18:00-06:00")
        assert window.wraps_midnight
        assert window.contains(datetime.time(23, 0))
        assert window.contains(datetime.time(3, 0))
        assert not window.contains(datetime.time(12, 0))

    def test_windows_partition_the_day(self) -> None:
        import datetime

        day = parse_window("06:00-18:00")
        night = parse_window("18:00-06:00")
        for hour in range(24):
            moment = datetime.time(hour, 0)
            assert day.contains(moment) != night.contains(moment)

    def test_empty_schedule_is_always_active(self) -> None:
        assert in_any_window([])

    def test_invalid_window_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_window("25:00-26:00")
        with pytest.raises(ValueError):
            parse_window("not a window")

    def test_duration_formatting(self) -> None:
        assert humanise_duration(45) == "45s"
        assert humanise_duration(125) == "2m05s"
        assert humanise_duration(7900) == "2h11m"


class TestCameraValidation:
    def test_rule_referencing_an_unknown_zone_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown zone"):
            CameraConfig(
                id="cam", url="rtsp://x",
                rules=[RuleConfig(id="r", type="intrusion", zones=["nope"])],
            )

    def test_rule_referencing_an_unknown_tripwire_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown tripwire"):
            CameraConfig(
                id="cam", url="rtsp://x",
                rules=[RuleConfig(id="r", type="line_crossing", tripwires=["nope"])],
            )

    def test_duplicate_zone_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate zone"):
            CameraConfig(
                id="cam", url="rtsp://x",
                zones=[
                    ZoneConfig(id="z", points=[(0, 0), (1, 0), (1, 1)]),
                    ZoneConfig(id="z", points=[(0, 0), (1, 0), (0, 1)]),
                ],
            )

    def test_zone_points_must_be_normalised(self) -> None:
        """Pixels here would silently mean something different on every camera."""
        with pytest.raises(ValueError, match="normalised"):
            ZoneConfig(id="z", points=[(0, 0), (1920, 0), (1920, 1080)])

    def test_name_defaults_to_id(self) -> None:
        camera = CameraConfig(id="cam-north", url="rtsp://x")
        assert camera.name == "cam-north"

    def test_duplicate_camera_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate camera"):
            Settings(cameras=[
                CameraConfig(id="cam", url="rtsp://a"),
                CameraConfig(id="cam", url="rtsp://b"),
            ])


class TestSettingsLoading:
    def test_yaml_round_trip(self, workspace: Path) -> None:
        config = workspace / "site.yaml"
        config.write_text(
            "site_id: bop-test\n"
            "site_name: Test Post\n"
            "cameras:\n"
            "  - id: cam-1\n"
            "    url: rtsp://10.0.0.1/stream\n"
            "    target_fps: 6\n"
        )
        settings = load_settings(config)
        assert settings.site_id == "bop-test"
        assert settings.cameras[0].target_fps == 6

    def test_include_merges_shared_files(self, workspace: Path) -> None:
        (workspace / "base.yaml").write_text("site_name: Shared Base\ntier: edge\n")
        config = workspace / "site.yaml"
        config.write_text("include: [base.yaml]\nsite_id: bop-test\n")

        settings = load_settings(config)
        assert settings.site_name == "Shared Base"
        assert settings.site_id == "bop-test"

    def test_overrides_win(self, workspace: Path) -> None:
        config = workspace / "site.yaml"
        config.write_text("site_id: from-file\n")
        assert load_settings(config, overrides={"site_id": "from-override"}).site_id == "from-override"

    def test_missing_file_is_an_error(self) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_settings("/nonexistent/site.yaml")

    def test_invalid_yaml_is_an_error(self, workspace: Path) -> None:
        config = workspace / "bad.yaml"
        config.write_text("cameras: [unclosed\n")
        with pytest.raises(ConfigError, match="invalid YAML"):
            load_settings(config)

    def test_invalid_configuration_is_an_error(self, workspace: Path) -> None:
        config = workspace / "bad.yaml"
        config.write_text("cameras:\n  - id: cam\n    url: rtsp://x\n    target_fps: -5\n")
        with pytest.raises(ConfigError, match="invalid configuration"):
            load_settings(config)

    def test_camera_lookup(self, settings: Settings) -> None:
        assert settings.camera("cam-test").id == "cam-test"
        with pytest.raises(ConfigError, match="unknown camera"):
            settings.camera("nope")

    def test_generated_secret_is_long_enough(self) -> None:
        """Anything shorter is rejected by TokenService, so the generator and
        the validator must agree."""
        assert len(Settings().ensure_secret().encode()) >= 32
