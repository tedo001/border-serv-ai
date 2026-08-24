"""Zone, tripwire and polygon geometry."""

from __future__ import annotations

import numpy as np
import pytest

from ibvap.core.geometry import (
    CrossingDirection,
    Tripwire,
    Zone,
    point_in_polygon,
    points_in_polygon,
    polygon_area,
    polygon_centroid,
    segments_intersect,
    side_of_line,
)

SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)


class TestPolygons:
    def test_containment(self) -> None:
        assert point_in_polygon(5, 5, SQUARE)
        assert not point_in_polygon(15, 5, SQUARE)
        assert not point_in_polygon(-1, 5, SQUARE)

    def test_vectorised_matches_scalar(self) -> None:
        rng = np.random.default_rng(0)
        points = rng.uniform(-5, 15, (200, 2))
        vector = points_in_polygon(points, SQUARE)
        scalar = np.array([point_in_polygon(x, y, SQUARE) for x, y in points])
        assert np.array_equal(vector, scalar)

    def test_concave_polygon(self) -> None:
        """A C-shape must exclude the notch, which convex-hull logic would not."""
        c_shape = np.array([[0, 0], [10, 0], [10, 3], [3, 3], [3, 7], [10, 7], [10, 10], [0, 10]], float)
        assert point_in_polygon(1, 5, c_shape)
        assert not point_in_polygon(7, 5, c_shape)  # inside the notch

    def test_area_and_centroid(self) -> None:
        assert polygon_area(SQUARE) == pytest.approx(100.0)
        assert polygon_centroid(SQUARE) == pytest.approx((5.0, 5.0))

    def test_degenerate_inputs(self) -> None:
        assert polygon_area(np.array([[0, 0], [1, 1]], float)) == 0.0
        assert not point_in_polygon(0, 0, np.array([[0, 0]], float))
        assert points_in_polygon(np.empty((0, 2)), SQUARE).shape == (0,)


class TestZone:
    def test_normalised_coordinates_survive_resolution_change(self) -> None:
        """The same zone must cover the same scene fraction at any resolution.

        This is the property that lets a camera be re-profiled (1080p by day,
        720p sub-stream at night) without invalidating operator-drawn zones.
        """
        zone = Zone("z", "Fence", [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])
        for width, height in ((1920, 1080), (1280, 720), (640, 360)):
            assert zone.contains(width * 0.75, height * 0.75, width, height)
            assert not zone.contains(width * 0.25, height * 0.25, width, height)

    def test_requires_three_points(self) -> None:
        with pytest.raises(ValueError, match="at least 3 points"):
            Zone("z", "bad", [(0.0, 0.0), (1.0, 1.0)])


class TestTripwire:
    def test_crossing_direction(self) -> None:
        wire = Tripwire("w", "Line", (0.5, 0.0), (0.5, 1.0))
        assert wire.crossing((400, 500), (600, 500), 1000, 1000) is CrossingDirection.RIGHT
        assert wire.crossing((600, 500), (400, 500), 1000, 1000) is CrossingDirection.LEFT
        assert wire.crossing((100, 500), (200, 500), 1000, 1000) is CrossingDirection.NONE

    def test_movement_beyond_the_wire_extent_does_not_fire(self) -> None:
        """The classic false-alarm bug: treating a segment as an infinite line.

        A vehicle crossing the *extension* of a short fence-line wire - a road
        200 m past its end - must not register as a crossing.
        """
        short_wire = Tripwire("w", "Short", (0.5, 0.0), (0.5, 0.2))
        assert short_wire.crossing((400, 500), (600, 500), 1000, 1000) is CrossingDirection.NONE

    def test_direction_filter(self) -> None:
        wire = Tripwire("w", "Line", (0.5, 0.0), (0.5, 1.0), direction=CrossingDirection.RIGHT)
        assert wire.matches_direction(CrossingDirection.RIGHT)
        assert not wire.matches_direction(CrossingDirection.LEFT)
        assert not wire.matches_direction(CrossingDirection.NONE)

    def test_labels(self) -> None:
        wire = Tripwire(
            "w", "Line", (0.5, 0.0), (0.5, 1.0),
            left_label="exfiltration", right_label="infiltration",
        )
        assert wire.label_for(CrossingDirection.RIGHT) == "infiltration"
        assert wire.label_for(CrossingDirection.LEFT) == "exfiltration"

    def test_rejects_zero_length(self) -> None:
        from ibvap.core.config import TripwireConfig

        with pytest.raises(ValueError, match="must differ"):
            TripwireConfig(id="w", start=(0.5, 0.5), end=(0.5, 0.5))


class TestSegments:
    def test_intersection(self) -> None:
        assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
        assert not segments_intersect((0, 0), (1, 1), (5, 5), (6, 6))

    def test_side_of_line_separates_half_planes(self) -> None:
        """Only the sign matters, and it must differ across the line."""
        above = side_of_line((0, 0), (10, 0), (5, -5))
        below = side_of_line((0, 0), (10, 0), (5, 5))
        assert above * below < 0, "points on opposite sides must have opposite signs"
        assert side_of_line((0, 0), (10, 0), (5, 0)) == 0


class TestDirectionMatchesRenderedArrow:
    """The configured direction must fire for the motion the console depicts.

    An operator sets `direction: left` after looking at the arrow drawn over
    their fence line. If the two disagree, every crossing alert on that wire is
    inverted - and nothing in the system would reveal it.
    """

    @pytest.mark.parametrize(
        ("start", "end"),
        [((0.5, 0.0), (0.5, 1.0)), ((0.0, 0.5), (1.0, 0.5)), ((0.2, 0.1), (0.8, 0.9))],
    )
    def test_left_matches_arrow(self, start, end) -> None:
        import math

        size = 1000
        wire = Tripwire("w", "wire", start, end)
        (ax, ay), (bx, by) = wire.to_pixels(size, size)
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        # The left-hand normal, exactly as events.annotate.draw_tripwire uses.
        nx, ny = -dy / length, dx / length
        mid = ((ax + bx) / 2, (ay + by) / 2)

        before = (mid[0] - nx * 80, mid[1] - ny * 80)
        after = (mid[0] + nx * 80, mid[1] + ny * 80)
        assert wire.crossing(before, after, size, size) is CrossingDirection.LEFT
        assert wire.crossing(after, before, size, size) is CrossingDirection.RIGHT
