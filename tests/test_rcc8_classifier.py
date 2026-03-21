"""Unit tests for the RCC-8 classifier covering all 8 relations and edge cases."""

import math
import pytest
from shapely.geometry import box, Polygon, MultiPolygon
from rcc_drive.rcc8_classifier import classify_rcc8


class TestAllEightRelations:
    """Test each of the 8 RCC-8 relations with synthetic polygons."""

    def test_dc_disjoint(self):
        a = box(0, 0, 1, 1)
        b = box(5, 5, 6, 6)
        assert classify_rcc8(a, b) == "DC"

    def test_ec_touching_edge(self):
        a = box(0, 0, 1, 1)
        b = box(1, 0, 2, 1)
        assert classify_rcc8(a, b) == "EC"

    def test_ec_touching_corner(self):
        a = box(0, 0, 1, 1)
        b = box(1, 1, 2, 2)
        assert classify_rcc8(a, b) == "EC"

    def test_po_partial_overlap(self):
        a = box(0, 0, 2, 2)
        b = box(1, 1, 3, 3)
        assert classify_rcc8(a, b) == "PO"

    def test_eq_equal(self):
        a = box(0, 0, 1, 1)
        b = box(0, 0, 1, 1)
        assert classify_rcc8(a, b) == "EQ"

    def test_tpp_tangential_proper_part(self):
        # A shares boundary with B at x=0 and y=0 edges
        a = box(0, 0, 1, 1)
        b = box(0, 0, 2, 2)
        assert classify_rcc8(a, b) == "TPP"

    def test_ntpp_non_tangential_proper_part(self):
        # A fully inside B, no shared boundary
        a = box(0.5, 0.5, 1.5, 1.5)
        b = box(0, 0, 2, 2)
        assert classify_rcc8(a, b) == "NTPP"

    def test_tppi_inverse(self):
        # B is TPP of A → A is TPPi of B
        a = box(0, 0, 2, 2)
        b = box(0, 0, 1, 1)
        assert classify_rcc8(a, b) == "TPPi"

    def test_ntppi_inverse(self):
        # B fully inside A, no shared boundary
        a = box(0, 0, 2, 2)
        b = box(0.5, 0.5, 1.5, 1.5)
        assert classify_rcc8(a, b) == "NTPPi"


class TestSymmetry:
    """Verify that swapping A and B produces the correct inverse relation."""

    def test_dc_symmetric(self):
        a = box(0, 0, 1, 1)
        b = box(5, 5, 6, 6)
        assert classify_rcc8(a, b) == "DC"
        assert classify_rcc8(b, a) == "DC"

    def test_ec_symmetric(self):
        a = box(0, 0, 1, 1)
        b = box(1, 0, 2, 1)
        assert classify_rcc8(a, b) == "EC"
        assert classify_rcc8(b, a) == "EC"

    def test_po_symmetric(self):
        a = box(0, 0, 2, 2)
        b = box(1, 1, 3, 3)
        assert classify_rcc8(a, b) == "PO"
        assert classify_rcc8(b, a) == "PO"

    def test_tpp_ntpp_inversion(self):
        a = box(0, 0, 1, 1)
        b = box(0, 0, 2, 2)
        assert classify_rcc8(a, b) == "TPP"
        assert classify_rcc8(b, a) == "TPPi"

    def test_ntpp_ntppi_inversion(self):
        a = box(0.5, 0.5, 1.5, 1.5)
        b = box(0, 0, 2, 2)
        assert classify_rcc8(a, b) == "NTPP"
        assert classify_rcc8(b, a) == "NTPPi"


class TestEdgeCases:
    """Test edge cases: degenerate, rotated, near-touching, invalid input."""

    def test_degenerate_zero_area(self):
        # Line segment (zero area)
        a = Polygon([(0, 0), (1, 0), (1, 0), (0, 0)])
        b = box(5, 5, 6, 6)
        assert classify_rcc8(a, b) == "DC"

    def test_near_touching_small_gap(self):
        # 0.001m gap — should be DC, not EC
        a = box(0, 0, 1, 1)
        b = box(1.001, 0, 2, 1)
        assert classify_rcc8(a, b) == "DC"

    def test_rotated_polygons(self):
        # Diamond (45-degree rotated square) fully inside a larger box
        diamond = Polygon([(1, 0), (2, 1), (1, 2), (0, 1)])
        outer = box(-1, -1, 3, 3)
        assert classify_rcc8(diamond, outer) == "NTPP"

    def test_rotated_polygon_tppi(self):
        # Box fully inside diamond, boundaries touch at corners
        diamond = Polygon([(1, 0), (2, 1), (1, 2), (0, 1)])
        inner = box(0.5, 0.5, 1.5, 1.5)
        assert classify_rcc8(diamond, inner) == "TPPi"

    def test_tppi_rotated_explicit(self):
        # Diamond contains box; box corners touch diamond edges exactly
        diamond = Polygon([(1, 0), (2, 1), (1, 2), (0, 1)])
        inner = box(0.5, 0.5, 1.5, 1.5)
        assert classify_rcc8(diamond, inner) == "TPPi"
        # Note: corners of inner land exactly on diamond boundary edges by construction

    def test_rotated_polygon_po(self):
        # Diamond partially overlapping a box (neither contains the other)
        diamond = Polygon([(1, 0), (2, 1), (1, 2), (0, 1)])
        partial = box(0.5, 0.5, 2.5, 1.5)
        assert classify_rcc8(diamond, partial) == "PO"

    def test_self_intersecting_polygon(self):
        # Bowtie shape (self-intersecting) — buffer(0) should fix it
        bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        normal = box(5, 5, 6, 6)
        result = classify_rcc8(bowtie, normal)
        assert result == "DC"

    def test_very_small_polygon(self):
        tiny = box(0, 0, 1e-5, 1e-5)
        normal = box(0, 0, 1, 1)
        # Tiny polygon area = 1e-10, below min_polygon_area threshold
        assert classify_rcc8(tiny, normal) == "DC"

    def test_large_containment(self):
        # Very large polygon containing a small one
        small = box(100, 100, 101, 101)
        large = box(0, 0, 200, 200)
        assert classify_rcc8(small, large) == "NTPP"

    def test_identical_rotated_squares(self):
        # Two identical rotated squares
        sq = Polygon([(1, 0), (2, 1), (1, 2), (0, 1)])
        assert classify_rcc8(sq, sq) == "EQ"


class TestDrivingScenarios:
    """Test with realistic driving-scale polygons (meters)."""

    def test_ego_pedestrian_dc(self):
        # Ego vehicle at origin facing +x, pedestrian 10m away
        ego = box(-2.042, -0.865, 2.042, 0.865)  # ~4m x 1.7m
        ped = box(9.8, 2.0, 10.2, 2.4)  # small pedestrian footprint
        assert classify_rcc8(ego, ped) == "DC"

    def test_pedestrian_on_crosswalk(self):
        # Pedestrian fully inside a large crosswalk
        ped = box(5.0, 1.0, 5.4, 1.4)
        crosswalk = box(3.0, 0.0, 8.0, 4.0)
        assert classify_rcc8(ped, crosswalk) == "NTPP"

    def test_pedestrian_stepping_off_crosswalk(self):
        # Pedestrian partially on crosswalk
        ped = box(7.8, 1.0, 8.2, 1.4)
        crosswalk = box(3.0, 0.0, 8.0, 4.0)
        assert classify_rcc8(ped, crosswalk) == "PO"

    def test_vehicles_side_by_side(self):
        # Two cars in adjacent lanes, not touching
        car_a = box(0, 0, 4.5, 1.8)
        car_b = box(0, 2.5, 4.5, 4.3)
        assert classify_rcc8(car_a, car_b) == "DC"
