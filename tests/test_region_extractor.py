"""Unit tests for the region extractor (ego footprint and filtering)."""

import math
import pytest
import numpy as np
from rcc_drive.region_extractor import RegionExtractor


class TestEgoFootprint:
    """Test ego vehicle polygon construction."""

    def setup_method(self):
        self.extractor = RegionExtractor()

    def _make_ego_pose(self, x, y, yaw_deg=0):
        """Helper to create an ego_pose dict with given position and yaw."""
        from pyquaternion import Quaternion
        yaw_rad = math.radians(yaw_deg)
        # Create quaternion from yaw rotation around z-axis
        q = Quaternion(axis=[0, 0, 1], angle=yaw_rad)
        return {
            "translation": [x, y, 0.0],
            "rotation": list(q),  # [w, x, y, z]
        }

    def test_area_correct(self):
        pose = self._make_ego_pose(0, 0, 0)
        poly = self.extractor.ego_footprint(pose)
        expected_area = 4.084 * 1.730
        assert abs(poly.area - expected_area) < 1e-6

    def test_centered_at_origin(self):
        pose = self._make_ego_pose(0, 0, 0)
        poly = self.extractor.ego_footprint(pose)
        cx, cy = poly.centroid.x, poly.centroid.y
        assert abs(cx) < 1e-6
        assert abs(cy) < 1e-6

    def test_translated(self):
        pose = self._make_ego_pose(100, 200, 0)
        poly = self.extractor.ego_footprint(pose)
        cx, cy = poly.centroid.x, poly.centroid.y
        assert abs(cx - 100) < 1e-6
        assert abs(cy - 200) < 1e-6

    def test_rotated_90_degrees(self):
        pose_0 = self._make_ego_pose(0, 0, 0)
        pose_90 = self._make_ego_pose(0, 0, 90)
        poly_0 = self.extractor.ego_footprint(pose_0)
        poly_90 = self.extractor.ego_footprint(pose_90)

        # After 90-degree rotation, the bounding boxes should swap dimensions
        b0 = poly_0.bounds  # (minx, miny, maxx, maxy)
        b90 = poly_90.bounds

        width_0 = b0[2] - b0[0]   # should be ego_length
        height_0 = b0[3] - b0[1]  # should be ego_width

        width_90 = b90[2] - b90[0]   # should be ego_width
        height_90 = b90[3] - b90[1]  # should be ego_length

        assert abs(width_0 - 4.084) < 1e-3
        assert abs(height_0 - 1.730) < 1e-3
        assert abs(width_90 - 1.730) < 1e-3
        assert abs(height_90 - 4.084) < 1e-3

    def test_area_preserved_after_rotation(self):
        pose = self._make_ego_pose(50, 50, 45)
        poly = self.extractor.ego_footprint(pose)
        expected_area = 4.084 * 1.730
        assert abs(poly.area - expected_area) < 1e-6

    def test_buffer_increases_area(self):
        from rcc_drive.config import DEFAULT_CONFIG
        config = {**DEFAULT_CONFIG, "ego_buffer": 0.5}
        extractor = RegionExtractor(config=config)
        pose = self._make_ego_pose(0, 0, 0)
        poly = extractor.ego_footprint(pose)
        base_area = 4.084 * 1.730
        assert poly.area > base_area


class TestFilterSafetyCritical:
    """Test annotation category filtering."""

    def setup_method(self):
        self.extractor = RegionExtractor()

    def _make_ann(self, category):
        return {
            "token": f"tok_{category}",
            "category_name": category,
            "translation": [0, 0, 0],
            "size": [1, 1, 1],
            "rotation": [1, 0, 0, 0],
        }

    def test_pedestrian_categories(self):
        anns = [
            self._make_ann("human.pedestrian.adult"),
            self._make_ann("human.pedestrian.child"),
            self._make_ann("human.pedestrian.construction_worker"),
        ]
        result = self.extractor.filter_safety_critical(anns)
        assert len(result["pedestrians"]) == 3
        assert len(result["vehicles"]) == 0

    def test_vehicle_categories(self):
        anns = [
            self._make_ann("vehicle.car"),
            self._make_ann("vehicle.truck"),
            self._make_ann("vehicle.bus.rigid"),
        ]
        result = self.extractor.filter_safety_critical(anns)
        assert len(result["pedestrians"]) == 0
        assert len(result["vehicles"]) == 3

    def test_non_safety_critical_filtered(self):
        anns = [
            self._make_ann("movable_object.barrier"),
            self._make_ann("static_object.bicycle_rack"),
            self._make_ann("human.pedestrian.adult"),
        ]
        result = self.extractor.filter_safety_critical(anns)
        assert len(result["pedestrians"]) == 1
        assert len(result["vehicles"]) == 0

    def test_empty_annotations(self):
        result = self.extractor.filter_safety_critical([])
        assert len(result["pedestrians"]) == 0
        assert len(result["vehicles"]) == 0


import importlib
_HAS_NUSCENES = importlib.util.find_spec("nuscenes") is not None


@pytest.mark.skipif(not _HAS_NUSCENES, reason="nuscenes-devkit not installed")
class TestAnnotationFootprint:
    """Tests for annotation_footprint — requires nuscenes-devkit."""

    def setup_method(self):
        self.extractor = RegionExtractor()

    def test_basic_area_and_centroid(self):
        # size is [width, length, height]; footprint area = width * length = 1 * 2 = 2
        ann = {
            "translation": [10.0, 20.0, 0.0],
            "size": [1.0, 2.0, 1.5],
            "rotation": [1, 0, 0, 0],  # identity quaternion [w, x, y, z]
        }
        poly = self.extractor.annotation_footprint(ann)
        assert abs(poly.area - 2.0) < 1e-3
        assert abs(poly.centroid.x - 10.0) < 1e-3
        assert abs(poly.centroid.y - 20.0) < 1e-3

    def test_at_origin(self):
        ann = {
            "translation": [0.0, 0.0, 0.0],
            "size": [2.0, 4.0, 1.8],
            "rotation": [1, 0, 0, 0],
        }
        poly = self.extractor.annotation_footprint(ann)
        assert abs(poly.area - 8.0) < 1e-3
        assert abs(poly.centroid.x) < 1e-3
        assert abs(poly.centroid.y) < 1e-3

    def test_rotated_preserves_area(self):
        # 90-degree yaw rotation should preserve area
        import math
        from pyquaternion import Quaternion
        q = Quaternion(axis=[0, 0, 1], angle=math.pi / 2)
        ann = {
            "translation": [5.0, 5.0, 0.0],
            "size": [1.5, 3.0, 1.8],
            "rotation": list(q),
        }
        poly = self.extractor.annotation_footprint(ann)
        assert abs(poly.area - 4.5) < 1e-3
        assert abs(poly.centroid.x - 5.0) < 1e-3
        assert abs(poly.centroid.y - 5.0) < 1e-3

    def test_returns_valid_polygon(self):
        ann = {
            "translation": [100.0, 200.0, 0.5],
            "size": [2.0, 4.5, 1.9],
            "rotation": [1, 0, 0, 0],
        }
        poly = self.extractor.annotation_footprint(ann)
        assert poly.is_valid
        assert not poly.is_empty
        assert poly.area > 0
