"""Convert nuScenes annotations and ego pose into Shapely ground-plane polygons."""

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Polygon

from rcc_drive.config import DEFAULT_CONFIG, PEDESTRIAN_CATEGORIES, VEHICLE_CATEGORIES


class RegionExtractor:
    """Extracts BEV ground-plane polygons from nuScenes annotations."""

    def __init__(self, config=None):
        cfg = config or DEFAULT_CONFIG
        self.ego_length = cfg["ego_length"]
        self.ego_width = cfg["ego_width"]
        self.ego_buffer = cfg["ego_buffer"]
        self.max_distance = cfg["max_object_distance"]

    def ego_footprint(self, ego_pose: dict) -> Polygon:
        """Build a rotated rectangle for the ego vehicle on the ground plane.

        Args:
            ego_pose: Dict with 'translation' [x, y, z] and 'rotation' [w, x, y, z].

        Returns:
            Shapely Polygon representing the ego vehicle footprint.
        """
        # nuScenes quaternion is [w, x, y, z] — pyquaternion expects same order
        q = Quaternion(ego_pose["rotation"])
        yaw = q.yaw_pitch_roll[0]

        half_l = self.ego_length / 2
        half_w = self.ego_width / 2

        # Corners in local frame (front-left, front-right, rear-right, rear-left)
        local_corners = np.array([
            [half_l, -half_w],
            [half_l,  half_w],
            [-half_l,  half_w],
            [-half_l, -half_w],
        ])

        # Rotate by yaw
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        rotated = (rotation @ local_corners.T).T

        # Translate to ego position (x, y only)
        tx, ty = ego_pose["translation"][0], ego_pose["translation"][1]
        global_corners = rotated + np.array([tx, ty])

        poly = Polygon(global_corners)

        if self.ego_buffer > 0:
            poly = poly.buffer(self.ego_buffer)

        return poly

    def annotation_footprint(self, annotation: dict) -> Polygon:
        """Build a ground-plane polygon from a nuScenes sample_annotation.

        nuScenes sample_annotation stores translation and rotation in global
        coordinates. We construct a Box and extract its bottom corners.

        Args:
            annotation: nuScenes sample_annotation record with 'translation',
                       'size' [w, l, h], and 'rotation' [w, x, y, z].

        Returns:
            Shapely Polygon representing the object's ground footprint.
        """
        from nuscenes.utils.data_classes import Box

        # nuScenes size convention: [width, length, height]
        # Box footprint area = size[0] * size[1] (width * length)
        box = Box(
            annotation["translation"],
            annotation["size"],
            Quaternion(annotation["rotation"]),
        )

        # bottom_corners() returns shape (3, 4): rows are x, y, z; cols are corners
        corners = box.bottom_corners()
        xy_pairs = [(corners[0, i], corners[1, i]) for i in range(4)]

        return Polygon(xy_pairs)

    def filter_safety_critical(self, annotations: list) -> dict:
        """Filter annotations into safety-critical categories.

        Args:
            annotations: List of nuScenes sample_annotation records.

        Returns:
            Dict with 'pedestrians' and 'vehicles' lists.
        """
        pedestrians = []
        vehicles = []

        for ann in annotations:
            cat = ann["category_name"]
            if any(cat.startswith(prefix) for prefix in PEDESTRIAN_CATEGORIES):
                pedestrians.append(ann)
            elif any(cat.startswith(prefix) for prefix in VEHICLE_CATEGORIES):
                vehicles.append(ann)

        return {"pedestrians": pedestrians, "vehicles": vehicles}

    def extract_all_regions(
        self, ego_pose: dict, annotations: list, crosswalks: list
    ) -> dict:
        """Extract all ground-plane polygons for a sample.

        Args:
            ego_pose: Ego pose record.
            annotations: List of sample_annotation records.
            crosswalks: List of dicts with 'token' and 'polygon' keys.

        Returns:
            Dict with 'ego' (Polygon), 'pedestrians', 'vehicles', 'crosswalks' lists.
            Each list item has 'token', 'category' (if applicable), and 'polygon'.
        """
        ego_poly = self.ego_footprint(ego_pose)
        ego_x, ego_y = ego_pose["translation"][0], ego_pose["translation"][1]

        filtered = self.filter_safety_critical(annotations)

        def _within_range(ann):
            ax, ay = ann["translation"][0], ann["translation"][1]
            dist = np.sqrt((ax - ego_x) ** 2 + (ay - ego_y) ** 2)
            return dist <= self.max_distance

        pedestrians = []
        for ann in filtered["pedestrians"]:
            if _within_range(ann):
                try:
                    pedestrians.append({
                        "token": ann["token"],
                        "category": ann["category_name"],
                        "polygon": self.annotation_footprint(ann),
                    })
                except Exception:
                    continue

        vehicles = []
        for ann in filtered["vehicles"]:
            if _within_range(ann):
                try:
                    vehicles.append({
                        "token": ann["token"],
                        "category": ann["category_name"],
                        "polygon": self.annotation_footprint(ann),
                    })
                except Exception:
                    continue

        return {
            "ego": ego_poly,
            "pedestrians": pedestrians,
            "vehicles": vehicles,
            "crosswalks": crosswalks,
        }
