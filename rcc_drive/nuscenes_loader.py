"""nuScenes data loader wrapping the devkit SDK.

All nuScenes SDK imports are contained here so other modules
don't depend on nuscenes-devkit directly.
"""

import os
from typing import List, Optional

from rcc_drive.config import DEFAULT_CONFIG


class NuScenesDataLoader:
    """Loads and provides access to nuScenes data: annotations, ego poses, and map features."""

    def __init__(self, version=None, dataroot=None):
        from nuscenes.nuscenes import NuScenes
        version = version or DEFAULT_CONFIG["nuscenes_version"]
        dataroot = dataroot or DEFAULT_CONFIG["dataroot"]

        self.dataroot = dataroot
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self._map_cache = {}

    def get_scene_tokens(self) -> List[str]:
        """Return all scene tokens in the dataset."""
        return [s["token"] for s in self.nusc.scene]

    def get_sample_tokens_for_scene(self, scene_token: str) -> List[str]:
        """Return ordered list of sample tokens for a scene by walking the linked list."""
        scene = self.nusc.get("scene", scene_token)
        tokens = []
        sample_token = scene["first_sample_token"]
        while sample_token:
            tokens.append(sample_token)
            sample = self.nusc.get("sample", sample_token)
            sample_token = sample["next"] if sample["next"] != "" else None
        return tokens

    def get_annotations_for_sample(self, sample_token: str) -> List[dict]:
        """Return full annotation records for all annotations in a sample."""
        sample = self.nusc.get("sample", sample_token)
        return [self.nusc.get("sample_annotation", ann_token) for ann_token in sample["anns"]]

    def get_ego_pose_for_sample(self, sample_token: str) -> dict:
        """Get ego pose for a sample via the LIDAR_TOP sample_data.

        Returns dict with 'translation' [x, y, z] and 'rotation' [w, x, y, z]
        in global coordinates.
        """
        sample = self.nusc.get("sample", sample_token)
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        return self.nusc.get("ego_pose", lidar_data["ego_pose_token"])

    def get_map_name_for_scene(self, scene_token: str) -> str:
        """Get the map name for a scene: scene -> log -> location."""
        scene = self.nusc.get("scene", scene_token)
        log = self.nusc.get("log", scene["log_token"])
        return log["location"]

    def _get_map(self, map_name: str):
        """Get or create a cached NuScenesMap instance."""
        if map_name not in self._map_cache:
            from nuscenes.map_expansion.map_api import NuScenesMap
            self._map_cache[map_name] = NuScenesMap(
                dataroot=self.dataroot, map_name=map_name
            )
        return self._map_cache[map_name]

    def get_crosswalk_polygons_near(
        self, map_name: str, x: float, y: float, radius: float = None
    ) -> List[dict]:
        """Get crosswalk (ped_crossing) polygons near a position.

        Args:
            map_name: nuScenes map name (e.g. 'boston-seaport')
            x, y: Center position in global coordinates
            radius: Search radius in meters

        Returns:
            List of dicts with 'token' and 'polygon' (Shapely Polygon) keys.
        """
        if radius is None:
            radius = DEFAULT_CONFIG["crosswalk_search_radius"]

        nusc_map = self._get_map(map_name)

        # get_records_in_radius returns dict of layer_name -> list of tokens
        records = nusc_map.get_records_in_radius(x, y, radius, ["ped_crossing"])
        crossing_tokens = records.get("ped_crossing", [])

        result = []
        for token in crossing_tokens:
            record = nusc_map.get("ped_crossing", token)
            polygon_token = record["polygon_token"]
            polygon = nusc_map.extract_polygon(polygon_token)

            if not polygon.is_empty and polygon.area > 0:
                result.append({
                    "token": token,
                    "polygon": polygon,
                })

        return result

    def get_camera_image_path(
        self, sample_token: str, camera: str = "CAM_FRONT"
    ) -> str:
        """Get the file path to a camera image for a sample.

        Convenience method for Person 2's pipeline.
        """
        sample = self.nusc.get("sample", sample_token)
        cam_data = self.nusc.get("sample_data", sample["data"][camera])
        return os.path.join(self.dataroot, cam_data["filename"])
