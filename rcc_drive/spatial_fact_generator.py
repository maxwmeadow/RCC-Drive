"""Orchestrator: generates spatial fact sheets (RCC-8 relations) for nuScenes samples."""

import json
import logging
import os
from typing import List, Optional

from rcc_drive.config import DEFAULT_CONFIG
from rcc_drive.nuscenes_loader import NuScenesDataLoader
from rcc_drive.region_extractor import RegionExtractor
from rcc_drive.rcc8_classifier import classify_rcc8

logger = logging.getLogger(__name__)


class SpatialFactGenerator:
    """Generates structured spatial fact JSON files from nuScenes data."""

    def __init__(self, version=None, dataroot=None, config=None):
        self.config = config or DEFAULT_CONFIG
        self.loader = NuScenesDataLoader(
            version=version or self.config["nuscenes_version"],
            dataroot=dataroot or self.config["dataroot"],
        )
        self.extractor = RegionExtractor(config=self.config)

    def generate_for_sample(self, sample_token: str) -> dict:
        """Generate the spatial fact sheet for a single sample.

        Returns a dict ready to be serialized to JSON.
        """
        sample = self.loader.nusc.get("sample", sample_token)
        map_name = self.loader.get_map_name_for_scene(sample["scene_token"])

        # Load data
        ego_pose = self.loader.get_ego_pose_for_sample(sample_token)
        annotations = self.loader.get_annotations_for_sample(sample_token)
        ego_x, ego_y = ego_pose["translation"][0], ego_pose["translation"][1]
        crosswalks = self.loader.get_crosswalk_polygons_near(map_name, ego_x, ego_y)

        # Extract polygons
        regions = self.extractor.extract_all_regions(ego_pose, annotations, crosswalks)

        # Compute RCC-8 relations
        relations = []
        dc_vehicle_pairs_dropped = 0
        dc_ped_crosswalk_pairs_dropped = 0

        # Ego vs. each pedestrian
        for ped in regions["pedestrians"]:
            rel = classify_rcc8(regions["ego"], ped["polygon"])
            relations.append({
                "subject": "ego",
                "object": ped["token"],
                "object_category": ped["category"],
                "relation": rel,
                "pair_type": "ego_vs_pedestrian",
            })

        # Each pedestrian vs. each crosswalk (DC pairs dropped — uninformative for prompts)
        for ped in regions["pedestrians"]:
            for cw in regions["crosswalks"]:
                rel = classify_rcc8(ped["polygon"], cw["polygon"])
                if rel == "DC":
                    dc_ped_crosswalk_pairs_dropped += 1
                    continue
                relations.append({
                    "subject": ped["token"],
                    "subject_category": ped["category"],
                    "object": cw["token"],
                    "object_category": "ped_crossing",
                    "relation": rel,
                    "pair_type": "pedestrian_vs_crosswalk",
                })

        # Ego vs. each vehicle
        for veh in regions["vehicles"]:
            rel = classify_rcc8(regions["ego"], veh["polygon"])
            relations.append({
                "subject": "ego",
                "object": veh["token"],
                "object_category": veh["category"],
                "relation": rel,
                "pair_type": "ego_vs_vehicle",
            })

        # Vehicle vs. vehicle (pairwise, distance-filtered, DC post-filtered)
        vehicles = regions["vehicles"]
        veh_pair_max_dist = self.config["vehicle_pair_max_distance"]
        drop_dc = self.config["drop_dc_vehicle_pairs"]

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                dist = vehicles[i]["polygon"].centroid.distance(
                    vehicles[j]["polygon"].centroid
                )
                if dist < veh_pair_max_dist:
                    rel = classify_rcc8(vehicles[i]["polygon"], vehicles[j]["polygon"])
                    if drop_dc and rel == "DC":
                        dc_vehicle_pairs_dropped += 1
                        continue
                    relations.append({
                        "subject": vehicles[i]["token"],
                        "subject_category": vehicles[i]["category"],
                        "object": vehicles[j]["token"],
                        "object_category": vehicles[j]["category"],
                        "relation": rel,
                        "pair_type": "vehicle_vs_vehicle",
                    })

        return {
            "sample_token": sample_token,
            "scene_token": sample["scene_token"],
            "timestamp": sample["timestamp"],
            "ego_pose": {
                "translation": ego_pose["translation"],
                "rotation": ego_pose["rotation"],
            },
            "num_pedestrians": len(regions["pedestrians"]),
            "num_vehicles": len(regions["vehicles"]),
            "num_crosswalks": len(regions["crosswalks"]),
            "relations": relations,
            "metadata": {
                "dc_vehicle_pairs_dropped": dc_vehicle_pairs_dropped,
                "dc_ped_crosswalk_pairs_dropped": dc_ped_crosswalk_pairs_dropped,
            },
            "config": {
                "max_object_distance": self.config["max_object_distance"],
                "crosswalk_search_radius": self.config["crosswalk_search_radius"],
                "ego_dimensions": [self.config["ego_length"], self.config["ego_width"]],
                "coordinate_frame": "global_bev",
            },
        }

    def generate_for_scene(self, scene_token: str) -> List[dict]:
        """Generate spatial facts for all samples in a scene."""
        sample_tokens = self.loader.get_sample_tokens_for_scene(scene_token)
        results = []
        for st in sample_tokens:
            try:
                results.append(self.generate_for_sample(st))
            except Exception as e:
                logger.error(f"Failed to process sample {st}: {e}")
        return results

    def generate_all(self, output_dir: str) -> dict:
        """Process all scenes and write one JSON per sample.

        Returns a summary dict with counts.
        """
        os.makedirs(output_dir, exist_ok=True)

        total_samples = 0
        total_relations = 0
        total_dc_dropped = 0
        samples_with_crosswalks = 0
        samples_with_pedestrians = 0
        samples_with_both = 0

        for scene_token in self.loader.get_scene_tokens():
            facts = self.generate_for_scene(scene_token)
            for fact in facts:
                total_samples += 1
                total_relations += len(fact["relations"])
                total_dc_dropped += (
                    fact["metadata"]["dc_vehicle_pairs_dropped"]
                    + fact["metadata"].get("dc_ped_crosswalk_pairs_dropped", 0)
                )

                has_cw = fact["num_crosswalks"] > 0
                has_ped = fact["num_pedestrians"] > 0
                if has_cw:
                    samples_with_crosswalks += 1
                if has_ped:
                    samples_with_pedestrians += 1
                if has_cw and has_ped:
                    samples_with_both += 1

                path = os.path.join(output_dir, f"{fact['sample_token']}.json")
                with open(path, "w") as f:
                    json.dump(fact, f, indent=2)

        summary = {
            "total_samples": total_samples,
            "total_relations": total_relations,
            "dc_vehicle_pairs_dropped": total_dc_dropped,
            "samples_with_crosswalks": samples_with_crosswalks,
            "samples_with_pedestrians": samples_with_pedestrians,
            "samples_with_both_crosswalks_and_pedestrians": samples_with_both,
        }

        # Write summary
        summary_path = os.path.join(output_dir, "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Generated {total_samples} spatial fact files")
        logger.info(f"Crosswalk coverage: {samples_with_both}/{total_samples} samples "
                     f"have both pedestrians and crosswalks")

        return summary


if __name__ == "__main__":
    import pprint
    gen = SpatialFactGenerator()
    token = gen.loader.nusc.sample[0]["token"]
    result = gen.generate_for_sample(token)
    pprint.pprint(result)
