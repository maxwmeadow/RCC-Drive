"""Integration tests requiring nuScenes v1.0-mini to be downloaded.

These tests are skipped if the dataset is not available.
"""

import json
import os
import tempfile

import pytest

from rcc_drive.config import DEFAULT_CONFIG, RCC8_RELATIONS

DATAROOT = DEFAULT_CONFIG["dataroot"]
SKIP_REASON = "nuScenes v1.0-mini not found at {}".format(DATAROOT)

# Check for both metadata and map expansion
HAS_NUSCENES = (
    os.path.isdir(os.path.join(DATAROOT, "v1.0-mini"))
    and os.path.isdir(os.path.join(DATAROOT, "maps"))
)


@pytest.mark.skipif(not HAS_NUSCENES, reason=SKIP_REASON)
class TestIntegration:
    """End-to-end tests on the nuScenes mini dataset."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from rcc_drive.spatial_fact_generator import SpatialFactGenerator
        self.gen = SpatialFactGenerator(version="v1.0-mini", dataroot=DATAROOT)

    def test_coordinate_sanity_check(self):
        """Verify that annotation translations are in global coords (near ego pose)."""
        scene_token = self.gen.loader.get_scene_tokens()[0]
        sample_tokens = self.gen.loader.get_sample_tokens_for_scene(scene_token)
        sample_token = sample_tokens[0]

        ego_pose = self.gen.loader.get_ego_pose_for_sample(sample_token)
        annotations = self.gen.loader.get_annotations_for_sample(sample_token)

        ego_x, ego_y = ego_pose["translation"][0], ego_pose["translation"][1]

        if annotations:
            ann = annotations[0]
            ax, ay = ann["translation"][0], ann["translation"][1]
            dist = ((ax - ego_x) ** 2 + (ay - ego_y) ** 2) ** 0.5
            # Annotations should be within reasonable range of ego (not at origin)
            assert dist < 200, (
                f"Annotation at ({ax:.1f}, {ay:.1f}) is {dist:.1f}m from ego at "
                f"({ego_x:.1f}, {ego_y:.1f}). Annotations may not be in global coords!"
            )
            # Ego should NOT be near the origin (nuScenes uses large global coords)
            assert abs(ego_x) > 10 or abs(ego_y) > 10, (
                f"Ego at ({ego_x:.1f}, {ego_y:.1f}) is suspiciously near origin"
            )

    def test_generate_single_sample(self):
        """Generate spatial facts for one sample and validate structure."""
        scene_token = self.gen.loader.get_scene_tokens()[0]
        sample_tokens = self.gen.loader.get_sample_tokens_for_scene(scene_token)
        result = self.gen.generate_for_sample(sample_tokens[0])

        # Structure checks
        assert result["sample_token"] == sample_tokens[0]
        assert "relations" in result
        assert "ego_pose" in result
        assert "num_pedestrians" in result
        assert "num_vehicles" in result
        assert "num_crosswalks" in result
        assert "metadata" in result
        assert "dc_vehicle_pairs_dropped" in result["metadata"]

        # All relations have valid RCC-8 labels
        for rel in result["relations"]:
            assert rel["relation"] in RCC8_RELATIONS, f"Invalid relation: {rel['relation']}"
            assert "pair_type" in rel
            assert rel["pair_type"] in (
                "ego_vs_pedestrian",
                "pedestrian_vs_crosswalk",
                "ego_vs_vehicle",
                "vehicle_vs_vehicle",
            )

    def test_json_serializable(self):
        """Ensure output is valid JSON."""
        scene_token = self.gen.loader.get_scene_tokens()[0]
        sample_tokens = self.gen.loader.get_sample_tokens_for_scene(scene_token)
        result = self.gen.generate_for_sample(sample_tokens[0])
        # Should not raise
        serialized = json.dumps(result)
        # Should round-trip
        parsed = json.loads(serialized)
        assert parsed["sample_token"] == result["sample_token"]

    def test_generate_full_scene(self):
        """Generate facts for all samples in a scene."""
        scene_token = self.gen.loader.get_scene_tokens()[0]
        results = self.gen.generate_for_scene(scene_token)
        assert len(results) > 0
        # Each sample in the scene should produce a valid result
        for r in results:
            assert r["scene_token"] == scene_token
            json.dumps(r)  # should not raise

    def test_generate_all_with_summary(self):
        """Run generate_all and check the summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = self.gen.generate_all(tmpdir)

            assert summary["total_samples"] > 0
            assert "samples_with_crosswalks" in summary
            assert "samples_with_pedestrians" in summary
            assert "samples_with_both_crosswalks_and_pedestrians" in summary

            # Check files were written
            json_files = [f for f in os.listdir(tmpdir) if f.endswith(".json") and f != "_summary.json"]
            assert len(json_files) == summary["total_samples"]

            # Summary file exists
            assert os.path.exists(os.path.join(tmpdir, "_summary.json"))

    def test_crosswalk_coverage_audit(self):
        """Report crosswalk + pedestrian coverage for Person 3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = self.gen.generate_all(tmpdir)

            total = summary["total_samples"]
            both = summary["samples_with_both_crosswalks_and_pedestrians"]
            cw = summary["samples_with_crosswalks"]
            ped = summary["samples_with_pedestrians"]

            print(f"\n=== Crosswalk Coverage Audit ===")
            print(f"Total samples: {total}")
            print(f"Samples with crosswalks: {cw} ({100*cw/total:.1f}%)")
            print(f"Samples with pedestrians: {ped} ({100*ped/total:.1f}%)")
            print(f"Samples with BOTH: {both} ({100*both/total:.1f}%)")
            print(f"DC vehicle pairs dropped: {summary['dc_vehicle_pairs_dropped']}")

    def test_visualizer(self):
        """Test that the visualizer produces a PNG without errors."""
        from rcc_drive.visualizer import plot_sample_bev

        scene_token = self.gen.loader.get_scene_tokens()[0]
        sample_tokens = self.gen.loader.get_sample_tokens_for_scene(scene_token)
        sample_token = sample_tokens[0]

        ego_pose = self.gen.loader.get_ego_pose_for_sample(sample_token)
        annotations = self.gen.loader.get_annotations_for_sample(sample_token)
        map_name = self.gen.loader.get_map_name_for_scene(scene_token)
        ego_x, ego_y = ego_pose["translation"][0], ego_pose["translation"][1]
        crosswalks = self.gen.loader.get_crosswalk_polygons_near(map_name, ego_x, ego_y)

        regions = self.gen.extractor.extract_all_regions(ego_pose, annotations, crosswalks)
        result = self.gen.generate_for_sample(sample_token)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_bev.png")
            plot_sample_bev(
                regions=regions,
                relations=result["relations"],
                ego_pose=ego_pose,
                sample_token=sample_token,
                output_path=out_path,
            )
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 1000  # non-trivial PNG
