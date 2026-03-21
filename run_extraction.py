"""CLI entry point: extract RCC-8 spatial facts from nuScenes."""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract deterministic RCC-8 spatial facts from nuScenes."
    )
    parser.add_argument(
        "--version", default="v1.0-mini",
        help="nuScenes version (default: v1.0-mini)",
    )
    parser.add_argument(
        "--dataroot", default=None,
        help="Path to nuScenes data directory (default: ./data/nuscenes)",
    )
    parser.add_argument(
        "--output", default="output/spatial_facts",
        help="Output directory for spatial fact JSON files",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate BEV visualizations alongside JSON output",
    )
    parser.add_argument(
        "--vis-output", default="output/visualizations",
        help="Output directory for BEV visualization PNGs",
    )
    parser.add_argument(
        "--scene", default=None,
        help="Process only this scene token (default: all scenes)",
    )
    args = parser.parse_args()

    from rcc_drive.spatial_fact_generator import SpatialFactGenerator
    from rcc_drive.visualizer import plot_sample_bev

    gen = SpatialFactGenerator(version=args.version, dataroot=args.dataroot)

    if args.scene:
        # Single scene
        logger.info(f"Processing scene {args.scene}")
        facts = gen.generate_for_scene(args.scene)
        os.makedirs(args.output, exist_ok=True)
        for fact in facts:
            path = os.path.join(args.output, f"{fact['sample_token']}.json")
            with open(path, "w") as f:
                json.dump(fact, f, indent=2)
            logger.info(f"  {fact['sample_token'][:16]}... "
                        f"({len(fact['relations'])} relations)")

            if args.visualize:
                _visualize_sample(gen, fact, args.vis_output)

        logger.info(f"Wrote {len(facts)} files to {args.output}")
    else:
        # All scenes
        logger.info("Processing all scenes...")
        summary = gen.generate_all(args.output)

        logger.info(f"\n{'='*50}")
        logger.info(f"Summary:")
        logger.info(f"  Total samples: {summary['total_samples']}")
        logger.info(f"  Total relations: {summary['total_relations']}")
        logger.info(f"  DC vehicle pairs dropped: {summary['dc_vehicle_pairs_dropped']}")
        logger.info(f"  Samples with crosswalks: {summary['samples_with_crosswalks']}")
        logger.info(f"  Samples with pedestrians: {summary['samples_with_pedestrians']}")
        logger.info(f"  Samples with BOTH: {summary['samples_with_both_crosswalks_and_pedestrians']}")

        if args.visualize:
            logger.info(f"\nGenerating visualizations...")
            for scene_token in gen.loader.get_scene_tokens():
                sample_tokens = gen.loader.get_sample_tokens_for_scene(scene_token)
                for st in sample_tokens:
                    fact_path = os.path.join(args.output, f"{st}.json")
                    with open(fact_path) as f:
                        fact = json.load(f)
                    _visualize_sample(gen, fact, args.vis_output)
            logger.info(f"Visualizations saved to {args.vis_output}")


def _visualize_sample(gen, fact, vis_output):
    """Generate a BEV visualization for a single sample."""
    from rcc_drive.visualizer import plot_sample_bev

    sample_token = fact["sample_token"]
    ego_pose = gen.loader.get_ego_pose_for_sample(sample_token)
    annotations = gen.loader.get_annotations_for_sample(sample_token)
    scene = gen.loader.nusc.get("sample", sample_token)
    map_name = gen.loader.get_map_name_for_scene(scene["scene_token"])
    ego_x, ego_y = ego_pose["translation"][0], ego_pose["translation"][1]
    crosswalks = gen.loader.get_crosswalk_polygons_near(map_name, ego_x, ego_y)
    regions = gen.extractor.extract_all_regions(ego_pose, annotations, crosswalks)

    vis_path = os.path.join(vis_output, f"{sample_token}.png")
    plot_sample_bev(
        regions=regions,
        relations=fact["relations"],
        ego_pose=ego_pose,
        sample_token=sample_token,
        output_path=vis_path,
    )


if __name__ == "__main__":
    main()
