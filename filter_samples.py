"""Filter spatial_facts samples to those where safety-critical pedestrians
(NTPP or PO with a crosswalk) are within the CAM_FRONT field of view.

Reads from output/spatial_facts/*.json — no need to re-run extraction.
Writes output/curated_samples.json.
"""

import json
import math
import os
import argparse
import logging

import numpy as np
from pyquaternion import Quaternion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Fallback half-FOV used only if calibrated_sensor lookup fails.
# Approximate value for the nuScenes Basler cameras (~70° horizontal FOV).
_FALLBACK_HALF_FOV_DEG = 35.0


def get_cam_front_half_fov(nusc, sample_token: str) -> float:
    """Compute the CAM_FRONT half-FOV from the calibrated sensor intrinsics.

    Derivation:
        FOV_horizontal = 2 * arctan(image_width / (2 * fx))
        half_fov       = FOV_horizontal / 2

    where fx = camera_intrinsic[0][0] and image_width is read from the
    sample_data record so the result is robust to any sensor configuration.

    Args:
        nusc: Loaded NuScenes instance.
        sample_token: Token of the sample to look up.

    Returns:
        Half the horizontal FOV in degrees, derived from sensor calibration.
        Falls back to _FALLBACK_HALF_FOV_DEG if the lookup fails.
    """
    try:
        sample = nusc.get("sample", sample_token)
        sd_token = sample["data"]["CAM_FRONT"]
        sd = nusc.get("sample_data", sd_token)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

        fx = cs["camera_intrinsic"][0][0]
        image_width = sd["width"]  # 1600 px for all standard nuScenes cameras

        half_fov_rad = math.atan(image_width / (2.0 * fx))
        return math.degrees(half_fov_rad)
    except Exception as e:
        logger.warning(
            f"Could not read CAM_FRONT intrinsics for {sample_token}: {e}. "
            f"Falling back to {_FALLBACK_HALF_FOV_DEG}°."
        )
        return _FALLBACK_HALF_FOV_DEG


def ped_in_cam_front_fov(ego_pose: dict, ped_translation: list,
                          half_fov_deg: float) -> bool:
    """Check if a pedestrian is within the CAM_FRONT horizontal field of view.

    Strategy: transform the ego-to-pedestrian vector into the ego vehicle's
    local frame. In the local frame, the forward direction is the x-axis.
    If the angle between the vector and the x-axis is within ±half_fov_deg,
    the pedestrian is in view.

    Args:
        ego_pose: Dict with 'translation' [x,y,z] and 'rotation' [w,x,y,z].
        ped_translation: Pedestrian global position [x, y, z].
        half_fov_deg: Half the horizontal FOV in degrees (from sensor calibration).

    Returns:
        True if the pedestrian is within the horizontal FOV.
    """
    ego_pos = np.array(ego_pose["translation"][:2])  # x, y only
    ped_pos = np.array(ped_translation[:2])

    # Vector from ego to pedestrian in global frame
    global_vec = ped_pos - ego_pos
    if np.linalg.norm(global_vec) < 1e-6:
        return False  # degenerate: pedestrian at ego position

    # Rotate the vector into the ego local frame using the inverse ego rotation.
    # nuScenes quaternion is [w, x, y, z].
    q = Quaternion(ego_pose["rotation"])
    q_inv = q.inverse

    # Extend to 3D for quaternion rotation, then back to 2D
    global_vec_3d = np.array([global_vec[0], global_vec[1], 0.0])
    local_vec_3d = q_inv.rotate(global_vec_3d)
    local_x, local_y = local_vec_3d[0], local_vec_3d[1]

    # Angle from the forward (x-axis) direction in the local frame
    angle_deg = math.degrees(math.atan2(local_y, local_x))

    return abs(angle_deg) <= half_fov_deg


def filter_samples(facts_dir: str, nusc=None) -> dict:
    """Filter all spatial_facts samples by CAM_FRONT FOV visibility.

    The half-FOV is computed per sample from the nuScenes calibrated_sensor
    intrinsics (fx and image width), ensuring the filter reflects the actual
    sensor configuration rather than an assumed value.

    Args:
        facts_dir: Directory containing per-sample JSON files.
        nusc: Loaded NuScenes instance (for annotation and intrinsic lookup).

    Returns:
        Dict with kept/dropped sample tokens, per-sample FOV values, and
        summary stats including min/mean/max half-FOV across processed samples.
    """
    files = sorted([
        f for f in os.listdir(facts_dir)
        if f.endswith(".json") and f != "_summary.json"
    ])

    kept = []
    dropped = []
    no_critical_peds = []  # samples with no NTPP/PO crosswalk relations at all
    half_fov_values = []   # calibrated half-FOV per sample that reached the FOV check

    for fname in files:
        with open(os.path.join(facts_dir, fname)) as f:
            fact = json.load(f)

        sample_token = fact["sample_token"]
        ego_pose = fact["ego_pose"]

        # Find pedestrians with safety-critical crosswalk relations
        critical_ped_tokens = set()
        for rel in fact["relations"]:
            if (rel["pair_type"] == "pedestrian_vs_crosswalk"
                    and rel["relation"] in ("NTPP", "PO")):
                critical_ped_tokens.add(rel["subject"])

        if not critical_ped_tokens:
            no_critical_peds.append(sample_token)
            continue

        # Derive half-FOV from calibrated sensor intrinsics for this sample
        half_fov_deg = get_cam_front_half_fov(nusc, sample_token)
        half_fov_values.append(half_fov_deg)

        # Check if any critical pedestrian is within CAM_FRONT FOV
        any_in_fov = False
        for ped_token in critical_ped_tokens:
            ann = nusc.get("sample_annotation", ped_token)
            ped_translation = ann["translation"]

            if ped_in_cam_front_fov(ego_pose, ped_translation, half_fov_deg):
                any_in_fov = True
                break

        if any_in_fov:
            kept.append(sample_token)
        else:
            dropped.append(sample_token)

    fov_stats = {}
    if half_fov_values:
        fov_stats = {
            "cam_front_half_fov_deg_min": round(min(half_fov_values), 4),
            "cam_front_half_fov_deg_mean": round(sum(half_fov_values) / len(half_fov_values), 4),
            "cam_front_half_fov_deg_max": round(max(half_fov_values), 4),
            "cam_front_half_fov_source": "calibrated_sensor intrinsics (fx, image_width)",
        }

    return {
        "kept": kept,
        "dropped": dropped,
        "no_critical_pedestrians": no_critical_peds,
        "summary": {
            "total_samples": len(files),
            "kept": len(kept),
            "dropped_behind_ego": len(dropped),
            "no_critical_pedestrians": len(no_critical_peds),
            **fov_stats,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter samples to those with safety-critical peds in CAM_FRONT FOV."
    )
    parser.add_argument("--facts-dir", default="output/spatial_facts")
    parser.add_argument("--output", default="output/curated_samples.json")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--dataroot", default="data/nuscenes")
    args = parser.parse_args()

    logger.info("Loading nuScenes for annotation and intrinsic lookup...")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    logger.info(f"Filtering {args.facts_dir} (FOV from calibrated_sensor intrinsics)...")
    result = filter_samples(args.facts_dir, nusc=nusc)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    s = result["summary"]
    logger.info(f"\n{'='*50}")
    logger.info(f"Total samples:                {s['total_samples']}")
    logger.info(f"No safety-critical peds:      {s['no_critical_pedestrians']}")
    logger.info(f"Peds behind/beside ego:       {s['dropped_behind_ego']}")
    logger.info(f"Kept (peds in CAM_FRONT FOV): {s['kept']}")
    if "cam_front_half_fov_deg_mean" in s:
        logger.info(
            f"Half-FOV (calibrated):        "
            f"min={s['cam_front_half_fov_deg_min']}°  "
            f"mean={s['cam_front_half_fov_deg_mean']}°  "
            f"max={s['cam_front_half_fov_deg_max']}°"
        )
    logger.info(f"\nCurated set: {s['kept']} samples → {args.output}")


if __name__ == "__main__":
    main()
