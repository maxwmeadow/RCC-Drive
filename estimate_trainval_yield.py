"""Estimate how many trainval samples would pass the CAM_FRONT FOV filter.

Uses metadata + map polygons only — no full sensor data required.
Pedestrian-on-crosswalk check uses a point-in-polygon proxy (centroid distance
< pedestrian half-width) instead of full Shapely footprints, so the count is
a close approximation, not an exact match to running the full pipeline.
"""

import math
import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Point
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

DATAROOT = "data/nuscenes"
VERSION = "v1.0-trainval"
CROSSWALK_RADIUS = 80.0
MAX_DIST = 50.0
PED_HALF_WIDTH = 0.45  # proxy for PO: centroid within ~half pedestrian width of crosswalk

print("Loading nuScenes...")
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
print(f"  {len(nusc.scene)} scenes, {len(nusc.sample)} samples")

map_names = list({
    nusc.get("log", nusc.get("scene", s["token"])["log_token"])["location"]
    for s in nusc.scene
})
maps = {name: NuScenesMap(dataroot=DATAROOT, map_name=name) for name in map_names}
print(f"  Maps loaded: {map_names}")


def get_map_name(scene_token):
    return nusc.get("log", nusc.get("scene", scene_token)["log_token"])["location"]


# Index pedestrian annotations by sample_token
print("Indexing pedestrian annotations...")
ped_by_sample = {}
for ann in nusc.sample_annotation:
    if ann["category_name"].startswith("human.pedestrian."):
        ped_by_sample.setdefault(ann["sample_token"], []).append(ann)
print(f"  {sum(len(v) for v in ped_by_sample.values())} ped annotations "
      f"across {len(ped_by_sample)} samples")

# CAM_FRONT calibrated half-FOV
cs_lookup = {cs["token"]: cs for cs in nusc.calibrated_sensor}
sample_to_cam_sd = {
    sd["sample_token"]: sd
    for sd in nusc.sample_data
    if sd["channel"] == "CAM_FRONT" and sd["is_key_frame"]
}


def get_half_fov(sample_token):
    sd = sample_to_cam_sd.get(sample_token)
    if not sd:
        return 32.42  # fallback mean from mini calibration
    cs = cs_lookup[sd["calibrated_sensor_token"]]
    fx = cs["camera_intrinsic"][0][0]
    return math.degrees(math.atan(sd["width"] / (2.0 * fx)))


def ped_in_fov(ego_pose, ped_translation, half_fov):
    ego_pos = np.array(ego_pose["translation"][:2])
    ped_pos = np.array(ped_translation[:2])
    vec = ped_pos - ego_pos
    if np.linalg.norm(vec) < 1e-6:
        return False
    v3 = Quaternion(ego_pose["rotation"]).inverse.rotate([vec[0], vec[1], 0.0])
    return abs(math.degrees(math.atan2(v3[1], v3[0]))) <= half_fov


print("\nScanning samples...")
total = kept = behind = no_critical = 0

for sample in nusc.sample:
    total += 1

    if total % 500 == 0:
        print(f"  {total} samples processed, {kept} kept so far...")

    sample_token = sample["token"]
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    ex, ey = ego_pose["translation"][0], ego_pose["translation"][1]

    nusc_map = maps[get_map_name(sample["scene_token"])]
    nearby = nusc_map.get_records_in_radius(ex, ey, CROSSWALK_RADIUS, ["ped_crossing"])
    if not nearby["ped_crossing"]:
        no_critical += 1
        continue

    cw_polys = [
        nusc_map.extract_polygon(nusc_map.get("ped_crossing", t)["polygon_token"])
        for t in nearby["ped_crossing"]
    ]

    critical = []
    for ann in ped_by_sample.get(sample_token, []):
        t = ann["translation"]
        if math.hypot(t[0] - ex, t[1] - ey) > MAX_DIST:
            continue
        pt = Point(t[0], t[1])
        if any(poly.contains(pt) or poly.distance(pt) < PED_HALF_WIDTH for poly in cw_polys):
            critical.append(t)

    if not critical:
        no_critical += 1
        continue

    half_fov = get_half_fov(sample_token)
    if any(ped_in_fov(ego_pose, t, half_fov) for t in critical):
        kept += 1
    else:
        behind += 1

print(f"\n{'='*50}")
print(f"Total samples:           {total}")
print(f"No critical ped/cw:      {no_critical}")
print(f"Peds behind ego:         {behind}")
print(f"Kept (in CAM_FRONT FOV): {kept}")
print(f"Yield rate:              {kept/total*100:.1f}%")
print(f"\nMini result:             59 / 404 = {59/404*100:.1f}%")
print(f"Full download estimate:  ~{kept} curated samples from trainval")
