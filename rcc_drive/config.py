"""Configuration constants for the RCC-Drive spatial extraction pipeline."""

import os

DEFAULT_CONFIG = {
    # Distance thresholds (meters)
    "max_object_distance": 50.0,         # Ignore annotations farther than this from ego
    "crosswalk_search_radius": 80.0,     # Radius to search for crosswalks around ego
    "vehicle_pair_max_distance": 20.0,   # Only compute veh-veh RCC-8 if centroids closer than this

    # Ego vehicle dimensions (Renault Zoe - nuScenes data collection vehicle)
    "ego_length": 4.084,                 # meters, front-to-back
    "ego_width": 1.730,                  # meters, side-to-side
    "ego_buffer": 0.0,                   # Optional buffer around ego footprint (meters)

    # RCC-8 classifier
    "boundary_tolerance": 1e-6,          # Tolerance for equals_exact comparison
    "min_polygon_area": 1e-8,            # Below this area, treat polygon as degenerate

    # Post-filtering
    "drop_dc_vehicle_pairs": True,       # Omit DC relations from vehicle-vs-vehicle output

    # nuScenes
    "nuscenes_version": "v1.0-mini",
    "dataroot": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nuscenes"),
}

# Safety-critical category prefixes
PEDESTRIAN_CATEGORIES = ("human.pedestrian",)
VEHICLE_CATEGORIES = ("vehicle.",)

# RCC-8 relation labels
RCC8_RELATIONS = ("DC", "EC", "PO", "EQ", "TPP", "TPPi", "NTPP", "NTPPi")
