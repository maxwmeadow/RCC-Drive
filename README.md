# RCC-Drive — Phase 1: Spatial Extraction Pipeline

**CSE 834 Group Project** | Person 1 deliverable

Deterministically computed RCC-8 topological relations from nuScenes sensor data, used to ground VLM prompts in formal spatial constraints for autonomous driving scene understanding.

---

## Overview

Phase 1 converts nuScenes samples into structured JSON files (`spatial_facts`) containing RCC-8 region connection calculus relations for every safety-critical object pair visible to the ego vehicle. These facts are consumed by Phase 2 (VLM prompting) and Phase 3 (evaluation).

The core claim: RCC-8 relations computed from sensor data are **deterministic and verifiable**, unlike VLM spatial assertions which can hallucinate. Injecting these as hard constraints into prompts should reduce spatial hallucination in driving decisions.

---

## Quick Start

```bash
# Requires Python 3.9 or 3.12 (nuScenes devkit constraint)

# 1. Clone and set up environment
git clone <repo>
cd RCC-Drive
python -m venv venv
source venv/Scripts/activate      # Windows/MSYS2
# source venv/bin/activate        # Linux/macOS
pip install -r requirements.txt

# 2. Download data (see Data Setup below)

# 3. Run full extraction (v1.0-mini, ~2 min)
python run_extraction.py

# 4. Run CAM_FRONT FOV filter
python filter_samples.py

# 5. Outputs
#   output/spatial_facts/<sample_token>.json  — one file per sample
#   output/spatial_facts/_summary.json        — pipeline stats
#   output/curated_samples.json               — filtered sample list for Phase 2/3
```

Run tests:
```bash
pytest tests/test_rcc8_classifier.py tests/test_region_extractor.py -v   # no dataset required, pure geometry + mock annotations
pytest tests/test_integration.py -v                                        # requires v1.0-mini dataset
```

Inspect a single sample without running the full pipeline (useful for debugging and Person 2 integration):
```bash
python -m rcc_drive.spatial_fact_generator   # pretty-prints fact sheet for sample[0]
```

---

## Data Setup

### Required downloads

| File | Size | Purpose |
|------|------|---------|
| `v1.0-mini.tgz` | ~0.4 GB | Development/validation split (metadata + sensor data) |
| `nuScenes-map-expansion-v1.3.zip` | ~0.4 GB | Crosswalk polygons (required for both splits) |
| `v1.0-trainval_meta.tgz` | ~0.4 GB | Trainval metadata only (no sensor blobs needed for Phase 1) |

> **Note:** The trainval sensor blobs (~350 GB total) are **not required** for Phase 1. Only the metadata JSON tables are needed to run the extraction pipeline and the yield estimation script.

### Directory structure

```
RCC-Drive/
├── data/nuscenes/
│   ├── v1.0-mini/          # metadata JSON tables
│   ├── v1.0-trainval/      # metadata JSON tables (metadata tgz only)
│   ├── samples/            # sensor images (mini only)
│   ├── sweeps/             # lidar/radar sweeps (mini only)
│   └── maps/
│       ├── expansion/      # map expansion v1.3 (4 JSON files)
│       ├── basemap/        # basemap PNGs
│       └── prediction/
├── rcc_drive/              # pipeline source
├── tests/
├── output/
│   ├── spatial_facts/      # generated, not committed (404 files ~15 MB)
│   ├── visualizations/     # generated, not committed
│   └── curated_samples.json  # committed — primary Phase 2/3 handoff artifact
└── dummy/
    └── spatial_facts_dummy.json  # hand-crafted example for Person 2
```

Extract map expansion into `data/nuscenes/maps/`:
```bash
unzip nuScenes-map-expansion-v1.3.zip -d data/nuscenes/maps/
```

---

## Running Phase 1

### Full extraction (v1.0-mini)

```bash
python run_extraction.py
# With BEV visualizations:
python run_extraction.py --visualize
```

Writes one JSON per sample to `output/spatial_facts/` and a summary to `output/spatial_facts/_summary.json`.

### CAM_FRONT FOV filter

```bash
python filter_samples.py
# Trainval (after downloading v1.0-trainval_meta.tgz):
python filter_samples.py --version v1.0-trainval --facts-dir output/spatial_facts_trainval
```

Writes `output/curated_samples.json`.

### Trainval yield estimation (without full extraction)

```bash
python estimate_trainval_yield.py
```

Estimates the curated sample count from trainval metadata + map polygons in ~5 minutes, without running the full spatial_facts extraction. Used to decide whether the ~350 GB sensor download is worth it.

---

## Output Format

Each `output/spatial_facts/<sample_token>.json`:

```json
{
  "sample_token": "ca9a282c...",
  "scene_token": "cc8c0bf5...",
  "timestamp": 1532402927647951,
  "ego_pose": { "translation": [x, y, z], "rotation": [w, x, y, z] },
  "num_pedestrians": 2,
  "num_vehicles": 5,
  "num_crosswalks": 1,
  "relations": [
    {
      "subject": "ego",
      "object": "<annotation_token>",
      "object_category": "human.pedestrian.adult",
      "relation": "DC",
      "pair_type": "ego_vs_pedestrian"
    },
    {
      "subject": "<ped_annotation_token>",
      "subject_category": "human.pedestrian.adult",
      "object": "<crosswalk_map_token>",
      "object_category": "ped_crossing",
      "relation": "NTPP",
      "pair_type": "pedestrian_vs_crosswalk"
    }
  ],
  "metadata": {
    "dc_vehicle_pairs_dropped": 12,
    "dc_ped_crosswalk_pairs_dropped": 47
  },
  "config": {
    "max_object_distance": 50.0,
    "crosswalk_search_radius": 80.0,
    "ego_dimensions": [4.084, 1.730],
    "coordinate_frame": "global_bev"
  }
}
```

**Pair types:** `ego_vs_pedestrian`, `pedestrian_vs_crosswalk`, `ego_vs_vehicle`, `vehicle_vs_vehicle`

---

## Key Design Decisions

These decisions are deliberate and should appear in the paper methodology section.

### 1. Coordinate System: BEV Ground Plane in Global Coordinates

All RCC-8 relations are computed on the **bird's-eye-view (BEV) ground plane in nuScenes global coordinates**, not in camera image space.

Reasons:
- The ego vehicle has no corresponding camera mask — it is invisible in all onboard camera images and cannot be represented as an RCC-8 region in image space.
- Crosswalks are map features stored as vector polygons in global coordinates (nuScenes Map Expansion `ped_crossing` layer). They have no image-space annotation.
- Ground-plane topology directly corresponds to driving safety semantics: "is the pedestrian on the crosswalk?" is a ground-plane question.
- Camera perspective distortion would make RCC-8 relations physically ambiguous — a pedestrian occluded by a vehicle would appear to overlap it in image space but is spatially disjoint in the real world.

This is a deliberate design choice, not a default or limitation.

### 2. RCC-8 Classifier Implementation

The classifier uses Shapely DE-9IM predicates via a decision tree:

1. Repair invalid geometries with `buffer(0)`; guard against non-Polygon results (GeometryCollection from degenerate input → return DC)
2. Degenerate area check (`< 1e-8 m²` → DC)
3. `equals_exact(tolerance=1e-6)` → EQ
4. `disjoint()` → DC
5. `touches()` → EC
6. `a.within(b)` → check `a.boundary.intersects(b.boundary)` → TPP (True) or NTPP (False)
7. `b.within(a)` → check `b.boundary.intersects(a.boundary)` → TPPi (True) or NTPPi (False)
8. Otherwise → PO

**Empirically verified:** Shapely's `within()` returns `True` for both TPP and NTPP cases — it is not sufficient for the TPP/NTPP distinction. The `boundary.intersects()` check is required. This behavior was verified empirically during development and is covered by unit tests for all 8 relations, including the geometrically subtle TPP case where shared boundary edges exist.

### 3. Annotation Coordinate Frame

nuScenes `sample_annotation` records store `translation` and `rotation` in **global coordinates already**. No sensor-to-ego transform is required to place pedestrian or vehicle footprints in the global BEV frame. This is distinct from `sample_data` records, which are in sensor frame. The integration test (`test_coordinate_sanity_check`) verifies this by asserting annotation translations are within 50 m of the ego pose for every sample.

### 4. Ego Vehicle Representation

The ego vehicle footprint is a rotated rectangle with dimensions **4.084 m × 1.730 m** (Renault Zoe, the nuScenes data collection vehicle). Yaw is extracted from the ego pose quaternion via `q.yaw_pitch_roll[0]`. The footprint is placed in global BEV coordinates from the ego pose translation.

### 5. CAM_FRONT Selection and FOV Computation

Phase 2/3 evaluation uses **CAM_FRONT only**. The justification for the paper:

> *"We restrict evaluation to scenes where safety-critical pedestrians are within the CAM_FRONT field of view, ensuring the baseline VLM has access to the relevant visual information and isolating RCC-8 grounding as the sole experimental variable."*

The CAM_FRONT half-FOV is computed **per sample** from the `calibrated_sensor` intrinsic matrix:

```
half_fov = arctan(image_width / (2 × fx))
```

where `fx = camera_intrinsic[0][0]` and `image_width` is read from the `sample_data` record. This is not hardcoded. The measured value from v1.0-mini calibration data:

| Stat | Value |
|------|-------|
| Min | 32.28° |
| Mean | **32.42°** |
| Max | 32.56° |

Note: this is ~65° full FOV — slightly narrower than the commonly cited "~70°" figure. Computing from calibration rather than assuming a nominal value is the methodologically correct approach and removes a potential reviewer question about whether the FOV threshold was tuned to produce a desired sample count. Variance across all mini samples is <0.3°, confirming the sensor hardware is consistent across logs and that the fallback constant (32.42°) is safe if the calibration lookup ever fails.

### 6. DC Pair Filtering

**Vehicle-vs-vehicle** and **pedestrian-vs-crosswalk** DC pairs are dropped from output. Distant vehicles being disconnected from each other is spatially uninformative for VLM prompts. The drop count is logged in each JSON's `metadata` field for auditability. Distance thresholds applied before RCC-8 computation:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_object_distance` | 50 m | Ignore annotations farther than this from ego |
| `crosswalk_search_radius` | 80 m | Map expansion query radius for `ped_crossing` records |
| `vehicle_pair_max_distance` | 20 m | Only compute vehicle-vehicle RCC-8 for nearby pairs |

### 7. Relation Distribution in Real Data

Only **DC, NTPP, and PO** appear in the v1.0-mini dataset (404 samples, 12,834 relations). EC, EQ, TPP, TPPi, and NTPPi do not occur. This is expected and physically meaningful: GPS/sensor-fused BEV polygon localization has enough noise that exact boundary contact never arises in practice. The classifier logic for all 8 relations is verified by unit tests; they simply do not arise with real nuScenes sensor data.

### 8. Dataset Splits and Crosswalk Coverage

Phase 1 was developed and validated on **v1.0-mini**, then yield-estimated on **v1.0-trainval** metadata.

| Split | Samples | With peds + crosswalks | Curated (CAM_FRONT FOV) | Yield |
|-------|---------|----------------------|------------------------|-------|
| v1.0-mini | 404 | 285 | **59** | 14.6% |
| v1.0-trainval (est.) | 34,149 | — | **~3,398** | 10.0% |

The yield rate difference is expected — mini scenes were hand-selected for interesting/busy urban environments; trainval includes quiet roads and highway segments. The trainval estimate uses a point-in-polygon centroid proxy rather than full Shapely footprints, so ~3,398 is approximate.

Mini pipeline summary (from `output/spatial_facts/_summary.json`):
- 12,834 total relations computed
- 53,167 vehicle-vehicle DC pairs dropped
- 341/404 samples have crosswalks nearby; 313/404 have pedestrians; 285/404 have both

---

## Handoff to Phase 2

**Primary artifact:** `output/curated_samples.json`

Contains:
- `kept`: 59 sample tokens where a safety-critical pedestrian (NTPP or PO with a crosswalk) is within CAM_FRONT FOV — use these for prompting experiments
- `dropped`: samples with qualifying pedestrians but outside CAM_FRONT FOV
- `no_critical_pedestrians`: samples with no pedestrian-crosswalk NTPP/PO relations

For each kept sample token, Person 2 should:
1. Load the corresponding `output/spatial_facts/<token>.json` for RCC-8 relations
2. Load the CAM_FRONT image via `nusc.get_sample_data_path(sample['data']['CAM_FRONT'])`
3. Filter `relations` by `pair_type == "pedestrian_vs_crosswalk"` and `relation in ("NTPP", "PO")` to find the safety-critical constraint to inject into the prompt

See `dummy/spatial_facts_dummy.json` for annotated examples of all relation types and how to interpret them for prompt construction.

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `rcc_drive/config.py` | All constants and thresholds |
| `rcc_drive/nuscenes_loader.py` | nuScenes SDK wrapper (loader, map queries) |
| `rcc_drive/region_extractor.py` | Annotations → Shapely BEV polygons |
| `rcc_drive/rcc8_classifier.py` | RCC-8 decision tree via Shapely predicates |
| `rcc_drive/spatial_fact_generator.py` | Orchestrator → per-sample JSON |
| `rcc_drive/visualizer.py` | BEV debug plots (matplotlib) |
| `run_extraction.py` | CLI: run full pipeline on all scenes |
| `filter_samples.py` | CLI: filter by CAM_FRONT FOV visibility |
| `estimate_trainval_yield.py` | Estimate trainval curated count (metadata only) |
