"""BEV (bird's eye view) visualization for spatial fact debugging."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch
from pyquaternion import Quaternion


# Color scheme
COLORS = {
    "ego": "#2196F3",
    "pedestrian": "#F44336",
    "vehicle": "#4CAF50",
    "crosswalk": "#FFC107",
}


def _polygon_to_patch(shapely_poly, **kwargs):
    """Convert a Shapely polygon to a matplotlib patch."""
    coords = np.array(shapely_poly.exterior.coords)
    return MplPolygon(coords, **kwargs)


def plot_sample_bev(
    regions: dict,
    relations: list,
    ego_pose: dict,
    sample_token: str = "",
    output_path: str = None,
    view_radius: float = 60.0,
):
    """Render a BEV plot showing all polygons and RCC-8 relations.

    Args:
        regions: Dict from RegionExtractor.extract_all_regions() with
                 'ego', 'pedestrians', 'vehicles', 'crosswalks'.
        relations: List of relation dicts from SpatialFactGenerator.
        ego_pose: Ego pose dict with 'translation' and 'rotation'.
        sample_token: For the plot title.
        output_path: If provided, save the figure here.
        view_radius: Half-width of the view window in meters.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    ego_x, ego_y = ego_pose["translation"][0], ego_pose["translation"][1]

    # Draw crosswalks first (background layer)
    for cw in regions["crosswalks"]:
        patch = _polygon_to_patch(
            cw["polygon"],
            facecolor=COLORS["crosswalk"],
            edgecolor="#E6A800",
            alpha=0.3,
            linewidth=1.5,
            hatch="//",
            label="_nolegend_",
        )
        ax.add_patch(patch)
        cx, cy = cw["polygon"].centroid.x, cw["polygon"].centroid.y
        ax.text(cx, cy, "CW", fontsize=7, ha="center", va="center", color="#8B6914")

    # Draw vehicles
    for veh in regions["vehicles"]:
        patch = _polygon_to_patch(
            veh["polygon"],
            facecolor=COLORS["vehicle"],
            edgecolor="#2E7D32",
            alpha=0.5,
            linewidth=1.5,
        )
        ax.add_patch(patch)
        cx, cy = veh["polygon"].centroid.x, veh["polygon"].centroid.y
        short_cat = veh["category"].split(".")[-1]
        ax.text(cx, cy, short_cat, fontsize=6, ha="center", va="center", color="white",
                fontweight="bold")

    # Draw pedestrians
    for ped in regions["pedestrians"]:
        patch = _polygon_to_patch(
            ped["polygon"],
            facecolor=COLORS["pedestrian"],
            edgecolor="#C62828",
            alpha=0.7,
            linewidth=1.5,
        )
        ax.add_patch(patch)
        cx, cy = ped["polygon"].centroid.x, ped["polygon"].centroid.y
        ax.plot(cx, cy, "o", color="white", markersize=3)

    # Draw ego vehicle
    ego_patch = _polygon_to_patch(
        regions["ego"],
        facecolor=COLORS["ego"],
        edgecolor="#1565C0",
        alpha=0.7,
        linewidth=2,
    )
    ax.add_patch(ego_patch)
    ax.text(ego_x, ego_y, "EGO", fontsize=8, ha="center", va="center",
            color="white", fontweight="bold")

    # Draw ego heading arrow
    q = Quaternion(ego_pose["rotation"])
    yaw = q.yaw_pitch_roll[0]
    arrow_len = 5.0
    dx, dy = arrow_len * np.cos(yaw), arrow_len * np.sin(yaw)
    ax.annotate(
        "",
        xy=(ego_x + dx, ego_y + dy),
        xytext=(ego_x, ego_y),
        arrowprops=dict(arrowstyle="->", color=COLORS["ego"], lw=2.5),
    )

    # Draw relation lines for non-DC pairs
    # Build a lookup of token -> centroid
    centroids = {"ego": (ego_x, ego_y)}
    for ped in regions["pedestrians"]:
        c = ped["polygon"].centroid
        centroids[ped["token"]] = (c.x, c.y)
    for veh in regions["vehicles"]:
        c = veh["polygon"].centroid
        centroids[veh["token"]] = (c.x, c.y)
    for cw in regions["crosswalks"]:
        c = cw["polygon"].centroid
        centroids[cw["token"]] = (c.x, c.y)

    for rel in relations:
        if rel["relation"] == "DC":
            continue
        subj_key = rel["subject"]
        obj_key = rel["object"]
        if subj_key in centroids and obj_key in centroids:
            sx, sy = centroids[subj_key]
            ox, oy = centroids[obj_key]
            mx, my = (sx + ox) / 2, (sy + oy) / 2
            ax.plot([sx, ox], [sy, oy], "--", color="gray", alpha=0.6, linewidth=1)
            ax.text(
                mx, my, rel["relation"],
                fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                fontweight="bold",
            )

    # Formatting
    ax.set_xlim(ego_x - view_radius, ego_x + view_radius)
    ax.set_ylim(ego_y - view_radius, ego_y + view_radius)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    title = f"BEV Spatial Facts"
    if sample_token:
        title += f"\n{sample_token[:16]}..."
    ax.set_title(title, fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=COLORS["ego"], alpha=0.7, label="Ego Vehicle"),
        Patch(facecolor=COLORS["pedestrian"], alpha=0.7, label="Pedestrian"),
        Patch(facecolor=COLORS["vehicle"], alpha=0.5, label="Vehicle"),
        Patch(facecolor=COLORS["crosswalk"], alpha=0.3, label="Crosswalk"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)

    plt.tight_layout()

    if output_path:
        dirpath = os.path.dirname(output_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return output_path
