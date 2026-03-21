"""RCC-8 classifier using Shapely topological predicates.

Classifies the spatial relation between two polygons into one of the 8
Region Connection Calculus relations: DC, EC, PO, EQ, TPP, TPPi, NTPP, NTPPi.
"""

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

from rcc_drive.config import DEFAULT_CONFIG


def _ensure_polygon(geom):
    """Ensure geometry is a valid Polygon. Extract largest from collections."""
    if isinstance(geom, Polygon) and not geom.is_empty:
        return geom
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polygons = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if polygons:
            return max(polygons, key=lambda p: p.area)
    return None


def classify_rcc8(region_a, region_b, tolerance=None):
    """Classify the RCC-8 relation between region_a and region_b.

    The relation is read as "A <relation> B":
      - DC:    A is disconnected from B
      - EC:    A is externally connected to B (boundaries touch, interiors disjoint)
      - PO:    A partially overlaps B
      - EQ:    A equals B
      - TPP:   A is a tangential proper part of B (A inside B, boundaries touch)
      - NTPP:  A is a non-tangential proper part of B (A inside B, no boundary touch)
      - TPPi:  B is a tangential proper part of A (inverse of TPP)
      - NTPPi: B is a non-tangential proper part of A (inverse of NTPP)

    Args:
        region_a: Shapely Polygon
        region_b: Shapely Polygon
        tolerance: Tolerance for equals_exact comparison. Defaults to config value.

    Returns:
        One of: 'DC', 'EC', 'PO', 'EQ', 'TPP', 'TPPi', 'NTPP', 'NTPPi'
    """
    if tolerance is None:
        tolerance = DEFAULT_CONFIG["boundary_tolerance"]

    min_area = DEFAULT_CONFIG["min_polygon_area"]

    # Fix invalid geometries
    a = region_a.buffer(0) if not region_a.is_valid else region_a
    b = region_b.buffer(0) if not region_b.is_valid else region_b

    # Guard: buffer(0) can return non-Polygon types
    a = _ensure_polygon(a)
    b = _ensure_polygon(b)

    if a is None or b is None:
        return "DC"

    # Degenerate polygons (near-zero area)
    if a.area < min_area or b.area < min_area:
        return "DC"

    # EQ: regions are equal
    if a.equals_exact(b, tolerance=tolerance):
        return "EQ"

    # DC: regions are disjoint
    if a.disjoint(b):
        return "DC"

    # EC: boundaries touch but interiors do not intersect
    if a.touches(b):
        return "EC"

    # Containment: A within B
    if a.within(b):
        if a.boundary.intersects(b.boundary):
            return "TPP"
        return "NTPP"

    # Containment: B within A
    if b.within(a):
        if b.boundary.intersects(a.boundary):
            return "TPPi"
        return "NTPPi"

    # PO: partial overlap (interiors intersect, neither contains the other)
    return "PO"
