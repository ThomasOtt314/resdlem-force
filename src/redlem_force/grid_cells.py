"""Grid-cell selection for the PRISM precip sources (800 m and 4 km).

Built locally from the grid definition (CRS + affine transform of one image,
grid_from_ee); no Earth Engine compute. Same idea as the URMA cells, without
any water mask: each reservoir is reduced over whole pixels of the source's
own grid, each counted once.

Rule, per reservoir and grid (select):
  * candidates: every pixel the polygon shares area with (pixels that only
    touch it along an edge or at a corner are not candidates), each with
    its overlap area (areas in the grid's CRS)
  * take candidates in order of overlap, largest first, while the total
    area of the pixels taken is less than the polygon's area. That is
    ceil(polygon area / pixel area) pixels: at least the polygon's area,
    over by less than one pixel. A reservoir smaller than one pixel gets
    the one pixel it overlaps most.
Stored as the union of the selected pixel squares -- always a polygon (one
square for a single cell), like the URMA cells.

Sent to Earth Engine as that polygon (to_wgs84_geojson). Extraction
reduces it with an UNWEIGHTED mean in the grid's projection: a pixel is
used when its centre is inside the polygon, so it gives exactly the
selected pixels (a single cell's square holds only its own pixel centre).
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pyproj
import shapely

from .geometry import GeometrySet, reproject, to_geojson


_PARAM = r'PARAMETER\[\s*"{}"\s*,\s*([-+0-9.eE]+)\s*\]'


def local_crs(text: str) -> str:
    """The CRS to use with pyproj for a CRS text from Earth Engine, so local
    conversions match Earth Engine's.

    Earth Engine's URMA CRS ("NWS CONUS") is WKT1 with a WGS 84 GEOGCS but
    PARAMETER["semi_major", 6371200] / ["semi_minor", 6371200] inside the
    projection. Earth Engine projects on that sphere; pyproj ignores the two
    parameters and uses the WGS 84 ellipsoid -- up to 8 km (2-3 pixels)
    apart (checked 2026-10-15: the sphere matched Earth Engine to 0.000 m
    on 40 points; pyproj's reading was off by up to 8,009 m). When those parameters are present, the same
    projection is rebuilt on the a / b they give. Any other CRS text is
    returned unchanged."""
    ma = re.search(_PARAM.format('semi_major'), text)
    mb = re.search(_PARAM.format('semi_minor'), text)
    if not (ma and mb):
        return text
    with warnings.catch_warnings():       # to_dict warns about PROJ strings
        warnings.simplefilter('ignore')
        d = pyproj.CRS.from_user_input(text).to_dict()
    for k in ('datum', 'ellps', 'towgs84', 'R', 'a', 'b', 'rf', 'f', 'type'):
        d.pop(k, None)
    d['a'], d['b'] = float(ma.group(1)), float(mb.group(1))
    return pyproj.CRS.from_dict(d).srs


@dataclass(frozen=True)
class Grid:
    crs: str          # CRS for local (pyproj) work -- see local_crs()
    a: float          # pixel width (x scale)
    c: float          # x of the grid's left edge
    e: float          # pixel height (y scale, negative for north-up)
    f: float          # y of the grid's top edge
    ee_crs: str = ''  # the CRS text exactly as Earth Engine reports it

    @classmethod
    def from_projection(cls, proj: dict) -> 'Grid':
        """From ee.Image.projection().getInfo()."""
        if 'transform' not in proj:
            raise ValueError(f'projection has no affine transform: {proj}')
        a, b, c, d, e, f = proj['transform']
        if b != 0 or d != 0:
            raise ValueError(f'rotated grids are not supported: {proj}')
        text = proj.get('crs') or proj.get('wkt')
        if not text:
            raise ValueError(f'projection has no CRS: {proj}')
        return cls(local_crs(text), float(a), float(c), float(e), float(f),
                   text)

    @property
    def pixel_area(self) -> float:
        return abs(self.a * self.e)

    def col_row(self, x, y):
        return math.floor((x - self.c) / self.a), \
            math.floor((y - self.f) / self.e)

    def centre(self, col, row):
        return (self.c + (col + 0.5) * self.a,
                self.f + (row + 0.5) * self.e)

    def squares(self, cols, rows):
        """Pixel squares (vectorized) for arrays of col / row."""
        xa = self.c + cols * self.a
        xb = self.c + (cols + 1) * self.a
        ya = self.f + rows * self.e
        yb = self.f + (rows + 1) * self.e
        return shapely.box(np.minimum(xa, xb), np.minimum(ya, yb),
                           np.maximum(xa, xb), np.maximum(ya, yb))


def overlaps(geom, grid: Grid):
    """Every pixel sharing area with geom (geom in the grid CRS), with its
    overlap area. Returns (cols, rows, overlap) arrays.

    Pixels that only touch geom along an edge or at a corner are left out.
    Pixels fully inside geom get the pixel area without an intersection;
    only pixels on the boundary are intersected."""
    minx, miny, maxx, maxy = geom.bounds
    c0, r0 = grid.col_row(minx, maxy)
    c1, r1 = grid.col_row(maxx, miny)
    cc, rr = np.meshgrid(np.arange(min(c0, c1), max(c0, c1) + 1),
                         np.arange(min(r0, r1), max(r0, r1) + 1))
    cc, rr = cc.ravel(), rr.ravel()
    sq = grid.squares(cc, rr)
    shapely.prepare(geom)
    hit = shapely.intersects(geom, sq) & ~shapely.touches(geom, sq)
    cc, rr, sq = cc[hit], rr[hit], sq[hit]
    area = np.full(len(sq), grid.pixel_area)
    edge = ~shapely.contains_properly(geom, sq)
    if edge.any():
        area[edge] = shapely.area(shapely.intersection(geom, sq[edge]))
    keep = area > 0
    return cc[keep], rr[keep], area[keep]


def select(geom, pt, grid: Grid):
    """Cells for one polygon (geom in the grid CRS).

    Pixels are taken in order of overlap with the polygon, largest first,
    while the total area of the pixels taken is still less than the
    polygon's area: k = ceil(polygon area / pixel area) pixels, so the
    selection covers at least the polygon's area and goes over by less
    than one pixel. Fully covered pixels always come first. Ties between
    equal overlaps go to the lower row, then the lower column.

    pt (internal point, grid CRS) is only used if no pixel shares area
    with the polygon (not expected). Returns (list of (col, row),
    single_cell)."""
    cc, rr, ov = overlaps(geom, grid)
    if len(cc) == 0:                       # not expected; be safe
        return [grid.col_row(pt.x, pt.y)], True
    k = max(1, math.ceil(shapely.area(geom) / grid.pixel_area - 1e-9))
    k = min(k, len(cc))
    order = np.lexsort((cc, rr, -ov))[:k]  # last key sorts first
    cells = list(zip(cc[order].tolist(), rr[order].tolist()))
    return cells, len(cells) == 1


def cell_geometry(cells, grid: Grid):
    """Union of the selected pixel squares (exact pixel edges); one square
    for a single cell. Always a valid (Multi)Polygon; raises if not.

    coverage_union_all (fast) first. Some GEOS versions return an invalid
    ring there when cells touch only at a corner around a gap (a ring that
    touches itself at a point); then the squares are joined again with
    union_all (overlay), which returns valid geometry."""
    cols = np.array([c for c, _ in cells])
    rows = np.array([r for _, r in cells])
    sq = grid.squares(cols, rows)
    try:
        g = shapely.coverage_union_all(sq)
    except (AttributeError, shapely.errors.GEOSException):
        g = None
    if g is None or not g.is_valid:
        g = shapely.union_all(sq)
    if not g.is_valid:
        raise RuntimeError(
            f'invalid cell polygon built for cells {cells}: '
            f'{shapely.is_valid_reason(g)} (shapely {shapely.__version__}, '
            f'GEOS {shapely.geos_version_string})')
    return g


def build(gs: GeometrySet, points_wgs84: np.ndarray, grid: Grid):
    """Cells for every polygon in gs.

    points_wgs84: internal points (EPSG:4326), one per polygon.
    Returns (polygons in the GRID CRS as an object array, n_cells array,
    single_cell array -- True where one pixel was selected), in gs.ids
    order.
    """
    polys = reproject(gs.geoms, gs.crs, grid.crs)
    pts = reproject(points_wgs84, 'EPSG:4326', grid.crs)
    geoms, n, is_pt = [], [], []
    for g, p in zip(polys, pts):
        cells, point = select(g, p, grid)
        geoms.append(cell_geometry(cells, grid))
        n.append(len(cells))
        is_pt.append(point)
    return (np.array(geoms, dtype=object), np.array(n, dtype=np.int64),
            np.array(is_pt, dtype=bool))


def to_wgs84_geojson(geoms_grid: np.ndarray, grid: Grid) -> list:
    return to_geojson(reproject(geoms_grid, grid.crs, 'EPSG:4326'))
