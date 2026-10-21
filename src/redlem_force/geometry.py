"""Geometry input, repair, reprojection and the internal-centroid latitude.

Geometries arrive in a polars DataFrame column as WKB (bytes) or WKT (str),
with the CRS given separately. Nothing here talks to Earth Engine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
import pyproj
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polylabel

log = logging.getLogger('redlem_force')

# Projected CRS used for centroids when the input CRS is geographic.
# CONUS Albers, same CRS as revap_master_v6.
CENTROID_CRS = 'EPSG:5070'
# polylabel precision, in units of the projected CRS (meters for 5070).
POLYLABEL_TOL = 10.0


_HEX = frozenset('0123456789abcdefABCDEF')


def _parse_one(value):
    """One geometry value: a shapely geometry, WKB / EWKB bytes, hex WKB /
    EWKB text (what PostGIS returns), or WKT text."""
    if value is None:
        return None
    if isinstance(value, shapely.Geometry):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return shapely.from_wkb(bytes(value))
    if isinstance(value, str):
        v = value.strip()
        if v and set(v) <= _HEX:                 # hex WKB / EWKB
            return shapely.from_wkb(v)
        return shapely.from_wkt(v)
    raise TypeError(f'Geometry must be a shapely geometry, WKB bytes, hex '
                    f'WKB or WKT text, got {type(value).__name__}')


def polygon_parts(g) -> list:
    """Polygon parts only; make_valid() can return collections with stray
    lines or points."""
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if hasattr(g, 'geoms'):
        out = []
        for part in g.geoms:
            out.extend(polygon_parts(part))
        return out
    return []


def to_polygonal(g):
    """Polygon or MultiPolygon from any geometry; None if nothing polygonal."""
    parts = polygon_parts(g)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


@dataclass
class GeometrySet:
    ids: list           # str IDs, input order
    geoms: np.ndarray   # shapely geometries in `crs`
    crs: pyproj.CRS
    repaired_ids: list  # IDs whose geometry was repaired with make_valid


def ids_as_str(df: pl.DataFrame, id_col: str) -> list:
    """ID column as a list of str; rejects nulls and duplicates."""
    if id_col not in df.columns:
        raise KeyError(f'ID column {id_col!r} not in {df.columns}')
    ids = df.get_column(id_col).cast(pl.Utf8)
    if ids.null_count():
        raise ValueError(f'{ids.null_count()} null IDs in {id_col!r}')
    dup = ids.filter(ids.is_duplicated()).unique().to_list()
    if dup:
        raise ValueError(f'{len(dup)} duplicate IDs in {id_col!r}, '
                         f'e.g. {dup[:10]}')
    return ids.to_list()


def load_geometries(df: pl.DataFrame, id_col: str, geometry_col: str,
                    crs, label: str | None = None,
                    expect_valid: bool = False) -> GeometrySet:
    """Parse, validate and repair geometries.

    - IDs must be non-null and unique (cast to str).
    - Geometry must be non-null, non-empty and polygonal.
    - Invalid geometries are repaired with shapely.make_valid(), keeping only
      polygon parts; repaired IDs are logged and returned.
    label        : what the geometries are, for the log ('native reservoir',
                   'URMA cell', ...); default the column name
    expect_valid : True for cell polygons built by redlem-force (always
                   valid): a repair is logged as a WARNING, since it means
                   the file did not come from redlem-force or was altered
    """
    ids = ids_as_str(df, id_col)
    if geometry_col not in df.columns:
        raise KeyError(f'Geometry column {geometry_col!r} not in {df.columns}')

    geoms = np.empty(len(ids), dtype=object)
    for i, v in enumerate(df.get_column(geometry_col).to_list()):
        geoms[i] = _parse_one(v)

    missing = [ids[i] for i, g in enumerate(geoms) if g is None or g.is_empty]
    if missing:
        raise ValueError(f'{len(missing)} missing/empty geometries, '
                         f'e.g. {missing[:10]}')

    invalid = np.flatnonzero(~shapely.is_valid(geoms))
    repaired = []
    lost = []
    for i in invalid:
        fixed = to_polygonal(shapely.make_valid(geoms[i]))
        if fixed is None:
            lost.append(ids[i])
        else:
            geoms[i] = fixed
            repaired.append(ids[i])
    if lost:
        raise ValueError(f'{len(lost)} geometries had no polygon left after '
                         f'repair: {lost[:10]}')
    if repaired:
        what = label or repr(geometry_col)
        if expect_valid:
            log.warning('repaired %d invalid %s geometries with make_valid '
                        '(not expected: redlem-force builds valid cell '
                        'polygons), e.g. %s', len(repaired), what,
                        repaired[:5])
        else:
            log.info('repaired %d invalid %s geometries with make_valid, '
                     'e.g. %s', len(repaired), what, repaired[:5])

    nonpoly = [ids[i] for i, g in enumerate(geoms)
               if not isinstance(g, (Polygon, MultiPolygon))]
    if nonpoly:
        raise ValueError(f'{len(nonpoly)} non-polygon geometries, '
                         f'e.g. {nonpoly[:10]}')

    return GeometrySet(ids, geoms, pyproj.CRS.from_user_input(crs), repaired)


def reproject(geoms: np.ndarray, src, dst) -> np.ndarray:
    """Reproject an array of shapely geometries."""
    src, dst = pyproj.CRS.from_user_input(src), pyproj.CRS.from_user_input(dst)
    if src == dst:
        return geoms
    tr = pyproj.Transformer.from_crs(src, dst, always_xy=True)

    def f(coords):
        x, y = tr.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geoms, f)


def internal_points(gs: GeometrySet) -> np.ndarray:
    """A point guaranteed inside each polygon, in EPSG:4326 (lon/lat).

    Hybrid method (verified on all 44,751 revap_master_v6 reservoirs, 100%
    inside): the area centroid if it falls inside the polygon, otherwise the
    pole of inaccessibility (polylabel) of the largest part. Computed in a
    projected CRS; geographic input is projected to EPSG:5070 first.
    """
    if gs.crs.is_geographic:
        work = reproject(gs.geoms, gs.crs, CENTROID_CRS)
        work_crs = pyproj.CRS.from_user_input(CENTROID_CRS)
    else:
        work, work_crs = gs.geoms, gs.crs

    pts = shapely.centroid(work)
    inside = shapely.within(pts, work)
    for i in np.flatnonzero(~inside):
        part = max(polygon_parts(work[i]), key=lambda p: p.area)
        pts[i] = polylabel(part, tolerance=POLYLABEL_TOL)

    ok = shapely.within(pts, work)
    if not ok.all():
        bad = [gs.ids[i] for i in np.flatnonzero(~ok)]
        raise RuntimeError(f'internal point not inside polygon for '
                           f'{len(bad)} IDs: {bad[:10]}')
    return reproject(pts, work_crs, 'EPSG:4326')


def internal_point_lat(gs: GeometrySet) -> np.ndarray:
    """Latitude (deg) of internal_points()."""
    return shapely.get_y(internal_points(gs))


def to_geojson(geoms_wgs84: np.ndarray) -> list:
    """GeoJSON geometry dicts (plain lists, safe for Earth Engine)."""
    return [json.loads(shapely.to_geojson(g)) for g in geoms_wgs84]


def chunked(seq: list, n: int) -> list:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def geojson_bytes(geom: dict) -> int:
    """Approximate request size of one GeoJSON geometry (compact JSON)."""
    return len(json.dumps(geom, separators=(',', ':')))


def chunk_by_size(ids: list, sizes: dict, max_items: int,
                  max_bytes: int) -> list:
    """Split ids (order kept) into chunks of at most max_items whose summed
    sizes stay within max_bytes. sizes maps id -> bytes, or id -> tuple of
    bytes per request (e.g. (native, cells)); every component must fit.
    An id that alone exceeds max_bytes gets a chunk of its own."""
    chunks, cur, tot = [], [], None
    for i in ids:
        sz = sizes[i] if isinstance(sizes[i], tuple) else (sizes[i],)
        if tot is None:
            tot = [0] * len(sz)
        over = any(t + x > max_bytes for t, x in zip(tot, sz))
        if cur and (len(cur) >= max_items or over):
            chunks.append(cur)
            cur, tot = [], [0] * len(sz)
        cur.append(i)
        tot = [t + x for t, x in zip(tot, sz)]
    if cur:
        chunks.append(cur)
    return chunks
