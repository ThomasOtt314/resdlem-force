"""URMA cell selection: native reservoir polygon -> the URMA grid cells used
for the met variables.

Everything is computed locally on the URMA grid of the reference image
(resdlem_force.grids: the first URMA daily image of 2026). Earth Engine is
used for two things only:
  * the grid definition (CRS + affine transform), one small request
  * the URMA elevation and water mask read ONCE as arrays on that same grid
    (they sit on the same lattice; checked 2026-10-13), one value
    per URMA cell
No grid-cell asset and no per-reservoir Earth Engine requests.

Selection rule (same logic as urma_cells.py; cell values are now the pixel
values of the cell itself):
  * candidate cells = URMA cells the polygon intersects (sharing area;
    touching only at an edge / corner does not count)
  * water-mask flag: a column you pass, or the rule max(water mask of the
    candidate cells) == 1 (urma_cells.py used the max at 100 m over the
    polygon, i.e. the cells it intersects)
  * flag 1 -> candidates whose water mask is 1; if there are none, the rule
    below (cell_method = water_mask_fallback)
  * otherwise -> candidates with overlap area >= the median overlap, then of
    those, the ones with elevation <= their median elevation
    (overlap_elevation). Medians: numpy median of the candidates' values.
  * output geometry = union of the selected cells (exact pixel squares)

Output columns:
  <id_col>      str
  cell_geometry WKB, EPSG:4326
  n_cells       cells selected
  n_touching    candidate cells (intersecting the reservoir)
  wm_value      flag value used (rule: max water mask of the candidates;
                null = no water-mask data there)
  cell_method   water_mask | water_mask_fallback | overlap_elevation
  cell_elev_m   mean elevation of the selected cells (-> ELEV_M)
"""

from __future__ import annotations

import logging
import os

import ee
import numpy as np
import polars as pl
import shapely

from . import grids
from .config import ELEV_ID, WM_ID, RunConfig
from .ee_utils import init_ee, with_retries
from .geometry import load_geometries, reproject
from .grid_cells import Grid, cell_geometry

log = logging.getLogger('resdlem_force')

CELL_SCHEMA = {
    'cell_geometry': pl.Binary,
    'n_cells': pl.Int64,
    'n_touching': pl.Int64,
    'wm_value': pl.Float64,
    'cell_method': pl.Utf8,
    'cell_elev_m': pl.Float64,
}
ROWS_PER_REQUEST = 256        # computePixels tile height (rows of the grid)


# ---------------------------------------------------------------------------
# Per-cell values on the URMA grid (one download per run)
# ---------------------------------------------------------------------------
class CellValues:
    """URMA elevation / water mask per URMA cell, as arrays indexed by the
    reference grid's (col, row). NaN = no data."""

    def __init__(self, grid: Grid, col0, row0, elev, wm):
        self.grid, self.col0, self.row0 = grid, col0, row0
        self.elev, self.wm = elev, wm            # [row - row0, col - col0]

    def lookup(self, cols, rows):
        r = np.asarray(rows) - self.row0
        c = np.asarray(cols) - self.col0
        ok = (r >= 0) & (r < self.elev.shape[0]) & (c >= 0) & \
            (c < self.elev.shape[1])
        e = np.full(len(r), np.nan)
        w = np.full(len(r), np.nan)
        e[ok] = self.elev[r[ok], c[ok]]
        w[ok] = self.wm[r[ok], c[ok]]
        return e, w, ok


def _value_image():
    elev = ee.Image(ELEV_ID).select('elevation')       # m
    wm = ee.Image(WM_ID).select(0)
    # data values, and 1/0 masks so missing pixels can be told apart
    return ee.Image.cat([
        elev.unmask(0, False).toFloat().rename('elev'),
        elev.mask().unmask(0, False).toFloat().rename('elev_ok'),
        wm.unmask(0, False).toFloat().rename('wm'),
        wm.mask().unmask(0, False).toFloat().rename('wm_ok'),
    ])


def fetch_cell_values(config: RunConfig) -> CellValues:
    """Read URMA elevation / water mask on the URMA reference grid, over the
    reference image's pixel extent, in tiles of ROWS_PER_REQUEST rows."""
    info = grids.grid_info('urma', config)
    grid, w, h = info['grid'], info['width'], info['height']
    col0, row0 = info['col0'], info['row0']
    if not w or not h:
        raise RuntimeError(f'URMA reference image has no dimensions: {info}')
    init_ee(config.project, config.base_url)
    img = _value_image()
    crs = grid.ee_crs or grid.crs      # Earth Engine reads its own text
    elev = np.empty((h, w), dtype=np.float64)
    wm = np.empty((h, w), dtype=np.float64)
    for r in range(0, h, ROWS_PER_REQUEST):
        n = min(ROWS_PER_REQUEST, h - r)
        req = {
            'expression': img,
            'fileFormat': 'NUMPY_NDARRAY',
            'grid': {
                'dimensions': {'width': w, 'height': n},
                'affineTransform': {
                    'scaleX': grid.a, 'shearX': 0,
                    'translateX': grid.c + col0 * grid.a,
                    'shearY': 0, 'scaleY': grid.e,
                    'translateY': grid.f + (row0 + r) * grid.e},
                'crsWkt' if crs.lstrip().upper().startswith(
                    ('PROJCS', 'GEOGCS', 'PROJCRS', 'GEOGCRS'))
                else 'crsCode': crs,
            },
        }
        arr = with_retries(lambda: ee.data.computePixels(req),
                           config.max_retries, config.retry_wait_secs)
        elev[r:r + n] = np.where(arr['elev_ok'] > 0, arr['elev'], np.nan)
        wm[r:r + n] = np.where(arr['wm_ok'] > 0, arr['wm'], np.nan)
    log.debug('URMA cell values: %d x %d cells; elevation for %d, water mask '
             'for %d (%d water cells)', w, h, int(np.isfinite(elev).sum()),
             int(np.isfinite(wm).sum()), int((wm == 1).sum()))
    return CellValues(grid, col0, row0, elev, wm)


# ---------------------------------------------------------------------------
# Selection (local)
# ---------------------------------------------------------------------------
def candidates(geom, grid: Grid):
    """(cols, rows, overlap areas) of cells geom intersects (grid CRS)."""
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
    area = shapely.area(shapely.intersection(geom, sq))
    keep = area > 0
    return cc[keep], rr[keep], area[keep]


def select_one(geom, values: CellValues, flag=None):
    """Selection for one polygon in the grid CRS. flag: 0/1 from a column,
    or None for the rule. Returns a dict of CELL_SCHEMA values (geometry in
    the grid CRS, not yet WKB), plus 'cells': the selected (col, row)
    pairs when any were selected."""
    grid = values.grid
    cols, rows, area = candidates(geom, grid)
    n_touch = len(cols)
    if n_touch == 0:
        return {'geom': None, 'n_cells': 0, 'n_touching': 0,
                'wm_value': None if flag is None else float(flag),
                'cell_method': None, 'cell_elev_m': None}
    elev, wm, _ = values.lookup(cols, rows)
    if flag is None:
        wm_value = float(np.nanmax(wm)) if np.isfinite(wm).any() else None
    else:
        wm_value = float(flag)
    use_wm = wm_value == 1

    water = wm == 1
    if use_wm and water.any():
        sel, method = water, 'water_mask'
    else:
        method = 'water_mask_fallback' if use_wm else 'overlap_elevation'
        sel = area >= np.median(area)
        e = elev[sel]
        if np.isfinite(e).any():
            cut = np.median(e[np.isfinite(e)])
            sub = np.isfinite(elev) & (elev <= cut)
            sel = sel & sub
    n = int(sel.sum())
    if n == 0:
        return {'geom': None, 'n_cells': 0, 'n_touching': n_touch,
                'wm_value': wm_value, 'cell_method': method,
                'cell_elev_m': None}
    g = cell_geometry(list(zip(cols[sel].tolist(), rows[sel].tolist())),
                      grid)                          # valid, or raises
    e = elev[sel]
    return {'geom': g, 'cells': list(zip(cols[sel].tolist(),
                                         rows[sel].tolist())),
            'n_cells': n, 'n_touching': n_touch,
            'wm_value': wm_value, 'cell_method': method,
            'cell_elev_m': float(np.nanmean(e)) if np.isfinite(e).any()
            else None}


def _log_summary(df: pl.DataFrame, id_col: str):
    zero = df.filter(pl.col('cell_geometry').is_null())[id_col].to_list()
    if zero:
        log.warning('%d reservoirs got zero cells: %s', len(zero), zero[:20])
    fb = df.filter(pl.col('cell_method') == 'water_mask_fallback')[id_col]
    if len(fb):
        log.warning('%d water-mask reservoirs had no water cells and used the '
                    'overlap/elevation rule: %s', len(fb), fb.to_list()[:20])
    log.debug('cell_method counts: %s', df['cell_method'].value_counts().rows())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _select_cells_frame(native: pl.DataFrame, *, id_col: str, crs,
                        config: RunConfig, water_mask_col: str | None = None,
                        geometry_col: str = 'geometry',
                        values: CellValues | None = None) -> pl.DataFrame:
    """URMA cell selection for reservoirs given as a polars DataFrame.

    native         : id_col and geometry_col (WKB or WKT, in `crs`).
    water_mask_col : Boolean column (no nulls): True -> water-mask cells,
                     False -> overlap/elevation rule. None (default) -> the
                     rule (max water mask of the intersecting cells == 1).
    values         : CellValues to reuse (default: fetched here).
    Returns id_col + CELL_SCHEMA, sorted by id.
    """
    gs = load_geometries(native, id_col, geometry_col, crs,
                         label='native reservoir')
    if water_mask_col is None:
        flags = [None] * len(gs.ids)
    else:
        if water_mask_col not in native.columns:
            raise KeyError(f'water mask column {water_mask_col!r} not in '
                           f'{native.columns}')
        wm = native.get_column(water_mask_col)
        if wm.dtype != pl.Boolean:
            raise TypeError(f'{water_mask_col!r} must be Boolean, got '
                            f'{wm.dtype}')
        if wm.null_count():
            raise ValueError(f'{wm.null_count()} null values in '
                             f'{water_mask_col!r}')
        flags = [int(w) for w in wm.to_list()]

    values = values or fetch_cell_values(config)
    grid = values.grid
    polys = reproject(gs.geoms, gs.crs, grid.crs)
    rows = [select_one(g, values, f) for g, f in zip(polys, flags)]
    geoms = np.array([r['geom'] for r in rows], dtype=object)
    ok = np.array([g is not None for g in geoms])
    wkb = [None] * len(geoms)
    if ok.any():
        wgs = reproject(geoms[ok], grid.crs, 'EPSG:4326')
        for k, g in zip(np.flatnonzero(ok), wgs):
            wkb[k] = shapely.to_wkb(g)
    out = pl.DataFrame({
        id_col: gs.ids,
        'cell_geometry': wkb,
        **{c: [r[c] for r in rows] for c in CELL_SCHEMA
           if c != 'cell_geometry'},
    }, schema={id_col: pl.Utf8, **CELL_SCHEMA}).sort(id_col)
    log.debug('select_urma_cells: %d reservoirs, water mask from %s',
             out.height, water_mask_col or 'the rule')
    _log_summary(out, id_col)
    return out


def select_urma_cells(geometry, *, id_col: str, config: RunConfig,
                      layer: str | None = None,
                      sample: int | None = None, sample_seed: int = 1,
                      crs=None, geometry_col: str = 'geometry',
                      water_mask_col: str | None = None):
    """URMA cell selection (URMA met variables).

    geometry : the reservoir polygons -- a geopandas GeoDataFrame (e.g.
               gpd.read_file of a GeoPackage, or a database read), a path
               to a .shp / .gpkg, or a polars DataFrame with id_col and
               WKB/WKT geometry_col (then `crs` is required).

    GeoDataFrame / path -> GeoDataFrame: every original column + N_CELLS, ELEV_M (mean elevation of the
    selected cells, m), WATER_MASK (1 = water-mask rule, 0 =
    overlap/elevation rule);
    geometry = the selected cells, in the input CRS. Nothing is written to
    disk; saving it is up to the caller (see scripts/run_*_cells.py).
    polars DataFrame -> polars DataFrame: id_col, cell_geometry (WKB,
    EPSG:4326), n_cells, n_touching,
    wm_value, cell_method, cell_elev_m.

    layer          : layer to read from a multi-layer GeoPackage (path input)
    sample         : random N reservoirs, for a trial (path / GeoDataFrame)
    water_mask_col : existing 0/1 / Boolean column for the flag; None = the
                     rule (max water mask of the intersecting cells == 1)
    """
    log.info('select_urma_cells: started')
    if isinstance(geometry, pl.DataFrame):
        if crs is None:
            raise ValueError('crs is required when geometry is a DataFrame')
        log.info('select_urma_cells: processing %d geometries',
                 geometry.height)
        out = _select_cells_frame(geometry, id_col=id_col, crs=crs,
                                  config=config,
                                  water_mask_col=water_mask_col,
                                  geometry_col=geometry_col)
    else:
        from .files import is_geodataframe, urma_cells
        if not (is_geodataframe(geometry)
                or isinstance(geometry, (str, os.PathLike))):
            raise TypeError('geometry must be a GeoDataFrame, a file path '
                            '(.shp / .gpkg) or a polars DataFrame, got '
                            f'{type(geometry).__name__}')
        out = urma_cells(geometry, id_col=id_col, config=config,
                         layer=layer,
                         water_mask_col=water_mask_col, sample=sample,
                         sample_seed=sample_seed)
    log.info('select_urma_cells: finished')
    return out
