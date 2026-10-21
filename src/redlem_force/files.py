"""GeoDataFrame / file in, GeoDataFrame out: URMA and PRISM cell selection
in the same format.

select_urma_cells and select_prism_cells take the reservoir polygons as a
GeoDataFrame (e.g. gpd.read_file of a GeoPackage, or a database read) or a
path to a .shp / .gpkg, and both return a GeoDataFrame:

  * every original column, plus
      URMA : N_CELLS, ELEV_M (mean elevation of the selected cells, m),
             WATER_MASK (1 = cells picked by the water-mask rule,
             0 = by the overlap/elevation rule)
      PRISM: N_CELLS
  * geometry = the selected cells (union of the pixel squares), in the
    input CRS; null where a reservoir got no cells (URMA only)

Nothing is written to disk: saving the result is up to the caller.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
import shapely

from .cells import _select_cells_frame as select_cells_frame
from .config import RunConfig
from .geometry import reproject

log = logging.getLogger('redlem_force')

ELEV_COL = 'ELEV_M'
WM_COL = 'WATER_MASK'
N_COL = 'N_CELLS'


def check_written_valid(geoms, ids, what: str):
    """Cell polygons redlem-force returns must be valid in the output CRS
    (they are built valid; this guards the reprojection). Raises if not."""
    bad = [i for i, g in zip(ids, geoms)
           if g is not None and not g.is_valid]
    if bad:
        raise RuntimeError(f'{len(bad)} {what} polygons invalid after '
                           f'reprojection to the output CRS, e.g. {bad[:10]}')


def _geopandas():
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError('cell selection needs geopandas: pip install '
                          'geopandas') from e
    return gpd


def _attributes_frame(gdf) -> pl.DataFrame:
    """GeoDataFrame attributes -> polars, column by column (no pyarrow)."""
    cols = {}
    for c in gdf.columns:
        if c == gdf.geometry.name:
            continue
        s = gdf[c]
        if s.dtype == object:
            cols[c] = pl.Series(c, s.tolist())
        else:
            cols[c] = pl.Series(c, s.to_numpy())
    return pl.DataFrame(cols)


def is_geodataframe(obj) -> bool:
    return hasattr(obj, 'geometry') and hasattr(obj, 'crs') and \
        not isinstance(obj, pl.DataFrame)


# ---------------------------------------------------------------------------
# Shared steps
# ---------------------------------------------------------------------------
def _input(geometry, *, layer):
    """GeoDataFrame from a GeoDataFrame or a path."""
    gpd = _geopandas()
    if is_geodataframe(geometry):
        return geometry
    path = Path(geometry)
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    log.info('read %d features from %s (crs %s)', len(gdf), path, gdf.crs)
    return gdf


def _checked(gdf, *, id_col, new_cols, sample, sample_seed):
    if gdf.crs is None:
        raise ValueError('the reservoir geometry has no CRS')
    if id_col not in gdf.columns:
        raise KeyError(f'ID column {id_col!r} not in {list(gdf.columns)}')
    old = [c for c in new_cols if c in gdf.columns]
    if old:
        log.warning('input already has %s; overwritten', old)
        gdf = gdf.drop(columns=old)
    if sample:
        gdf = gdf.sample(n=min(sample, len(gdf)), random_state=sample_seed)
        log.info('sampled %d features (seed %d)', len(gdf), sample_seed)
    return gdf


def _native(gdf, id_col) -> pl.DataFrame:
    return pl.DataFrame({
        id_col: gdf[id_col].astype(str).tolist(),
        'geometry': list(shapely.to_wkb(gdf.geometry.values)),
    })


def _assemble(gdf, id_col, add: pl.DataFrame, what: str):
    """Original rows + the columns of `add` (joined on id_col; its
    'cell_geometry' is WKB EPSG:4326) -> GeoDataFrame, geometry = the cells
    in the input CRS."""
    gpd = _geopandas()
    attrs = _attributes_frame(gdf)
    key = attrs.get_column(id_col).cast(pl.Utf8).alias('_key')
    out = (attrs.with_columns(key)
                .join(add.rename({id_col: '_key'}), on='_key', how='left')
                .drop('_key'))
    if out.height != len(gdf):
        raise RuntimeError(f'row count changed in join: {len(gdf)} -> '
                           f'{out.height} (duplicate IDs?)')
    wkb = out.get_column('cell_geometry').to_list()
    geoms = np.array([shapely.from_wkb(w) if w is not None else None
                      for w in wkb], dtype=object)
    ok = np.array([g is not None for g in geoms])
    if ok.any():
        geoms[ok] = reproject(geoms[ok], 'EPSG:4326', gdf.crs.to_wkt())
    check_written_valid(geoms, out[id_col].to_list(), what)
    return gpd.GeoDataFrame(out.drop('cell_geometry').to_dict(
        as_series=False), geometry=list(geoms), crs=gdf.crs)


# ---------------------------------------------------------------------------
# URMA / PRISM
# ---------------------------------------------------------------------------
def urma_cells(geometry, *, id_col: str, config: RunConfig, layer=None,
               water_mask_col=None, sample=None, sample_seed=1):
    gdf = _input(geometry, layer=layer)
    gdf = _checked(gdf, id_col=id_col, new_cols=(N_COL, ELEV_COL, WM_COL),
                   sample=sample, sample_seed=sample_seed)
    native = _native(gdf, id_col)
    if water_mask_col is not None:
        if water_mask_col not in gdf.columns:
            raise KeyError(f'water mask column {water_mask_col!r} not in '
                           f'{list(gdf.columns)}')
        flag = gdf[water_mask_col]
        if flag.isna().any():
            raise ValueError(f'{int(flag.isna().sum())} null values in '
                             f'{water_mask_col!r}')
        native = native.with_columns(
            pl.Series(water_mask_col, flag.astype(int).eq(1).tolist()))
    cells = select_cells_frame(native, id_col=id_col, crs=gdf.crs.to_wkt(),
                               config=config, water_mask_col=water_mask_col)
    add = cells.select(
        pl.col(id_col),
        pl.col('n_cells').alias(N_COL),
        pl.col('cell_elev_m').alias(ELEV_COL),
        (pl.col('cell_method') == 'water_mask').cast(pl.Int8).alias(WM_COL),
        pl.col('cell_geometry'))
    out = _assemble(gdf, id_col, add, 'URMA cell')
    log.info('URMA cells: %d reservoirs; %s 1 -> %d, 0 -> %d; %d with no '
             'cells', len(out), WM_COL, int((out[WM_COL] == 1).sum()),
             int((out[WM_COL] == 0).sum()), int(out.geometry.isna().sum()))
    return out


def prism_cells(geometry, *, id_col: str, config: RunConfig, layer=None,
                sample=None, sample_seed=1):
    from .prism_cells import _select_frame
    gdf = _input(geometry, layer=layer)
    gdf = _checked(gdf, id_col=id_col, new_cols=(N_COL,), sample=sample,
                   sample_seed=sample_seed)
    cells = _select_frame(_native(gdf, id_col), id_col=id_col,
                          crs=gdf.crs.to_wkt(), config=config,
                          source='prism800')
    add = cells.select(pl.col(id_col), pl.col('n_cells').alias(N_COL),
                       pl.col('cell_geometry'))
    out = _assemble(gdf, id_col, add, 'PRISM cell')
    n = out[N_COL]
    log.info('PRISM cells: %d reservoirs; %d single cells, %d with 2+ cells',
             len(out), int((n == 1).sum()), int((n > 1).sum()))
    return out
