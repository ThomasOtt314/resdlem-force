"""PRISM cell selection: the precip counterpart of select_urma_cells.

select_prism_cells() picks, for each reservoir, the PRISM 800 m cells that
extraction reduces precip over (PRISM 800 m, and PRISM 4 km on the same
cells) -- rule in grid_cells:
  * candidates: the PRISM cells sharing area with the polygon (touching
    only at an edge / corner does not count), with their overlap areas
  * cells taken by overlap, largest first, until their total area reaches
    the polygon's area (ceil(area / cell area) cells; over by less than
    one cell). Smaller than one cell -> the cell it overlaps most.
Stored as polygons only (union of the cell squares; one square for a single
cell), like the URMA cells. Extraction sends these polygons as they are.
No water mask. Built locally from the grid definition; the only Earth Engine
call reads the grid (CRS + affine transform) of the PRISM 800 m reference
image (resdlem_force.grids).

Run it once and store the result next to the URMA cells; extract_urma
needs these cells for precip.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import polars as pl
import shapely

from . import grid_cells, grids
from .config import PRECIP_SOURCES, RunConfig
from .geometry import internal_points, load_geometries, reproject

log = logging.getLogger('resdlem_force')

# Cells are built on the PRISM 800 m grid only; PRISM 4 km is reduced over
# the same cells on that grid (extract.py).
PRISM_SOURCES = {name: (coll, band) for name, coll, band in PRECIP_SOURCES
                 if name == 'prism800'}
N_COL = 'N_CELLS'


def grid_from_ee(source: str, config: RunConfig) -> grid_cells.Grid:
    """PRISM 800 m grid from its reference image (resdlem_force.grids)."""
    if source not in PRISM_SOURCES:
        raise ValueError(f'source must be one of {sorted(PRISM_SOURCES)}, '
                         f'got {source!r}')
    return grids.grid_from_ee(source, config)


def cells_for(gs, points_wgs84, grid):
    """(polygons in grid CRS, n_cells, single_cell) for a GeometrySet."""
    geoms, n, single = grid_cells.build(gs, points_wgs84, grid)
    log.debug('PRISM cells (largest overlap until the polygon area is '
             'reached): %d reservoirs, %d single cells, median %d, max %d '
             'cells', len(n), int(single.sum()), int(np.median(n)),
             int(n.max()))
    return geoms, n, single


def _select_frame(df: pl.DataFrame, *, id_col, crs, config, source,
                  geometry_col='geometry', grid=None) -> pl.DataFrame:
    gs = load_geometries(df, id_col, geometry_col, crs,
                         label='native reservoir')
    grid = grid or grid_from_ee(source, config)
    geoms, n, _ = cells_for(gs, internal_points(gs), grid)
    wgs = reproject(geoms, grid.crs, 'EPSG:4326')
    return pl.DataFrame({
        id_col: gs.ids,
        'cell_geometry': list(shapely.to_wkb(wgs)),
        'n_cells': n,
    }, schema={id_col: pl.Utf8, 'cell_geometry': pl.Binary,
               'n_cells': pl.Int64})


def select_prism_cells(geometry, *, id_col: str, config: RunConfig,
                       layer: str | None = None,
                       sample: int | None = None, sample_seed: int = 1,
                       crs=None, geometry_col: str = 'geometry'):
    """PRISM 800 m cell selection (PRISM 800 m and 4 km precip).

    geometry : the reservoir polygons -- a geopandas GeoDataFrame (e.g.
               gpd.read_file of a GeoPackage, or a database read), a path
               to a .shp / .gpkg, or a polars DataFrame with id_col and
               WKB/WKT geometry_col (then `crs` is required).

    GeoDataFrame / path -> GeoDataFrame: every original column + N_CELLS;
    geometry = the selected cells, in the input CRS. Nothing is written to
    disk; saving it is up to the caller (see scripts/run_*_cells.py).
    polars DataFrame -> polars DataFrame: id_col, cell_geometry (WKB,
    EPSG:4326), n_cells.

    layer          : layer to read from a multi-layer GeoPackage (path input)
    sample         : random N reservoirs, for a trial (path / GeoDataFrame)
    """
    log.info('select_prism_cells: started')
    if isinstance(geometry, pl.DataFrame):
        if crs is None:
            raise ValueError('crs is required when geometry is a DataFrame')
        log.info('select_prism_cells: processing %d geometries',
                 geometry.height)
        out = _select_frame(geometry, id_col=id_col, crs=crs, config=config,
                            source='prism800', geometry_col=geometry_col)
    else:
        from .files import is_geodataframe, prism_cells
        if not (is_geodataframe(geometry)
                or isinstance(geometry, (str, os.PathLike))):
            raise TypeError('geometry must be a GeoDataFrame, a file path '
                            '(.shp / .gpkg) or a polars DataFrame, got '
                            f'{type(geometry).__name__}')
        out = prism_cells(geometry, id_col=id_col, config=config,
                          layer=layer,
                          sample=sample, sample_seed=sample_seed)
    log.info('select_prism_cells: finished')
    return out
