"""The one place grid information comes from: a reference image per source.

  urma     : the first URMA daily image of 2026 (by system:index), band TMAX.
             Checked: every band shares this grid; all 17,089 daily
             images 1980-01-01 to 2026-10-14 sit on the same pixel
             lattice; the URMA water mask sits on the same lattice
             (identical CRS and pixel size, origin offset 0).
  prism800 : the first image of the PRISM 800 m collection, band ppt.
             Checked: the same grid on 549 dates, 1981-2026.

grid_from_ee() reads the CRS + affine transform once (one small request)
for local work: cell squares, cell counts, centroids. The CRS text is made
pyproj-safe with grid_cells.local_crs() (URMA: Earth Engine projects on the
6,371,200 m sphere given in its WKT, which pyproj would otherwise ignore),
and every run checks five pixel centres against Earth Engine and stops if
local and Earth Engine conversions differ. ee_projection() gives
the same projection as an Earth Engine object for reductions, so local cells
and server-side reductions use exactly the same grid.
"""

from __future__ import annotations

import logging

import ee

from .config import GRID_REFERENCE, RunConfig
from .ee_utils import get_info, init_ee
from .grid_cells import Grid

log = logging.getLogger('resdlem_force')

_CACHE: dict = {}


def reference_image(source: str) -> ee.Image:
    """The reference image of a source (server-side object)."""
    if source not in GRID_REFERENCE:
        raise ValueError(f'unknown grid source {source!r}; use '
                         f'{sorted(GRID_REFERENCE)}')
    coll, band, year = GRID_REFERENCE[source]
    c = ee.ImageCollection(coll)
    if year:
        c = c.filter(ee.Filter.stringStartsWith('system:index', str(year)))
    return c.sort('system:index').first().select(band)


def ee_projection(source: str):
    """The reference grid as an ee.Projection (for crs= in reductions)."""
    return reference_image(source).projection()


def grid_info(source: str, config: RunConfig) -> dict:
    """{'grid': Grid, 'width', 'height', 'col0', 'row0' (pixel extent of
    the image on that grid), 'image': system:index} of the reference image.
    Three small requests (projection, band info, CRS check), cached per
    source and project."""
    key = (source, config.project)
    if key not in _CACHE:
        init_ee(config.project, config.base_url)
        img = reference_image(source)
        info = get_info(ee.Dictionary({
            'proj': img.projection(),
            'index': img.get('system:index'),
        }), config)
        # band info of the image: 'dimensions' [width, height] and, when the
        # pixels do not start at the transform's origin, 'origin' [col, row]
        band = get_info(img, config)['bands'][0]
        dims = band.get('dimensions') or [None, None]
        origin = band.get('origin') or [0, 0]
        grid = Grid.from_projection(info['proj'])
        _verify(source, grid, dims, origin, config)
        _CACHE[key] = {'grid': grid, 'image': info['index'],
                       'width': dims[0], 'height': dims[1],
                       'col0': int(origin[0]), 'row0': int(origin[1])}
        log.debug('%s grid from image %s: pixel %.10g x %.10g, origin '
                 '(%.10g, %.10g), %s x %s pixels from col/row %s', source,
                 info['index'], grid.a, grid.e, grid.c, grid.f, dims[0],
                 dims[1], origin)
    return _CACHE[key]


def _check_points(grid: Grid, dims, origin):
    """Five pixel centres of the image (corners inset by a tenth, and the
    middle) as (x, y) in the grid CRS."""
    w, h = (dims or [100, 100])
    c0, r0 = int(origin[0]), int(origin[1])
    cols = [c0 + w // 10, c0 + w - 1 - w // 10, c0 + w // 2]
    rows = [r0 + h // 10, r0 + h - 1 - h // 10, r0 + h // 2]
    picks = [(cols[0], rows[0]), (cols[1], rows[0]), (cols[0], rows[1]),
             (cols[1], rows[1]), (cols[2], rows[2])]
    return [grid.centre(c, r) for c, r in picks]


def _verify(source: str, grid: Grid, dims, origin, config: RunConfig):
    """Stop if local conversions do not match Earth Engine's for this grid:
    five pixel centres are converted to lon/lat locally, then back to the
    grid CRS by Earth Engine; they must land within 1/1000 of a pixel of
    where they started. One small request."""
    import pyproj
    pts = _check_points(grid, dims, origin)
    to_ll = pyproj.Transformer.from_crs(grid.crs, 'EPSG:4326',
                                        always_xy=True)
    ll = [to_ll.transform(x, y) for x, y in pts]
    proj = ee.Projection(grid.ee_crs or grid.crs)
    back = get_info(ee.List([
        ee.Geometry.Point([lo, la]).transform(proj, 0.001).coordinates()
        for lo, la in ll]), config)
    tol = abs(grid.a) * 1e-3
    worst = max(max(abs(bx - x), abs(by - y))
                for (x, y), (bx, by) in zip(pts, back))
    if worst > tol:
        raise RuntimeError(
            f'{source} grid: local CRS conversion does not match Earth '
            f'Engine (off by {worst:.6g} CRS units = {worst / abs(grid.a):.3f}'
            f' pixels). Cells built locally would land on the wrong pixels. '
            f'Local CRS: {grid.crs}')
    log.debug('%s grid: local CRS matches Earth Engine (max %.3g CRS units '
             'on 5 pixel centres)', source, worst)


def grid_from_ee(source: str, config: RunConfig) -> Grid:
    return grid_info(source, config)['grid']
