"""Map of the cells selected for a reservoir: the original polygon, its URMA
cells and its PRISM 800 m cells on one map, over satellite imagery.

plot_cells() takes the ORIGINAL reservoir geometry, runs the same two
selections extraction uses (cells.select_one for URMA, grid_cells.select
for PRISM) and writes one figure per reservoir.

Map
  * projection: Web Mercator (EPSG:3857), the projection of the imagery
    tiles, so the imagery is not resampled. It is conformal: shapes are
    true locally, so cells keep their true shape and orientation (URMA cells
    are tilted from north by the Lambert grid's convergence; PRISM cells,
    a lon/lat grid, are not). Cell edges are densified before reprojection,
    so they are drawn at their true position.
  * basemap 'esri': Esri World Imagery tiles via contextily (internet
    needed). No credit text is drawn on the map; credit the imagery in
    the caption. The provider's attribution text (xyzservices
    providers.Esri.WorldImagery.attribution) is logged when tiles load. The zoom is chosen so tile
    pixels are no coarser than the figure's pixels at `dpi`. If the tiles
    cannot be fetched, the map is drawn on white, with a warning.
    basemap=None: no imagery (white background, darker colours).
  * every selected cell outlined on its own, so cells can be counted
    (each edge drawn once)
  * flag in the top-left corner: 'URMA water mask: yes' when every URMA
    cell has water mask = 1, 'no' when none does. A reservoir's cells are
    either all water or all non-water (the water-mask rule keeps only water
    cells; otherwise no candidate is water); a mix shows 'mixed' and is
    logged as a warning.
  * scale bar in km, corrected for Web Mercator's scale at the reservoir's
    latitude (pyproj scale factor)
  * legend below the map: Reservoir, URMA 2.5 km cells, PRISM 800 m cells

Earth Engine: the two grid definitions and, for URMA, the rtma_elev /
rtma_water_mask arrays (~7 requests, once per Python session; pass
`values` to reuse them). Nothing per reservoir.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import polars as pl
import pyproj
import shapely

from . import cells as urma_cells
from . import grid_cells, grids
from .config import RunConfig
from .geometry import internal_points, load_geometries, reproject

log = logging.getLogger('redlem_force')

MAP_CRS = 'EPSG:3857'

# Line colours. On imagery: bright colours with a thin dark halo. On white:
# a palette checked for colour-vision separation (dataviz validator, all
# pairs: worst CVD dE 9.2, normal-vision dE 24.0). The legend names each
# colour, so colour is never the only label.
THEMES = {
    'imagery': {'lake': '#00d0ff', 'urma': '#ffe600', 'prism': '#ff3df2',
                'halo': (0, 0, 0, 0.55), 'lake_fill': None},
    'plain': {'lake': '#1c4f8f', 'urma': '#eb6834', 'prism': '#1baf7a',
              'halo': None, 'lake_fill': (0.165, 0.471, 0.839, 0.35)},
}
LINE_W = {'lake': 1.3, 'urma': 1.8, 'prism': 0.8}
INK = '#0b0b0b'
PAD = 0.08                   # extra map area around the features, per side

_VALUES_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Selections for one reservoir
# ---------------------------------------------------------------------------
def _urma_values(config: RunConfig):
    key = config.project
    if key not in _VALUES_CACHE:
        _VALUES_CACHE[key] = urma_cells.fetch_cell_values(config)
    return _VALUES_CACHE[key]


def _selections(gs, config, values, flags):
    """Per reservoir: the polygon in EPSG:4326 and the selected (col, row)
    cells of each grid."""
    ugrid = values.grid
    pgrid = grids.grid_from_ee('prism800', config)
    u_poly = reproject(gs.geoms, gs.crs, ugrid.crs)
    p_poly = reproject(gs.geoms, gs.crs, pgrid.crs)
    p_pts = reproject(internal_points(gs), 'EPSG:4326', pgrid.crs)
    wgs = reproject(gs.geoms, gs.crs, 'EPSG:4326')
    out = []
    for i, rid in enumerate(gs.ids):
        u = urma_cells.select_one(u_poly[i], values, flags[i])
        p_cells, _ = grid_cells.select(p_poly[i], p_pts[i], pgrid)
        u_cells = u.get('cells', [])
        # water mask of each selected URMA cell (1 = water)
        _, wm, _ = values.lookup([c for c, _ in u_cells],
                                 [r for _, r in u_cells])
        water = [bool(w == 1) for w in wm]
        if len(set(water)) > 1:
            log.warning('%s: URMA cells mix water-mask and non-water cells '
                        '(%d water, %d not; method %s) -- not expected',
                        rid, sum(water), len(water) - sum(water),
                        u['cell_method'])
        out.append({'id': rid, 'wgs': wgs[i],
                    'urma': u_cells, 'urma_water': water,
                    'urma_method': u['cell_method'], 'prism': p_cells})
    return ugrid, pgrid, out


def _cells_in_map_crs(cells, grid):
    """Each cell square, densified (8 points per side) and reprojected to
    the map CRS. Object array of polygons."""
    if not cells:
        return np.array([], dtype=object)
    sq = grid.squares(np.array([c for c, _ in cells]),
                      np.array([r for _, r in cells]))
    sq = shapely.segmentize(sq, abs(grid.a) / 8)
    return reproject(sq, grid.crs, MAP_CRS)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _path(geom):
    """matplotlib Path for a (multi)polygon, holes included."""
    from matplotlib.path import Path as MPath
    verts, codes = [], []
    for poly in getattr(geom, 'geoms', [geom]):
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            verts.extend(xy.tolist())
            codes.extend([MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2)
                         + [MPath.CLOSEPOLY])
    return MPath(verts, codes)


def _outline(ax, geoms, colour, lw, halo, zorder, fill=None):
    from matplotlib import patheffects as pe
    from matplotlib.patches import PathPatch
    effects = ([pe.Stroke(linewidth=lw + min(1.2, lw), foreground=halo),
                pe.Normal()] if halo else None)
    for g in geoms:
        if g is None or g.is_empty:
            continue
        p = PathPatch(_path(g), facecolor=fill or 'none', edgecolor=colour,
                      lw=lw, zorder=zorder, joinstyle='miter')
        if effects:
            p.set_path_effects(effects)
        ax.add_patch(p)


def _cell_edges(cells, grid):
    """Unique pixel edges of a set of cells (an edge shared by two cells
    drawn once), densified (8 points per edge)
    and reprojected to the map CRS. Object array of LineStrings."""
    edges = set()
    for c, r in cells:
        k = [(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)]
        for a, b in zip(k, k[1:] + k[:1]):
            edges.add(tuple(sorted((a, b))))
    if not edges:
        return np.array([], dtype=object)
    lines = shapely.linestrings([
        [(grid.c + a[0] * grid.a, grid.f + a[1] * grid.e),
         (grid.c + b[0] * grid.a, grid.f + b[1] * grid.e)]
        for a, b in sorted(edges)])
    lines = shapely.segmentize(lines, abs(grid.a) / 8)
    return reproject(lines, grid.crs, MAP_CRS)


def _edges(ax, lines, colour, lw, halo, zorder):
    from matplotlib import patheffects as pe
    from matplotlib.collections import LineCollection
    if len(lines) == 0:
        return
    lc = LineCollection([np.asarray(g.coords) for g in lines],
                        colors=colour, linewidths=lw, zorder=zorder,
                        capstyle='butt')
    if halo:
        lc.set_path_effects([pe.Stroke(linewidth=lw + min(1.2, lw),
                                       foreground=halo), pe.Normal()])
    ax.add_collection(lc)


def _water_flag(ax, water):
    """Top-left flag: were the URMA cells water-mask cells? 'yes' (all
    water mask = 1), 'no' (none), 'mixed' (not expected; also logged)."""
    if not water:
        txt = 'URMA water mask: no cells'
    elif all(water):
        txt = 'URMA water mask: yes'
    elif not any(water):
        txt = 'URMA water mask: no'
    else:
        txt = 'URMA water mask: mixed'
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=7.5, color=INK, zorder=9,
            bbox={'boxstyle': 'round,pad=0.4', 'facecolor': (1, 1, 1, 0.85),
                  'edgecolor': 'none'})


def _map_box(geoms):
    """Square-ish view box around the features, padded by PAD per side."""
    x0, y0, x1, y1 = shapely.union_all(
        [g for g in geoms if g is not None and not g.is_empty]).bounds
    w, h = x1 - x0, y1 - y0
    side = max(w, h) * 0.02
    w, h = max(w, side), max(h, side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = w * (1 + 2 * PAD), h * (1 + 2 * PAD)
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _merc_scale(lon, lat):
    """Web Mercator map units per metre on the ground at (lon, lat)."""
    f = pyproj.Proj(MAP_CRS).get_factors(lon, lat)
    return float(f.parallel_scale)


def _zoom_for(box, width_px):
    """Smallest tile zoom whose pixels are no coarser than the figure's."""
    need = (box[2] - box[0]) / width_px            # map units per pixel
    z = math.ceil(math.log2(156543.03392804097 / need))
    return int(min(max(z, 1), 18))


def _basemap(ax, box, width_px):
    """Esri World Imagery under the map. Returns the credit line, or None
    when the tiles could not be fetched."""
    try:
        import contextily as cx
        import xyzservices.providers as xp
    except ImportError:
        log.warning('contextily is not installed (pip install contextily); '
                    'drawing without imagery')
        return None
    src = xp.Esri.WorldImagery
    try:
        img, ext = cx.bounds2img(box[0], box[1], box[2], box[3], ll=False,
                                 source=src, zoom=_zoom_for(box, width_px))
    except Exception as e:                       # network, server, ...
        log.warning('imagery tiles could not be fetched (%s: %s); drawing '
                    'without imagery', type(e).__name__, str(e)[:200])
        return None
    ax.imshow(img, extent=ext, interpolation='bilinear', zorder=0)
    log.info('imagery credit (not drawn on the map): %s',
             src.get('attribution', 'Esri World Imagery'))
    return src.get('attribution', 'Esri World Imagery')


def _nice_km(width_km):
    """Scale bar length: the largest 1-2-5 x 10^n under 30% of the width."""
    best = 0.01
    for e in range(-2, 5):
        for m in (1, 2, 5):
            if m * 10 ** e <= width_km * 0.3:
                best = m * 10 ** e
    return best


def _scale_bar(ax, box, units_per_m, backing=True):
    """backing: white box behind the bar (on imagery only; on a white
    map it would hide cell edges)."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    x0, y0, x1, y1 = box
    km = _nice_km((x1 - x0) / units_per_m / 1000)
    length = km * 1000 * units_per_m
    h = (y1 - y0) * 0.012
    xs, ys = x0 + (x1 - x0) * 0.04, y0 + (y1 - y0) * 0.05
    if backing:
        ax.add_patch(FancyBboxPatch(
            (xs - (x1 - x0) * 0.02, ys - (y1 - y0) * 0.02),
            length + (x1 - x0) * 0.09, (y1 - y0) * 0.075,
            boxstyle='round,pad=0,rounding_size=%g' % ((x1 - x0) * 0.005),
            facecolor=(1, 1, 1, 0.85), edgecolor='none', zorder=7))
    for i in range(4):                             # alternating quarters
        ax.add_patch(Rectangle((xs + i * length / 4, ys), length / 4, h,
                               facecolor=INK if i % 2 == 0 else 'white',
                               edgecolor=INK, lw=0.6, zorder=8))
    ty = ys + h + (y1 - y0) * 0.008
    for frac, txt in ((0, '0'), (0.5, f'{km / 2:g}'), (1, f'{km:g}')):
        ax.text(xs + frac * length, ty, txt, ha='center', va='bottom',
                fontsize=7, color=INK, zorder=8)
    ax.text(xs + length + (x1 - x0) * 0.012, ys + h / 2, 'km', ha='left',
            va='center', fontsize=7, color=INK, zorder=8)


def _figure(sel, ugrid, pgrid, title, basemap, width_in, dpi):
    import matplotlib
    import matplotlib.patheffects  # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    lake = reproject(np.array([sel['wgs']], dtype=object), 'EPSG:4326',
                     MAP_CRS)[0]
    u_sq = _cells_in_map_crs(sel['urma'], ugrid)
    p_sq = _cells_in_map_crs(sel['prism'], pgrid)
    box = _map_box([lake, *u_sq, *p_sq])
    lon, lat = sel['wgs'].centroid.x, sel['wgs'].centroid.y

    aspect = (box[3] - box[1]) / (box[2] - box[0])
    # map width ~ figure width; + title (0.45 in) + legend (0.35 in)
    height = min(max((width_in - 0.1) * aspect + 0.85, 2.5), 11.0)
    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 8,
                         'savefig.facecolor': 'white'}):
        fig, ax = plt.subplots(figsize=(width_in, height),
                               layout='constrained')
        ax.set_xlim(box[0], box[2])
        ax.set_ylim(box[1], box[3])
        ax.set_aspect('equal')
        credit = None
        if basemap == 'esri':
            credit = _basemap(ax, box, width_in * dpi)
        theme = THEMES['imagery' if credit else 'plain']

        if theme['lake_fill']:            # fill under the cells
            _outline(ax, [lake], 'none', 0, None, zorder=1,
                     fill=theme['lake_fill'])
        _outline(ax, p_sq, theme['prism'], LINE_W['prism'], theme['halo'],
                 zorder=2)
        _edges(ax, _cell_edges(sel['urma'], ugrid), theme['urma'],
               LINE_W['urma'], theme['halo'], zorder=3)
        _water_flag(ax, sel['urma_water'])
        _outline(ax, [lake], theme['lake'], LINE_W['lake'], theme['halo'],
                 zorder=4)
        _scale_bar(ax, box, _merc_scale(lon, lat), backing=bool(credit))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.8)
        ax.set_title(title, fontsize=13, color=INK, pad=8)

        handles = [
            (Patch(facecolor=theme['lake_fill'], edgecolor=theme['lake'],
                   lw=LINE_W['lake'],
                   label='Reservoir')
             if theme['lake_fill'] else
             Line2D([], [], color=theme['lake'], lw=LINE_W['lake'] + 0.4,
                    label='Reservoir')),
            Line2D([], [], color=theme['urma'], lw=LINE_W['urma'],
                   label='URMA 2.5 km cells'),
            Line2D([], [], color=theme['prism'], lw=LINE_W['prism'] + 0.4,
                   label='PRISM 800 m cells')]
        leg = fig.legend(handles=handles, loc='outside lower center',
                         ncol=len(handles), frameon=False, fontsize=7.5,
                         handlelength=1.8, columnspacing=1.6)
        if theme['halo']:                 # bright lines need a dark key
            for h in leg.legend_handles:
                h.set_path_effects([matplotlib.patheffects.Stroke(
                    linewidth=h.get_linewidth() + 1.0,
                    foreground=theme['halo']),
                    matplotlib.patheffects.Normal()])
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def plot_cells(geometry, *, id_col: str, config: RunConfig, ids=None,
               out_dir=None, formats=('png',), dpi: int = 300,
               basemap: str | None = 'esri', title_col: str | None = None,
               layer: str | None = None, crs=None,
               geometry_col: str = 'geometry',
               water_mask_col: str | None = None, values=None,
               width_in: float = 6.5, return_figures: bool = False):
    """One map per reservoir: original polygon + URMA cells + PRISM 800 m
    cells, selected with the same rules as extraction.

    geometry  : path to a .shp / .gpkg, a GeoDataFrame, or a polars
                DataFrame with id_col and WKB/WKT geometry_col (then `crs`
                is required) -- the ORIGINAL reservoir polygons
    ids       : reservoir ids to plot (default: all rows -- keep it small)
    out_dir   : folder for <id>_cells.<format>; None = do not save
    formats   : e.g. ('png',), ('pdf',), ('png', 'pdf'); in pdf / svg the
                lines are vector, the imagery raster
    basemap   : 'esri' (Esri World Imagery, needs contextily + internet) or
                None (white background)
    title_col : column used as the map title (e.g. a name); default the id
    water_mask_col : as in select_urma_cells (None = the rule)
    values    : URMA CellValues to reuse (default: fetched once per session)
    return_figures : return {id: Figure}; otherwise figures are closed and
                {id: [saved paths]} is returned
    """
    if basemap not in ('esri', None):
        raise ValueError(f"basemap must be 'esri' or None, got {basemap!r}")
    gdf_like = hasattr(geometry, 'geometry') and hasattr(geometry, 'crs') \
        and not isinstance(geometry, pl.DataFrame)
    if isinstance(geometry, (str, os.PathLike)):
        import geopandas as gpd
        geometry = (gpd.read_file(geometry, layer=layer) if layer
                    else gpd.read_file(geometry))
        gdf_like = True
    if gdf_like:
        gdf = geometry
        if gdf.crs is None:
            raise ValueError('the geometry has no CRS')
        if ids is not None:
            gdf = gdf[gdf[id_col].astype(str).isin({str(i) for i in ids})]
        cols = {id_col: gdf[id_col].astype(str).tolist(),
                'geometry': list(shapely.to_wkb(gdf.geometry.values))}
        for c in (title_col, water_mask_col):
            if c:
                cols[c] = gdf[c].tolist()
        df = pl.DataFrame(cols)
        crs = gdf.crs.to_wkt()
        geometry_col = 'geometry'
    elif isinstance(geometry, pl.DataFrame):
        if crs is None:
            raise ValueError('crs is required when geometry is a DataFrame')
        df = geometry
        if ids is not None:
            df = df.filter(pl.col(id_col).cast(pl.Utf8).is_in(
                [str(i) for i in ids]))
    else:
        raise TypeError('geometry must be a file path, a GeoDataFrame or a '
                        f'polars DataFrame, got {type(geometry).__name__}')
    if df.height == 0:
        raise ValueError(f'no reservoirs to plot (ids={ids})')
    if ids is not None:
        missing = {str(i) for i in ids} - set(
            df[id_col].cast(pl.Utf8).to_list())
        if missing:
            log.warning('ids not found: %s', sorted(missing))
    if df.height > 50:
        log.warning('plotting %d reservoirs (one figure each)', df.height)

    gs = load_geometries(df, id_col, geometry_col, crs,
                         label='native reservoir')
    order = {r: i for i, r in enumerate(df[id_col].cast(pl.Utf8).to_list())}
    flags = ([int(bool(df[water_mask_col][order[i]])) for i in gs.ids]
             if water_mask_col else [None] * len(gs.ids))
    titles = ({i: str(df[title_col][order[i]]) for i in gs.ids}
              if title_col else {})

    values = values or _urma_values(config)
    ugrid, pgrid, sels = _selections(gs, config, values, flags)

    import matplotlib.pyplot as plt
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for sel in sels:
        fig = _figure(sel, ugrid, pgrid, titles.get(sel['id'], sel['id']),
                      basemap, width_in, dpi)
        paths = []
        if out_dir:
            safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '_'
                           for ch in sel['id'])
            for fmt in formats:
                p = out_dir / f'{safe}_cells.{fmt}'
                fig.savefig(p, dpi=dpi)
                paths.append(p)
            log.info('%s: %d URMA cells (%s), %d PRISM cells -> %s',
                     sel['id'], len(sel['urma']), sel['urma_method'],
                     len(sel['prism']), ', '.join(map(str, paths)))
        if return_figures:
            result[sel['id']] = fig
        else:
            result[sel['id']] = paths
            plt.close(fig)
    return result
