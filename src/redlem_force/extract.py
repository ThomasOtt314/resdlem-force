"""Tool 1: extract forcing data for one or many dates and reservoirs.

Input: ONE table (polars or geopandas), one row per reservoir, with
  - an ID column
  - the URMA cells   (select_urma_cells)  -> URMA met (+ URMA precip
                                             fallback)
  - the PRISM cells  (select_prism_cells) -> PRISM 800 m / 4 km precip
The original reservoir polygons are NOT used here: they are only needed
once, to select the cells. Output: URMA met variables (+ precip) per
reservoir and day; elevation / latitude are not part of it.

Everything is reduced over whole pixels of each source's own grid,
UNWEIGHTED, so each value is the plain mean of a fixed set of pixels:

1. URMA met: the URMA daily image (with derived vs2m / ea / vpd,
   met.prep_image), mean + median on the URMA grid:
     over the selected URMA CELL polygon -> exactly those pixels (a pixel
     is used when its centre is inside the polygon; a single cell's square
     holds only its own pixel centre). Reservoirs with no cell geometry are
     excluded.
2. Precip, per reservoir and day, first source with a value:
     prism800 : over the reservoir's PRISM 800 m cells -> plain mean of
                those pixels
     prism4k  : PRISM 4 km over the SAME 800 m cells, reduced on the 800 m
                grid (each 800 m cell takes the 4 km pixel it lies in)
     urma     : URMA PCP over the reservoir's URMA cells
   pr_source records which. Only the cell polygons are sent to Earth
   Engine.
3. urma_status: each URMA image's 'status' property ('permanent' /
   'provisional'), 'unknown' when absent.

All images are selected by date label (system:index == YYYYMMDD).

Work is split into jobs = blocks of days (config.days_per_request
consecutive dates of the request) x reservoir chunks (features_per_request
reservoirs). All days of a block are stacked as bands of ONE image, so one
request per source covers the whole block (benchmarked: stacking + small
grid geometry was ~5x faster than per-day requests over native polygons).
Jobs run in a process pool of n_workers.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import ee
import polars as pl

from .config import (FLOAT_COLS, ID_PROP, MEAN_BANDS, MEDIAN_BANDS,
                     OUTPUT_COLS_AFTER_ID, PRECIP_SOURCES, URMA_ID,
                     URMA_OUTPUT, RunConfig)
from .dates import label, normalize_dates
from .ee_utils import (Progress, download_csv, get_info, init_ee,
                       run_jobs, solve_split)
from . import grids
from .geometry import (chunk_by_size, geojson_bytes, ids_as_str,
                       load_geometries, reproject, to_geojson)
from .met import prep_image
from .quota import QuotaExceededError, check_quota

log = logging.getLogger('redlem_force')


# ---------------------------------------------------------------------------
# Input normalization (main process)
# ---------------------------------------------------------------------------
_URMA, _PRISM = '_urma_cells', '_prism_cells'
# Output columns that only exist when precip is extracted.
PRECIP_COLS = ('pr', 'pr_source')
MET_BANDS = MEAN_BANDS + MEDIAN_BANDS
# Precip sources reduced over the PRISM 800 m cells, on the 800 m grid
# (prism4k is resampled to that grid by Earth Engine, nearest pixel). The
# remaining source (urma) uses the URMA met geometry on the URMA grid.
PRISM_CELL_SOURCES = ('prism800', 'prism4k')
PRISM_GRID_SOURCE = 'prism800'


def _is_geodataframe(obj) -> bool:
    try:
        import geopandas as gpd
    except ImportError:
        return False
    return isinstance(obj, gpd.GeoDataFrame)


def _gpd_column(gdf, col, crs_override):
    """One geopandas column -> (list of WKB/WKT/None, CRS).
    A GeoSeries column carries its own CRS; a plain column (WKB bytes or WKT
    text) needs crs_override."""
    import geopandas as gpd
    import shapely
    if col not in gdf.columns:
        raise KeyError(f'geometry column {col!r} not in {list(gdf.columns)}')
    s = gdf[col]
    if isinstance(s, gpd.GeoSeries):
        crs = crs_override or s.crs
        if crs is None:
            raise ValueError(f'column {col!r} has no CRS; pass it explicitly')
        return list(shapely.to_wkb(s.values)), crs
    if crs_override is None:
        raise ValueError(f'column {col!r} is not a GeoSeries (e.g. a second '
                         'geometry column from a database read); pass its '
                         'CRS (urma_crs / prism_crs)')
    # plain column: shapely objects -> WKB; WKB bytes, hex WKB / EWKB text
    # (PostGIS) and WKT text are parsed later
    vals = [shapely.to_wkb(v) if isinstance(v, shapely.Geometry)
            else (None if v is None or (isinstance(v, float) and v != v)
                  else v)
            for v in s.tolist()]
    return vals, crs_override


def _normalize_input(data, *, id_col, urma_col, prism_col, urma_crs,
                     prism_crs):
    """polars or geopandas table -> (polars frame [id, _urma, _prism (if
    prism_col)], URMA cells CRS, PRISM cells CRS).

    No reservoir polygon is read: only the two cell geometries."""
    geo_cols = {_URMA: urma_col, **({_PRISM: prism_col} if prism_col else {})}
    geo_vals, crs_of = {}, {}
    if _is_geodataframe(data):
        if id_col not in data.columns:
            raise KeyError(f'column {id_col!r} not in {list(data.columns)}')
        geo_vals[_URMA], crs_of[_URMA] = _gpd_column(data, urma_col, urma_crs)
        if prism_col:
            geo_vals[_PRISM], crs_of[_PRISM] = _gpd_column(data, prism_col,
                                                           prism_crs)
        # missing IDs arrive as None or NaN; keep them null (rejected later)
        # instead of turning NaN into the text 'nan'
        ids = [None if v is None or (isinstance(v, float) and v != v)
               else str(v) for v in data[id_col].tolist()]
    elif isinstance(data, pl.DataFrame):
        for c in (id_col, *geo_cols.values()):
            if c not in data.columns:
                raise KeyError(f'column {c!r} not in {data.columns}')
        if urma_crs is None:
            raise ValueError('urma_crs (CRS of the URMA cells column) is '
                             'required for polars input')
        crs_of[_URMA] = urma_crs
        if prism_col:
            crs_of[_PRISM] = prism_crs or urma_crs
        for k, c in geo_cols.items():
            geo_vals[k] = data.get_column(c).to_list()
        ids = data.get_column(id_col).cast(pl.Utf8).to_list()
    else:
        raise TypeError('data must be a polars DataFrame or a geopandas '
                        f'GeoDataFrame, got {type(data).__name__}')

    def series(name, values):
        # Binary for WKB, Utf8 for WKT; all-null columns default to Binary
        kinds = {type(v) for v in values if v is not None}
        dtype = pl.Utf8 if kinds == {str} else pl.Binary
        return pl.Series(name, values, dtype=dtype)

    frame = pl.DataFrame([pl.Series(id_col, ids, dtype=pl.Utf8)]
                         + [series(k, v) for k, v in geo_vals.items()])
    return frame, crs_of[_URMA], crs_of.get(_PRISM)


# ---------------------------------------------------------------------------
# Planning (main process)
# ---------------------------------------------------------------------------
@dataclass
class Plan:
    id_col: str
    jobs: list = field(default_factory=list)
    excluded_ids: list = field(default_factory=list)   # no cell geometry
    precip: bool = True              # False -> no pr / pr_source


def _date_blocks(date_list, n):
    """Consecutive blocks of n dates (sorted), as YYYYMMDD labels."""
    labels = [label(d) for d in date_list]
    return [labels[k:k + n] for k in range(0, len(labels), n)]


def _prepare(data, *, id_col, urma_col, prism_col, urma_crs, prism_crs,
             dates, start, end, config: RunConfig,
             precip: bool = True) -> Plan:
    """Jobs from the URMA cells (met; URMA precip fallback) and the PRISM
    800 m cells (PRISM precip). No reservoir polygon is used."""
    date_list = normalize_dates(dates, start, end)
    if precip and not prism_col:
        raise ValueError('precip=True needs prism_cells_col (PRISM 800 m '
                         'cells from select_prism_cells); or precip=False')
    t, ucrs, pcrs = _normalize_input(
        data, id_col=id_col, urma_col=urma_col,
        prism_col=prism_col if precip else None,
        urma_crs=urma_crs, prism_crs=prism_crs)
    ids_as_str(t, id_col)                         # unique, non-null IDs

    # --- reservoirs without URMA cell geometry are excluded ----------------
    has_cells = ~t[_URMA].is_null().to_numpy()
    excluded = t.filter(~pl.Series(has_cells))[id_col].to_list()
    if excluded:
        log.warning('%d reservoirs have no URMA cell geometry and are '
                    'excluded: %s', len(excluded), excluded[:20])
    t = t.filter(pl.Series(has_cells))
    if t.height == 0:
        raise ValueError('no reservoirs to extract (none have URMA cell '
                         'geometry)')
    if precip:
        null = t.filter(pl.col(_PRISM).is_null())[id_col].to_list()
        if null:
            raise ValueError(f'{len(null)} reservoirs have no {prism_col!r} '
                             f'geometry, e.g. {null[:10]}')

    init_ee(config.project, config.base_url)

    # --- URMA cells: met geometry (and URMA precip fallback) -------------------
    cells = load_geometries(t, id_col, _URMA, ucrs, label='URMA cell',
                            expect_valid=True)
    met_geo = dict(zip(cells.ids, to_geojson(
        reproject(cells.geoms, cells.crs, 'EPSG:4326'))))
    log.info('URMA cells: %d reservoirs', len(met_geo))

    # --- PRISM 800 m cells: precip geometry (prism800 AND prism4k) ---------
    grid_geo = {}
    if precip:
        pc = load_geometries(t, id_col, _PRISM, pcrs, label='PRISM cell',
                             expect_valid=True)
        grid_geo[PRISM_GRID_SOURCE] = dict(zip(pc.ids, to_geojson(
            reproject(pc.geoms, pc.crs, 'EPSG:4326'))))
        log.info('PRISM cells: %d reservoirs', len(pc.ids))

    run_ids = sorted(met_geo)                     # sorted: stable chunks
    # every geometry a request may carry must fit the byte budget
    sizes = {i: tuple([geojson_bytes(met_geo[i])]
                      + [geojson_bytes(grid_geo[s][i]) for s in grid_geo])
             for i in run_ids}
    big = [i for i, sz in sizes.items() if max(sz) > config.max_request_bytes]
    if big:
        log.warning('%d reservoirs alone exceed max_request_bytes (%d); each '
                    'is sent on its own: %s', len(big),
                    config.max_request_bytes, big[:10])
    chunks = chunk_by_size(run_ids, sizes, config.features_per_request,
                           config.max_request_bytes)
    largest = max(max(sz) for sz in sizes.values())
    log.info('chunking: %d chunks (largest single geometry %.2f MB)',
             len(chunks), largest / 1e6)

    blocks = _date_blocks(date_list, config.days_per_request)
    jobs = []
    for labels in blocks:
        for k, chunk in enumerate(chunks):
            jobs.append({
                'labels': labels,
                'k': k,
                'cells': [(i, met_geo[i]) for i in chunk],   # URMA cells
                'grid': {s: [(i, grid_geo[s][i]) for i in chunk]
                         for s in grid_geo},
                'precip': precip,
                'cfg': config,
            })
    log.info('extract: %d dates in %d blocks of <= %d days x %d chunks = '
             '%d jobs, %d reservoirs%s', len(date_list), len(blocks),
             config.days_per_request, len(chunks), len(jobs), len(run_ids),
             '' if precip else ' (precip OFF: URMA met only)')
    return Plan(id_col, jobs, sorted(excluded), precip)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _ee_geometry(g):
    """GeoJSON dict -> ee.Geometry. Polygons get planar edges
    (geodesic=False): cell polygons follow pixel edges, which are straight
    lines in the grid's projection. Points have no edges, and Earth Engine's
    Point constructor rejects a geodesic argument ("Unrecognized arguments
    {'geodesic'}"), so it is left out for Point / MultiPoint."""
    if g.get('type') in ('Point', 'MultiPoint'):
        return ee.Geometry(g)
    return ee.Geometry(g, None, False)


def _fc(features):
    """Inline FeatureCollection from (id, GeoJSON) pairs."""
    return ee.FeatureCollection([
        ee.Feature(_ee_geometry(g), {ID_PROP: i}) for i, g in features])


def _by_labels(coll_id, labels):
    return ee.ImageCollection(coll_id).filter(
        ee.Filter.inList('system:index', labels))


def _one(coll_id, lab):
    return ee.ImageCollection(coll_id).filter(
        ee.Filter.eq('system:index', lab)).first()


def _met_image(labels):
    """URMA met bands for each day, prefixed d<label>_, as one image."""
    return ee.Image.cat([
        prep_image(_one(URMA_ID, lab)).select(
            MET_BANDS, [f'd{lab}_{b}' for b in MET_BANDS])
        for lab in labels])


def _met_long(df: pl.DataFrame, labels) -> pl.DataFrame:
    """Wide met download (d<label>_<band>_<stat>) -> one row per
    (id, date) with the output column names."""
    parts = []
    for lab in labels:
        cols = []
        for key, out in URMA_OUTPUT.items():
            src = f'd{lab}_{key}'
            if src not in df.columns:
                raise KeyError(f'URMA download missing column {src}')
            cols.append(pl.col(src).cast(pl.Float64, strict=False).alias(out))
        parts.append(df.select([pl.col(ID_PROP), pl.lit(lab).alias('date')]
                               + cols))
    return pl.concat(parts)


def _precip_long(df: pl.DataFrame, labels) -> pl.DataFrame:
    """Wide precip download (p<label>) -> (id, date, pr), nulls
    dropped."""
    bands = [f'p{lab}' for lab in labels]
    for b in bands:
        if b not in df.columns:
            df = df.with_columns(pl.lit(None, pl.Float64).alias(b))
    return (df.select([ID_PROP] + bands)
              .unpivot(index=ID_PROP, on=bands, variable_name='band',
                       value_name='pr')
              .with_columns(pl.col('band').str.slice(1).alias('date'),
                            pl.col('pr').cast(pl.Float64, strict=False))
              .filter(pl.col('pr').is_not_null())
              .select([ID_PROP, 'date', 'pr']))


def _prism800_proj():
    """PRISM 800 m reference grid (redlem_force.grids) as an ee.Projection,
    used for prism800 and prism4k."""
    return grids.ee_projection('prism800')


def _reduce_precip(coll_id, band, labels, features, cfg, grid_proj=None):
    """Plain (unweighted) mean of each feature's pixels for every day in
    labels, in one request, on grid_proj (a reference grid from
    redlem_force.grids; default: the source's own grid)."""
    bands = [f'p{lab}' for lab in labels]
    img = ee.Image.cat([_one(coll_id, lab).select([band], [b])
                        for lab, b in zip(labels, bands)])
    reducer = ee.Reducer.mean().unweighted()
    if len(bands) == 1:                 # one band: output is named 'mean'
        reducer = reducer.setOutputs(bands)
    proj = grid_proj or _one(coll_id, labels[0]).select(band).projection()
    rr = img.reduceRegions(collection=_fc(features), reducer=reducer,
                           crs=proj, tileScale=cfg.tile_scale)
    return _precip_long(download_csv(rr, [ID_PROP] + bands, cfg), labels)


def _reduce_met(days, features, cfg):
    """URMA met (mean + median, unweighted, URMA grid) for every day in
    days, in one request -> one row per (id, date)."""
    proj = grids.ee_projection('urma')
    rr = _met_image(days).reduceRegions(
        collection=_fc(features),
        reducer=ee.Reducer.mean().unweighted().combine(
            ee.Reducer.median().unweighted(), sharedInputs=True),
        crs=proj, tileScale=cfg.tile_scale)
    sel = [f'd{lab}_{k}' for lab in days for k in URMA_OUTPUT]
    return _met_long(download_csv(rr, [ID_PROP] + sel, cfg), days)


def _extract_chunk(cfg, info, status, met_geo, grid_geo, ids, precip=True):
    """Met (+ precip) for one list of reservoir IDs over the days in
    info['urma'] (days that have a URMA image). Raises on Earth Engine
    errors. Returns (frame, number of (reservoir, day) without precip)."""
    days = info['urma']

    # --- URMA met, all days in one request ---------------------------------
    met = _reduce_met(days, [(i, met_geo[i]) for i in ids], cfg)
    met = met.with_columns(pl.col('date').replace_strict(
        status, default='unknown').alias('urma_status'))
    if not precip:
        return met, 0

    # --- precip, per (reservoir, day): first source with a value -----------
    remaining = {(i, d) for i in ids for d in days}
    parts = []
    for name, coll_id, band in PRECIP_SOURCES:
        if not remaining:
            break
        src_days = [d for d in days if d in set(info[name])]
        todo_days = [d for d in src_days
                     if any((i, d) in remaining for i in ids)]
        if not todo_days:
            continue
        todo_ids = [i for i in ids
                    if any((i, d) in remaining for d in todo_days)]
        if name in PRISM_CELL_SOURCES:     # 800 m cells, 800 m grid
            geo, grid_proj = grid_geo[PRISM_GRID_SOURCE], _prism800_proj()
        else:                              # URMA geometry, URMA grid
            geo, grid_proj = met_geo, grids.ee_projection('urma')
        p = _reduce_precip(coll_id, band, todo_days,
                           [(i, geo[i]) for i in todo_ids], cfg, grid_proj)
        keep = [(i, d) in remaining for i, d in
                zip(p[ID_PROP].to_list(), p['date'].to_list())]
        p = p.filter(pl.Series(keep, dtype=pl.Boolean)).with_columns(
            pl.lit(name).alias('pr_source'))
        parts.append(p)
        remaining -= set(zip(p[ID_PROP].to_list(), p['date'].to_list()))

    pr = (pl.concat(parts) if parts else
          pl.DataFrame(schema={ID_PROP: pl.Utf8, 'date': pl.Utf8,
                               'pr': pl.Float64, 'pr_source': pl.Utf8}))
    out = met.join(pr, on=[ID_PROP, 'date'], how='left')
    return out, len(remaining)


def _block_info(labels, precip, cfg):
    """Which days of the block each source has, and URMA status per day.
    One request."""
    urma = _by_labels(URMA_ID, labels).map(
        lambda im: im.set('_st', ee.Algorithms.If(
            im.get('status'), im.get('status'), 'unknown')))
    d = {'urma': urma.aggregate_array('system:index'),
         'urma_status': urma.aggregate_array('_st')}
    if precip:
        for name, coll_id, _ in PRECIP_SOURCES:
            d[name] = _by_labels(coll_id, labels).aggregate_array(
                'system:index')
    info = get_info(ee.Dictionary(d), cfg)
    status = dict(zip(info['urma'], info['urma_status']))
    info['urma'] = [lab for lab in labels if lab in set(info['urma'])]
    return info, status


def _job_name(labels):
    return labels[0] if len(labels) == 1 else f'{labels[0]}-{labels[-1]}'


def _extract_job(job):
    """One (block of days, chunk) -> (status, name, k, payload,
    n_no_precip, bad).

    status 'ok'  : payload = DataFrame of the (reservoir, day) rows that
                   succeeded.
    status 'fail': payload = message; nothing succeeded.
    bad: [(what, message)] -- reservoirs that failed even alone, and
    ('*', ...) for days with no URMA image (no rows for those days).
    A request that is too big (payload size / timeout / memory) is split in
    half and retried, down to single reservoirs. Never raises.
    """
    cfg, labels, k = job['cfg'], job['labels'], job['k']
    name = _job_name(labels)
    precip = job.get('precip', True)
    try:
        info, status = _block_info(labels, precip, cfg)
        missing = [(f'* {lab}', 'no URMA image for this date')
                   for lab in labels if lab not in set(info['urma'])]
        if not info['urma']:
            return ('fail', name, k, 'no URMA image for any date in the '
                    'block', None, missing)
        met_geo = dict(job['cells'])
        grid_geo = {s: dict(v) for s, v in job.get('grid', {}).items()}

        def fetch(ids):
            return [_extract_chunk(cfg, info, status, met_geo, grid_geo,
                                   ids, precip)]

        results, bad = solve_split(fetch, [i for i, _ in job['cells']])
        if not results:
            msg = bad[0][1] if bad else 'no result'
            return ('fail', name, k, msg, None, bad + missing)
        out = pl.concat([r[0] for r in results])
        return ('ok', name, k, out, sum(r[1] for r in results),
                bad + missing)
    except Exception as e:
        return ('fail', name, k, f'{type(e).__name__}: {e}', None, [])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _finalize(df: pl.DataFrame, plan: Plan) -> pl.DataFrame:
    """Typed output columns, in order. processed_date: the local date when
    these rows were assembled ('YYYY-MM-DD' text)."""
    df = df.rename({ID_PROP: plan.id_col}).with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in FLOAT_COLS
         if c in df.columns]
        + [pl.lit(dt.date.today().isoformat(), pl.Utf8)
           .alias('processed_date')])
    cols = [c for c in OUTPUT_COLS_AFTER_ID
            if plan.precip or c not in PRECIP_COLS]
    return df.select([plan.id_col] + cols)


def _reservoir_days(jobs):
    return sum(len(j['cells']) * len(j['labels']) for j in jobs)


def _quota_gate(jobs, config):
    """Print the remaining monthly quota and this run's estimated cost, and
    stop (QuotaExceededError) if the estimate is larger than what is left.
    If the quota cannot be read, or anything else goes wrong while checking,
    print that it is unavailable and carry on."""
    if not config.check_quota or not jobs:
        return
    reservoir_days = _reservoir_days(jobs)
    try:
        info = check_quota(reservoir_days, config)
    except QuotaExceededError as e:
        print(f'QUOTA: {e}')
        raise
    except Exception as e:
        print(f'QUOTA: quota and estimate unavailable ({type(e).__name__}: '
              f'{e}); continuing without the check.')
        return
    print(f"QUOTA: estimated {info['estimated_eecu_h']:,.2f} EECU-h for "
          f"{reservoir_days:,} reservoir-days "
          f"({config.eecu_s_per_reservoir_day} EECU-s each); "
          f"{info['remaining_eecu_h']:,.1f} of {info['limit_eecu_h']:,.0f} "
          f"EECU-h left this month -> OK")


def _log_job(res, prog):
    status, name, k, payload, n_no_pr, bad = res
    prog.tick()
    if status == 'fail':
        log.error('job %s chunk %d failed: %s', name, k, payload)
    days = [b for b in bad if str(b[0]).startswith('* ')]
    res_bad = [b for b in bad if not str(b[0]).startswith('* ')]
    if days and k == 0:                  # log missing days once per block
        log.warning('%s: no URMA image for %s', name,
                    [d[0][2:] for d in days])
    if res_bad and status == 'ok':
        log.error('%s chunk %d: %d reservoirs failed even alone, e.g. %s: %s',
                  name, k, len(res_bad), res_bad[0][0], res_bad[0][1])
    if n_no_pr:
        log.warning('%s chunk %d: %d reservoir-days got no precip from any '
                    'source', name, k, n_no_pr)


_DATA_DOC = """
    data               : ONE table, polars DataFrame or geopandas
                         GeoDataFrame, one row per reservoir, with
                           id_col           unique reservoir ID
                           urma_cells_col   URMA cells (select_urma_cells;
                                            null = no cells -> row
                                            excluded, logged)
                           prism_cells_col  PRISM 800 m cells
                                            (select_prism_cells); needed
                                            when precip=True
                         The original reservoir polygons are not used.
    geopandas          : geometry columns that are GeoSeries carry their own
                         CRS (the two may differ); plain WKB/WKT columns need
                         urma_crs / prism_crs.
    polars             : geometry columns hold WKB bytes or WKT text;
                         urma_crs is required, prism_crs defaults to it.
    dates              : one date or a list ('YYYY-MM-DD', 'YYYYMMDD',
                         datetime.date); or start / end, both inclusive.
                         Grouped into blocks of config.days_per_request
                         consecutive dates of this list.
    precip             : True (default) -> pr / pr_source, per
                         reservoir and day: PRISM 800 m -> PRISM 4 km, both
                         over the PRISM 800 m cells on the 800 m grid ->
                         URMA PCP over the URMA cells. False -> URMA met
                         only: no precip requests, pr / pr_source left
                         out, prism_cells_col not needed.
"""


def _to_pandas(df: pl.DataFrame):
    """polars -> pandas without needing pyarrow: numeric columns as float64
    (null -> NaN), text columns as pandas text (object in pandas 2)."""
    import pandas as pd
    return pd.DataFrame({c: df[c].to_numpy() for c in df.columns})


def extract_urma(data, *, id_col: str, config: RunConfig, dates=None,
                 start=None, end=None, urma_cells_col: str = 'urma_cells',
                 prism_cells_col: str | None = 'prism_cells',
                 urma_crs=None, prism_crs=None, precip: bool = True,
                 allow_partial: bool = False):
    """Extract the URMA forcing variables (+ precip) and return them as a
    pandas DataFrame, one row per reservoir and day, sorted by id and date.
    What to do with it (write parquet, load to a database, ...) is up to the
    caller.
    {data}
    allow_partial      : False (default) -> any failure raises after the
                         run; True -> failures logged, successful rows
                         returned.

    Before any data request, if config.check_quota, the remaining monthly
    quota and the run's estimated cost (reservoir-days x
    config.eecu_s_per_reservoir_day) are printed, and the run stops with
    QuotaExceededError if the estimate exceeds what is left. If the quota
    cannot be read, that is printed and the run continues.

    Holds every row in memory (all reservoirs: ~1.4 million rows per month
    of dates). For long periods call it per year or month.
    """
    plan = _prepare(data, id_col=id_col, urma_col=urma_cells_col,
                    prism_col=prism_cells_col, urma_crs=urma_crs,
                    prism_crs=prism_crs, dates=dates, start=start, end=end,
                    config=config, precip=precip)
    _quota_gate(plan.jobs, config)
    frames, failed_jobs, failed = [], [], []
    prog = Progress(len(plan.jobs), 'extract')
    for res in run_jobs(_extract_job, plan.jobs, config):
        _log_job(res, prog)
        status, name, k, payload, _, bad = res
        if status == 'ok':
            frames.append(payload)
        else:
            failed_jobs.append(f'{name} chunk {k}: {payload}')
        failed += [f'{name} chunk {k} {i}: {m}' for i, m in bad]
    if (failed_jobs or failed) and not allow_partial:
        first = (failed_jobs or failed)[0]
        raise RuntimeError(
            f'{len(failed_jobs)} of {len(plan.jobs)} jobs failed, '
            f'{len(failed)} reservoir / day failures; first: {first}')
    if not frames:
        raise RuntimeError('no jobs succeeded')
    out = _finalize(pl.concat(frames), plan)
    return _to_pandas(out.sort([plan.id_col, 'date']))


extract_urma.__doc__ = extract_urma.__doc__.format(data=_DATA_DOC)
