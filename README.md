# redlem-force

Extract DLEM forcing data from Google Earth Engine for reservoir geometries.
Cell selection returns GeoDataFrames; extraction returns a pandas DataFrame.
Nothing is written to disk by the package — saving is up to you (the
scripts in `scripts/` show one way).

1. **`extract_urma`** — forcing data (pandas DataFrame) for one day or a
   period, one reservoir or many, from one table holding each reservoir's
   URMA cells and PRISM cells. The original
   reservoir polygons are not used: they are only needed to select the
   cells.
2. **URMA cell selection** — native reservoir polygon → the URMA grid cells
   used for the met variables (with or without the water mask):
   - `select_urma_cells(gdf, ...)` — GeoDataFrame (or a .shp / .gpkg path)
     in, GeoDataFrame out: all original columns plus `N_CELLS`, `ELEV_M`
     and `WATER_MASK`; geometry = the selected cells. Also accepts a polars
     DataFrame. Computed locally on the URMA grid (below).
3. **PRISM cell selection** — `select_prism_cells(path, ...)`: native
   polygon → the PRISM 800 m cells used for precip (no water mask). Candidates
   are the cells sharing area with the polygon (edge/corner contact does
   not count); cells are taken by overlap, largest first, until their total
   area reaches the polygon's area — ceil(area / cell area) cells, over by
   less than one cell. A reservoir smaller than one cell gets the cell it
   overlaps most. Stored as polygons only (one square for a single cell),
   like the URMA cells, and sent to Earth Engine as those polygons. Same
   format as `select_urma_cells`: GeoDataFrame in, GeoDataFrame out with
   the original columns plus `N_CELLS`. Needed for precip.
4. **Cell maps** — `plot_cells(path, id_col=..., ids=[...], ...)`: one
   map per reservoir from the ORIGINAL polygon — the polygon, its URMA
   cells and its PRISM 800 m cells (same rules as above), over Esri World
   Imagery, with a legend and a km scale
   bar. Needs `pip install matplotlib contextily` (or the `plot` extra).
   See *Cell maps* below.

All Earth Engine work runs as local requests on the high-volume endpoint in a
worker pool; nothing uses `ee.batch.Export`.

## Use without installing (while iterating)

Put `src/` on the import path at the top of a script, then import normally.
Edits under `src/redlem_force/` take effect on the next run:

```python
import sys
sys.path.insert(0, '/path/to/redlem-force/src')
import redlem_force
```

The scripts in `scripts/` already do this relative to their own location.
`print(redlem_force.__file__)` shows which copy is imported. If an installed
copy exists it would shadow the source folder: `pip uninstall redlem-force`.

## Install

```bash
pip install /path/to/redlem-force            # regular install
pip install -e /path/to/redlem-force         # editable, for development
pip install -e '/path/to/redlem-force[test]' # plus pytest
```

or build a wheel to share: `pip wheel --no-deps /path/to/redlem-force`, then
`pip install redlem_force-0.13.0-py3-none-any.whl`.

Requires an authenticated Earth Engine account (`earthengine authenticate`).

## Cell selection

`select_urma_cells` and `select_prism_cells` take the same arguments
(`select_urma_cells` adds `water_mask_col`) and return the same kind of
result:

```python
gdf = gpd.read_file('owen_master_v1.gpkg')        # or a database read
cfg = redlem_force.RunConfig(project='nasa-revap')
urma = redlem_force.select_urma_cells(gdf, id_col='OWEN_ID', config=cfg)
prism = redlem_force.select_prism_cells(gdf, id_col='OWEN_ID', config=cfg)
```

Each is a GeoDataFrame: every original column plus the columns below, with
geometry = the selected cells (union of the pixel squares) in the input CRS.
A .shp / .gpkg path works in place of the GeoDataFrame (`layer=` for a
multi-layer GeoPackage). Nothing is written; `scripts/run_urma_cells.py` and
`scripts/run_prism_cells.py` save the results (GeoPackage, plus a parquet of
the URMA attributes).

| column | meaning |
|---|---|
| `N_CELLS` | number of selected cells (URMA and PRISM) |
| `ELEV_M` | URMA only: mean elevation (m) of the selected URMA cells |
| `WATER_MASK` | URMA only: 1 = cells selected with the water-mask rule, 0 = overlap/elevation rule |

The water-mask flag is max(water mask of the intersecting cells) == 1
unless `water_mask_col` names an existing 0/1 column. A reservoir flagged as
water with no water cells falls back to the overlap/elevation rule and gets
`WATER_MASK = 0` (logged). Reservoirs with no cells keep their row with null
geometry and `ELEV_M`. `sample=200` runs a random trial.
`scripts/run_urma_cells.py` wraps this with a CONFIG block.

## Cell maps

```python
redlem_force.plot_cells('owen_master_v1.gpkg', id_col='OWEN_ID',
                      ids=['Hydrolakes_31392'], title_col=None,
                      out_dir='figs', dpi=300,
                      config=redlem_force.RunConfig(project='nasa-revap'))
```

- Web Mercator map (the imagery's own projection, so it is not
  resampled; conformal, so shapes are true locally). Cell edges are
  densified before reprojection. URMA cells appear tilted by the Lambert
  grid's convergence (about 10° near Lahontan); PRISM cells, a lon/lat
  grid, are not tilted.
- The scale bar uses Web Mercator's scale factor at the reservoir.
- `basemap='esri'` needs internet; if the tiles cannot be fetched, the map
  is drawn on white (warning). `basemap=None` draws on white with a
  colour-blind-checked palette.
- Earth Engine: the two grids and the URMA elevation / water-mask arrays,
  once per Python session (reused across calls).
- `scripts/plot_reservoir_cells.py` wraps this with a CONFIG block.

## Extraction input: one table of cells

`extract_urma` takes ONE table — a polars DataFrame or a
geopandas GeoDataFrame — with one row per reservoir. No reservoir polygon:

| column (default name) | used for | notes |
|---|---|---|
| ID (`id_col`) | joining | cast to text; unique, non-null |
| `urma_cells` | URMA cells → met variables (+ URMA precip fallback) | from `select_urma_cells`; null = no cells → reservoir excluded (listed in the log / summary) |
| `prism_cells` | PRISM 800 m cells → PRISM 800 m / 4 km precip | from `select_prism_cells`; required when `precip=True` |

Column names are set with `urma_cells_col` and `prism_cells_col`.
**geopandas:** geometry columns that are
GeoSeries carry their own CRS, and the two may differ. A column that is
not a GeoSeries — e.g. the second geometry column of a `gpd.read_postgis`
read — may hold WKB bytes, hex WKB / EWKB text (as PostGIS returns it),
WKT text or shapely objects; pass its CRS (`urma_crs` / `prism_crs`).
**polars:** geometry as WKB bytes, hex WKB / EWKB or WKT text; `urma_crs`
is required, `prism_crs` defaults to `urma_crs`.

## Examples

Run scripts under `if __name__ == '__main__':` — the worker pool needs it on
macOS and Windows.

```python
import logging
import geopandas as gpd
import redlem_force

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    cfg = redlem_force.RunConfig(project='nasa-revap', n_workers=20)

    gdf = gpd.read_file('owen_master_v1.gpkg')

    # 1. URMA and PRISM cells (GeoDataFrames)
    urma = redlem_force.select_urma_cells(gdf, id_col='OWEN_ID', config=cfg)
    prism = redlem_force.select_prism_cells(gdf, id_col='OWEN_ID', config=cfg)

    # 2. One table with both cell geometries
    table = urma[['OWEN_ID', 'geometry']].rename_geometry('urma_cells')
    p = prism.set_index('OWEN_ID').geometry
    table['prism_cells'] = gpd.GeoSeries(
        p.reindex(table['OWEN_ID']).values, index=table.index, crs=p.crs)

    # 4. Forcing data -> pandas DataFrame (one row per reservoir and day)
    df = redlem_force.extract_urma(table, id_col='OWEN_ID',
                                   start='2020-07-01', end='2020-07-31',
                                   config=cfg)

    # long periods: one call per year (or month), save each as you like
    for year in range(1980, 2027):
        df = redlem_force.extract_urma(table, id_col='OWEN_ID',
                                       start=f'{year}-01-01',
                                       end=f'{year}-12-31', config=cfg)
        df.to_parquet(f'forcing_{year}.parquet')
```

`extract_urma` holds its result in memory: all ~44,750 reservoirs are about
1.4 million rows per month of dates, so call it per year or month for long
periods.

Pass `precip=False` for URMA met only: no precip requests are made, the
`pr` / `pr_source` columns are left out, and no PRISM cells are needed
(`prism_cells_col=None`).

`RunConfig(days_per_request=5)` (default) stacks 5 consecutive dates into
one request per source.

`dates` accepts one date or a list (`'YYYY-MM-DD'`, `'YYYYMMDD'`,
`datetime.date`). `start` / `end` are both inclusive.

## Output columns

| column | source |
|---|---|
| ID | your ID column (text) |
| `date` | `'YYYYMMDD'` text |
| `pr` | Daily Precipitation, mm — PRISM 800 m `ppt`, mean over the reservoir's PRISM 800 m cells; fallback PRISM 4 km `ppt`, then URMA `PCP` (see below) |
| `tmax` | Daily Maximum Temperature, degC — URMA `TMAX`, mean |
| `tmin` | Daily Minimum Temperature, degC — URMA `TMIN`, mean |
| `vpd` | Daily Vapor Pressure Deficit, kPa — derived `vpd`, mean |
| `vs2m` | Daily 2 m wind speed, m/s — derived from URMA `WIND`, mean |
| `srad` | Daily Solar Radiation, W/m² — URMA `SRAD_TCDC`, mean |
| `etr` | Daily Alfalfa Reference ET, mm/day — URMA `ETr`, mean |
| `eto` | Daily Grass Reference ET, mm/day — URMA `ETo`, mean |
| `th` | Daily Wind Direction, deg — URMA `WDIR`, **median** |
| `sph` | Daily Specific Humidity, kg/kg — URMA `SPFH`, mean |
| `pres` | Daily Atmospheric Pressure, Pa — URMA `PRES`, mean |

| `pr_source` | `prism800`, `prism4k` or `urma` |
| `urma_status` | URMA image `status`: `permanent`, `provisional`, or `unknown` |
| `processed_date` | date the rows were produced, `'YYYY-MM-DD'` text, local date of the machine running the extraction |

A pandas DataFrame: numbers as float64 (missing = NaN), text as text
(`object` in pandas 2, `str` in pandas 3).
URMA variables are reduced over exactly the selected URMA cells (unweighted,
native grid). Names and units follow the DLEM variable config.

Met variables and the `vs2m` / `vpd` derivation are unchanged from
`urma_download_v2.py`. The reduction is not: URMA is reduced on its own
pixel grid, unweighted (no `scale=1000` sampling, which let edge samples
fall in neighbouring pixels):

over the selected URMA cell polygon, i.e. exactly the pixels whose centre
is inside it. A single cell is sent as its square too: the square holds
only its own pixel centre. (Earlier versions sent single cells as a point
at the pixel centre; it gave the same pixel and no measured speed gain.)

Note vs the old archive: `urma_elev_mean` and `lat_mean` are no longer in
the forcing output (elevation and latitude come from elsewhere; the URMA
cells file still has `ELEV_M`).

## Quota check

Before extracting, `extract_urma` prints the project's
remaining monthly Earth Engine compute quota and the run's estimated cost:

    QUOTA: estimated 1.86 EECU-h for 44,749 reservoir-days (0.15 EECU-s each);
           985.2 of 1,000 EECU-h left this month -> OK

The estimate is reservoir-days × `RunConfig.eecu_s_per_reservoir_day`
(default 0.15, an older and much higher rate; measured 2026-10-15 with
URMA met + PRISM precip and 5-day blocks: ~0.0102, so set e.g.
`eecu_s_per_reservoir_day=0.012`). If the
estimate exceeds the remaining quota (× `quota_fraction`), the run stops
with `QuotaExceededError` before any data request. If the quota cannot be
read, it prints `QUOTA: quota and estimate unavailable ...` and continues.
No per-request
tracking: usage from earlier runs is already in the remaining quota Google
reports (with some reporting delay).

Reading the quota needs `pip install google-cloud-monitoring` (or the
`[quota]` extra) and `gcloud auth application-default login`.
`redlem_force.quota_status('nasa-revap')` returns the numbers directly.
`RunConfig(check_quota=False)` turns the check off.

## How it works

**Precip, per reservoir and day** (first source with a value):

1. PRISM 800 m `ppt` — plain (unweighted) mean of the reservoir's PRISM
   800 m cells, on the 800 m grid;
2. PRISM 4 km `ppt` — over the **same 800 m cells**, on the 800 m grid
   (each 800 m cell takes the 4 km pixel it lies in);
3. URMA `PCP` — over the reservoir's URMA cell polygon, on
   the URMA grid.

Only cell polygons are sent to Earth Engine, never the native polygons.
This replaced the area-weighted mean over the native polygon (redlem-force
<= 0.5): on 5,000 reservoirs x 3 days, 27% of values differed by more than
0.01 mm (mean |difference| 0.04 mm, max 3.5 mm), and the precip requests ran
~5x faster (benchmark on 5,000 reservoirs, 2026-10-13). Every collection is matched by
date label (`system:index`), never `filterDate`: the 4 km images are stamped
12:00 UTC the day before their label.

**Latitude:** area centroid if it is inside the polygon, otherwise the pole
of inaccessibility of the largest part (computed in EPSG:5070 for geographic
input). Checked inside for every polygon; the run stops if one is not.

**Invalid geometry** is repaired with `shapely.make_valid` (polygon parts
kept) and logged.

**Grids — one source.** All grid information comes from one reference
image per source (`config.GRID_REFERENCE`, `redlem_force.grids`):

- URMA: the first URMA daily image of 2026 (`20260101`), band TMAX —
  "NWS CONUS" Lambert conformal conic, 2,539.703 m pixels. The first image
  of every year 2015–2026 has the same grid, and `rtma_elev` /
  `rtma_water_mask` sit on the same lattice. Its WKT names a WGS 84 datum
  but projects on a 6,371,200 m sphere (`semi_major` / `semi_minor`
  parameters). Earth Engine uses the sphere; pyproj on its own ignores
  those parameters and uses WGS 84 — up to 8 km (2–3 pixels) off.
  `grid_cells.local_crs()` rebuilds the projection on the sphere, which
  matched Earth Engine to 0.000 m on 40 test points (2026-10-15).
- Every run checks this: five pixel centres are converted to lon/lat
  locally and back by Earth Engine, and the run stops if they move by more
  than 1/1000 of a pixel.
- PRISM 800 m: the first image of its collection, band ppt.

Local work (cell squares, cell counts, centroids) uses the grid's CRS and
affine transform; Earth Engine reductions use the same image's projection.

**URMA cell selection** (local, same logic as `urma_cells.py`):
candidate cells are the URMA cells the polygon intersects (sharing area);
each cell's elevation and water mask are the `rtma_elev` /
`rtma_water_mask` pixel values of that cell, read once per run as arrays on
the URMA grid (6 requests). Water-mask reservoirs keep the candidates whose
water mask is 1; others keep candidates with overlap ≥ median, then
elevation ≤ median. If a water-mask reservoir has no water cells it falls
back to the overlap/elevation rule (`cell_method = water_mask_fallback`).
The flag comes from a Boolean column, or the rule max(water mask of the
candidates) == 1. Cell output columns: ID, `cell_geometry` (WKB,
EPSG:4326), `n_cells`, `n_touching`, `wm_value`, `cell_method`,
`cell_elev_m`.

**Parallel work:** jobs = blocks of `days_per_request` dates × reservoir
chunks (`features_per_request` each), run on `n_workers` processes. All
days of a block are bands of one image, so one request per source covers
the block. A day with no URMA image is reported and skipped (the rest of the
block is kept). Rate-limit (429) and
transient errors are retried with exponential backoff. "User memory limit
exceeded" is not retried: lower `features_per_request` or raise `tile_scale`.

## Checks done on Earth Engine (check scripts since removed)

- PRISM 800 m grid: identical on 549 dates, 1981–2026.
- URMA grid: all 17,089 daily images 1980-01-01 to 2026-10-14 on the same
  pixel lattice as the 2026 reference image (same projection, pixel size,
  origin whole pixels apart), with every band extraction reads; no dates
  missing in that range.
- URMA cells: on 2,000 sampled reservoirs, Earth Engine's unweighted pixel
  count over each cell polygon equals `n_cells` for all 2,000; `ELEV_M`
  vs the previous cells file: median difference 0 m, 95th percentile 0 m,
  max 81 m.
- Stacked days, PRISM pixels and precip values (v0.6 validation): passed.
- PRISM cells, current rule (largest overlap), 2,000 sampled reservoirs:
  Earth Engine's pixel count equals the selected cells for 1,998, fewer
  for 2, more for none; stacking 5 days vs 1 day identical (max diff
  8.9e-16). 20200715: all 1,173
  single-cell reservoirs get exactly their assigned pixel's precip (max
  |diff| 0; 573 of them with rain > 0), both with single cells sent as
  points (earlier) and as their squares (current).

## Not yet verified

- The `features_per_request` (1000) default is a guess; it needs a
  benchmark. Extraction chunks are also capped by
  `max_request_bytes` (4 MB of geometry): a full-set run with 1000
  reservoirs per request hit Earth Engine's 10,485,760-byte request limit
  on 6 of 45 chunks. Requests that still exceed it are split automatically.
- `days_per_request` beyond 5 is not benchmarked.
- `plot_cells` with Esri imagery has only been rendered with a stand-in
  image (the imagery server is not reachable from where it was written);
  the layout, cells, scale bar and legend were checked.
- Medians in the overlap/elevation rule are numpy medians; urma_cells.py
  used Earth Engine's percentile reducer, which may treat ties / even
  counts differently.

## Tests

`pytest tests/` covers dates, geometry parsing/repair, internal point and
output assembly, cell-result parsing, request splitting, the file-based
cell workflows, PRISM cell selection, the per-(reservoir, day) precip
fallback, day blocks, cell maps (no imagery) and an end-to-end run with Earth Engine replaced (no
Earth Engine).

Scripts: `scripts/run_urma_cells.py` and `scripts/run_prism_cells.py` run
cell selection on a file; `scripts/example_extract_forcing.py` runs the
forcing extraction; `scripts/plot_reservoir_cells.py` makes cell maps.
 
