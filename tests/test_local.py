"""Tests for the parts that do not need Earth Engine.

Run either way, no install needed:
  - in Spyder / plain Python: run this file (it calls pytest on itself)
  - from a terminal:          pytest tests/
"""

import sys
from pathlib import Path

# Use the package straight from this source folder (no pip install).
# tests/ sits next to src/, so src/ is one level up from this file.
SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import datetime as dt

import math

import numpy as np
import polars as pl
import pytest
import shapely
import shapely.affinity
from shapely.geometry import Polygon

from resdlem_force.config import RunConfig
from resdlem_force.dates import label, normalize_dates
from resdlem_force.geometry import (chunked, internal_point_lat, load_geometries,
                                 reproject, to_geojson)


# --- dates ------------------------------------------------------------------
def test_single_date_formats():
    assert normalize_dates('2020-07-15') == [dt.date(2020, 7, 15)]
    assert normalize_dates('20200715') == [dt.date(2020, 7, 15)]
    assert normalize_dates(dt.datetime(2020, 7, 15, 6)) == [dt.date(2020, 7, 15)]


def test_list_is_sorted_and_deduplicated():
    out = normalize_dates(['2020-07-16', '20200715', '2020-07-15'])
    assert out == [dt.date(2020, 7, 15), dt.date(2020, 7, 16)]


def test_start_end_inclusive():
    out = normalize_dates(start='2020-02-28', end='2020-03-01')
    assert out == [dt.date(2020, 2, 28), dt.date(2020, 2, 29),
                   dt.date(2020, 3, 1)]


def test_date_argument_errors():
    with pytest.raises(ValueError):
        normalize_dates()
    with pytest.raises(ValueError):
        normalize_dates('2020-01-01', start='2020-01-01', end='2020-01-02')
    with pytest.raises(ValueError):
        normalize_dates(start='2020-01-02', end='2020-01-01')
    with pytest.raises(ValueError):
        normalize_dates('01/02/2020')


def test_label():
    assert label(dt.date(2026, 10, 9)) == '20261009'


# --- geometry ---------------------------------------------------------------
SQUARE = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
BOWTIE = Polygon([(0, 0), (1000, 1000), (1000, 0), (0, 1000)])  # invalid
# C shape: centroid falls in the empty middle.
C_SHAPE = Polygon([(0, 0), (3000, 0), (3000, 500), (500, 500), (500, 2500),
                   (3000, 2500), (3000, 3000), (0, 3000)])


def _df(geoms, ids=None, as_wkt=False):
    ids = ids or [f'r{i}' for i in range(len(geoms))]
    vals = [shapely.to_wkt(g) if as_wkt else shapely.to_wkb(g) for g in geoms]
    return pl.DataFrame({'id': ids, 'geometry': vals})


def test_wkb_and_wkt_parse_identically():
    a = load_geometries(_df([SQUARE]), 'id', 'geometry', 'EPSG:5070')
    b = load_geometries(_df([SQUARE], as_wkt=True), 'id', 'geometry',
                        'EPSG:5070')
    assert a.geoms[0].equals(b.geoms[0])


def test_invalid_geometry_is_repaired():
    gs = load_geometries(_df([SQUARE, BOWTIE]), 'id', 'geometry', 'EPSG:5070')
    assert gs.repaired_ids == ['r1']
    assert all(shapely.is_valid(gs.geoms))


def test_duplicate_and_null_ids_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        load_geometries(_df([SQUARE, SQUARE], ids=['a', 'a']), 'id',
                        'geometry', 'EPSG:5070')
    df = pl.DataFrame({'id': [None], 'geometry': [shapely.to_wkb(SQUARE)]})
    with pytest.raises(ValueError, match='null'):
        load_geometries(df, 'id', 'geometry', 'EPSG:5070')


def test_numeric_ids_become_strings():
    df = pl.DataFrame({'id': [7], 'geometry': [shapely.to_wkb(SQUARE)]})
    gs = load_geometries(df, 'id', 'geometry', 'EPSG:5070')
    assert gs.ids == ['7']


def test_internal_point_used_when_centroid_outside():
    # Place shapes near Reno in EPSG:5070 so latitude is realistic.
    shift = (-2_000_000, 2_000_000)
    shapes = [shapely.affinity.translate(g, *shift) for g in (SQUARE, C_SHAPE)]
    assert not shapes[1].centroid.within(shapes[1])  # premise of the test
    gs = load_geometries(_df(shapes), 'id', 'geometry', 'EPSG:5070')
    lat = internal_point_lat(gs)
    assert lat.shape == (2,) and np.all(np.isfinite(lat))
    # The lat must belong to a point inside the polygon: rebuild it.
    assert 30 < lat[1] < 50


def test_internal_point_from_geographic_input():
    poly_ll = Polygon([(-120, 39), (-119.9, 39), (-119.9, 39.1), (-120, 39.1)])
    gs = load_geometries(_df([poly_ll]), 'id', 'geometry', 'EPSG:4326')
    lat = internal_point_lat(gs)
    assert 39.0 < lat[0] < 39.1


def test_reproject_round_trip():
    poly_ll = Polygon([(-120, 39), (-119.9, 39), (-119.9, 39.1), (-120, 39.1)])
    arr = np.array([poly_ll], dtype=object)
    back = reproject(reproject(arr, 'EPSG:4326', 'EPSG:5070'), 'EPSG:5070',
                     'EPSG:4326')
    assert shapely.equals_exact(back[0], poly_ll, tolerance=1e-9)


def test_geojson_uses_lists():
    gj = to_geojson(np.array([SQUARE], dtype=object))[0]
    assert gj['type'] == 'Polygon'
    assert isinstance(gj['coordinates'][0][0], list)


def test_chunked():
    assert chunked(list(range(5)), 2) == [[0, 1], [2, 3], [4]]


# --- config -----------------------------------------------------------------
def test_runconfig_validation():
    RunConfig(project='x')
    with pytest.raises(ValueError):
        RunConfig(project='')
    with pytest.raises(ValueError):
        RunConfig(project='x', n_workers=0)


# --- output assembly (no Earth Engine) ---------------------------------------
def test_finalize_schema_and_join():
    from resdlem_force.config import ID_PROP, OUTPUT_COLS_AFTER_ID, URMA_OUTPUT
    from resdlem_force.extract import Plan, _finalize

    row = {ID_PROP: ['a', 'b'], 'date': ['20200715'] * 2,
           'pr': ['1.5', None], 'pr_source': ['prism800', None],
           'urma_status': ['permanent'] * 2}
    for c in URMA_OUTPUT.values():
        row[c] = ['1', '2']           # CSV columns can arrive as text
    out = _finalize(pl.DataFrame(row), Plan('OWEN_ID'))
    assert out.columns == ['OWEN_ID'] + OUTPUT_COLS_AFTER_ID
    assert 'elev_m' not in out.columns and 'lat' not in out.columns
    import datetime as dt
    assert out['processed_date'].dtype == pl.Utf8
    assert out['processed_date'].to_list() == [dt.date.today().isoformat()] * 2
    assert out['tmin'].dtype == pl.Float64
    assert out['pr'].to_list() == [1.5, None]


def test_finalize_without_precip_drops_precip_columns():
    from resdlem_force.config import ID_PROP, OUTPUT_COLS_AFTER_ID, URMA_OUTPUT
    from resdlem_force.extract import Plan, _finalize

    row = {ID_PROP: ['a'], 'date': ['20200715'], 'urma_status': ['permanent']}
    for c in URMA_OUTPUT.values():
        row[c] = ['1']
    out = _finalize(pl.DataFrame(row), Plan('OWEN_ID', precip=False))
    expected = [c for c in OUTPUT_COLS_AFTER_ID
                if c not in ('pr', 'pr_source')]
    assert out.columns == ['OWEN_ID'] + expected


# --- request splitting (no Earth Engine) ---------------------------------------
def test_solve_split_halves_too_big_requests():
    from resdlem_force.ee_utils import solve_split
    calls = []

    def fetch(items):
        calls.append(len(items))
        if len(items) > 2:
            raise RuntimeError('Computation timed out.')
        if 'bad' in items:
            raise RuntimeError('Something else')
        return [i * 10 for i in items if i != 'bad']

    out, fails = solve_split(fetch, [1, 2, 3, 4, 5])
    assert sorted(out) == [10, 20, 30, 40, 50]
    assert fails == []
    out, fails = solve_split(fetch, [1, 'bad'])
    assert out == [] and [f[0] for f in fails] == [1, 'bad']



# --- URMA cell selection, local (synthetic cell values, no Earth Engine) ------
def _values(w=20, h=20, elev=None, wm=None):
    """CellValues on a 1 km test grid in EPSG:5070 with origin (0, 0),
    rows going south."""
    from resdlem_force.cells import CellValues
    from resdlem_force.grid_cells import Grid
    grid = Grid('EPSG:5070', 1000.0, 0.0, -1000.0, 0.0)
    e = np.zeros((h, w)) if elev is None else elev
    m = np.zeros((h, w)) if wm is None else wm
    return CellValues(grid, 0, 0, e, m)


def _cell_box(col0, row0, ncol, nrow, pad=0.0):
    """Box over cells col0.. / row0.. of the test grid (+pad metres)."""
    return shapely.box(col0 * 1000 - pad, -(row0 + nrow) * 1000 - pad,
                       (col0 + ncol) * 1000 + pad, -row0 * 1000 + pad)


def test_urma_select_water_mask_rule():
    from resdlem_force.cells import select_one
    wm = np.zeros((20, 20))
    wm[5, 5] = wm[5, 6] = 1                              # two water cells
    v = _values(wm=wm)
    geom = _cell_box(4, 4, 4, 3, pad=-100)               # 4x3 cells, inset
    r = select_one(geom, v)
    assert r['n_touching'] == 12
    assert r['cell_method'] == 'water_mask' and r['wm_value'] == 1.0
    assert r['n_cells'] == 2
    assert r['geom'].equals(_cell_box(5, 5, 2, 1))      # exact pixel squares


def test_urma_select_overlap_then_elevation():
    from resdlem_force.cells import select_one
    elev = np.zeros((20, 20))
    elev[5:8, 5:8] = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
    v = _values(elev=elev)
    # covers cells 5..7 fully, plus a 200 m sliver of column 8
    geom = shapely.box(5000, -8000, 8200, -5000)
    r = select_one(geom, v)
    assert r['cell_method'] == 'overlap_elevation' and r['wm_value'] == 0.0
    assert r['n_touching'] == 12                         # 4 columns x 3 rows
    # overlap >= median keeps the 9 full cells; elev <= median (50) keeps 5
    assert r['n_cells'] == 5
    assert r['cell_elev_m'] == pytest.approx(np.mean([10, 20, 30, 40, 50]))
    # touching only at an edge does not count
    edge = _cell_box(5, 5, 1, 1)
    assert select_one(edge, v)['n_touching'] == 1


def test_urma_select_flag_column_and_fallback():
    from resdlem_force.cells import select_one
    v = _values()                                         # no water anywhere
    geom = _cell_box(2, 2, 2, 2, pad=-10)
    r = select_one(geom, v, flag=1)
    assert r['cell_method'] == 'water_mask_fallback' and r['n_cells'] == 4
    r0 = select_one(geom, v, flag=0)
    assert r0['cell_method'] == 'overlap_elevation' and r0['wm_value'] == 0.0
    # outside the value arrays: no water-mask value -> rule flag None
    far = shapely.affinity.translate(geom, 100_000, 0)
    rf = select_one(far, v)
    assert rf['wm_value'] is None and rf['cell_elev_m'] is None


def test_select_cells_frame_local(monkeypatch):
    import resdlem_force.cells as cells_mod
    v = _values()
    calls = []
    monkeypatch.setattr(cells_mod, 'fetch_cell_values',
                        lambda cfg: calls.append(1) or v)
    native = _df([_cell_box(1, 1, 2, 2, pad=-50), _cell_box(10, 10, 1, 1,
                  pad=-300)], ids=['b', 'a'])
    out = cells_mod.select_urma_cells(native, id_col='id', crs='EPSG:5070',
                                      config=RunConfig(project='x'))
    assert calls == [1]                                  # one download
    assert out['id'].to_list() == ['a', 'b']             # sorted
    assert out['n_cells'].to_list() == [1, 4]
    g = shapely.from_wkb(out.filter(pl.col('id') == 'b')['cell_geometry'][0])
    back = reproject(np.array([g], dtype=object), 'EPSG:4326', 'EPSG:5070')[0]
    assert back.symmetric_difference(_cell_box(1, 1, 2, 2)).area < 1e-3
    flagged = native.with_columns(pl.Series('wm', [True, False]))
    out2 = cells_mod.select_urma_cells(flagged, id_col='id', crs='EPSG:5070',
                                       config=RunConfig(project='x'),
                                       water_mask_col='wm')
    assert dict(zip(out2['id'], out2['cell_method'])) == {
        'a': 'overlap_elevation', 'b': 'water_mask_fallback'}


# --- cells_from_file (no Earth Engine: select_urma_cells is faked) ---------------
def _fake_select_cells(record):
    """Fake select_urma_cells: 'cells' = the reservoir polygon itself (EPSG:4326);
    reservoir 'b' gets no cells; 'a' is water-mask."""
    from resdlem_force.cells import CELL_SCHEMA
    from resdlem_force.geometry import load_geometries

    def fake(native, *, id_col, crs, config, water_mask_col=None,
             geometry_col='geometry', out_dir=None):
        record.update(water_mask_col=water_mask_col, native=native)
        gs = load_geometries(native, id_col, geometry_col, crs)
        ll = reproject(gs.geoms, gs.crs, 'EPSG:4326')
        rows = []
        for i, g in zip(gs.ids, ll):
            rows.append({id_col: i,
                         'cell_geometry': None if i == 'b' else shapely.to_wkb(g),
                         'n_cells': 0 if i == 'b' else 1, 'n_touching': 1,
                         'wm_value': 1.0 if i == 'a' else 0.0,
                         'cell_method': 'water_mask' if i == 'a'
                         else 'overlap_elevation',
                         'cell_elev_m': None if i == 'b' else 100.0})
        return pl.DataFrame(rows, schema={id_col: pl.Utf8, **CELL_SCHEMA})
    return fake


def _reservoir_file(tmp_path, kind):
    gpd = pytest.importorskip('geopandas')
    base = (-2_000_000, 2_000_000)
    geoms = [shapely.affinity.translate(SQUARE, base[0] + 5000 * k, base[1])
             for k in range(3)]
    gdf = gpd.GeoDataFrame({'OWEN_ID': ['a', 'b', 'c'], 'SORT_ID': [3, 1, 2],
                            'NAME': ['x', 'y', 'z'], 'flag': [1, 0, 1]},
                           geometry=geoms, crs='EPSG:5070')
    path = tmp_path / f'res.{kind}'
    if kind == 'gpkg':
        gdf.to_file(path, layer='reservoirs', driver='GPKG')
    else:
        gdf.to_file(path)
    return gdf, path


@pytest.mark.parametrize('kind', ['shp', 'gpkg'])
def test_select_urma_cells_output(tmp_path, monkeypatch, kind):
    """Path input -> GeoDataFrame: originals + N_CELLS, ELEV_M, WATER_MASK;
    geometry = the cells in the input CRS; nothing written to disk."""
    pytest.importorskip('geopandas')
    import resdlem_force
    import resdlem_force.files as files
    rec = {}
    monkeypatch.setattr(files, 'select_cells_frame', _fake_select_cells(rec))
    src, path = _reservoir_file(tmp_path, kind)
    before = sorted(p.name for p in tmp_path.iterdir())
    out = resdlem_force.select_urma_cells(
        path, id_col='OWEN_ID', config=RunConfig(project='x'),
        layer='reservoirs' if kind == 'gpkg' else None)
    assert sorted(p.name for p in tmp_path.iterdir()) == before   # no files
    assert rec['water_mask_col'] is None                 # EE rule by default
    assert list(out.columns) == ['OWEN_ID', 'SORT_ID', 'NAME', 'flag',
                                 'N_CELLS', 'ELEV_M', 'WATER_MASK',
                                 'geometry']
    assert out.crs.equals(src.crs)                        # input CRS kept
    by_id = {r['OWEN_ID']: r for _, r in out.iterrows()}
    assert by_id['a']['WATER_MASK'] == 1 and by_id['c']['WATER_MASK'] == 0
    assert by_id['b']['WATER_MASK'] == 0 and by_id['b']['geometry'] is None
    assert by_id['a']['ELEV_M'] == 100.0 and by_id['a']['N_CELLS'] == 1
    orig = dict(zip(src['OWEN_ID'], src.geometry))
    assert by_id['a']['geometry'].normalize().equals_exact(
        orig['a'].normalize(), tolerance=1e-3)


def test_select_urma_cells_flag_column_and_errors(tmp_path, monkeypatch):
    pytest.importorskip('geopandas')
    import resdlem_force
    import resdlem_force.files as files
    rec = {}
    monkeypatch.setattr(files, 'select_cells_frame', _fake_select_cells(rec))
    _, path = _reservoir_file(tmp_path, 'shp')
    resdlem_force.select_urma_cells(path, id_col='OWEN_ID',
                                   config=RunConfig(project='x'),
                                   water_mask_col='flag')
    assert rec['water_mask_col'] == 'flag'
    assert rec['native']['flag'].to_list() == [True, False, True]
    for fn in (resdlem_force.select_urma_cells,
               resdlem_force.select_prism_cells):
        with pytest.raises(TypeError):
            fn(123, id_col='OWEN_ID', config=RunConfig(project='x'))
        with pytest.raises(ValueError, match='crs'):
            fn(_df([SQUARE]), id_col='id', config=RunConfig(project='x'))


def test_urma_and_prism_cell_functions_same_format():
    """Same arguments (URMA adds water_mask_col), same kind of output."""
    import inspect
    import resdlem_force
    u = inspect.signature(resdlem_force.select_urma_cells).parameters
    p = inspect.signature(resdlem_force.select_prism_cells).parameters
    assert [k for k in u if k != 'water_mask_col'] == list(p)


# --- extract input: one table, two geometry columns (no Earth Engine) -------
BIG = Polygon([(0, 0), (4000, 0), (4000, 4000), (0, 4000)])  # 16 km2 > 1 cell


# URMA reference grid as Earth Engine reports it (probe_urma_crs.py,
# 2026-10-15): NWS CONUS Lambert, 2539.703 m pixels. The WKT names a WGS 84
# GEOGCS but projects on a 6,371,200 m sphere; Earth Engine uses the sphere.
URMA_WKT = '''PROJCS["NWS CONUS",
  GEOGCS["WGS 84",
    DATUM["World Geodetic System 1984",
      SPHEROID["WGS 84", 6378137.0, 298.257223563, AUTHORITY["EPSG","7030"]],
      AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]],
    UNIT["degree", 0.017453292519943295],
    AXIS["Geodetic longitude", EAST],
    AXIS["Geodetic latitude", NORTH],
    AUTHORITY["EPSG","4326"]],
  PROJECTION["Lambert_Conformal_Conic_1SP"],
  PARAMETER["semi_major", 6371200.0],
  PARAMETER["semi_minor", 6371200.0],
  PARAMETER["central_meridian", -95.0],
  PARAMETER["latitude_of_origin", 25.0],
  PARAMETER["scale_factor", 1.0],
  PARAMETER["false_easting", 0.0],
  PARAMETER["false_northing", 0.0],
  UNIT["m", 1.0],
  AXIS["x", EAST],
  AXIS["y", NORTH]]'''
URMA_SPHERE = ('+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 +lon_0=-95 +x_0=0 '
               '+y_0=0 +R=6371200 +units=m +no_defs')
URMA_TRANSFORM = [2539.703, 0, -3272417.139794261, 0, -2539.703,
                  3790838.3367873165]
URMA_TEST_CRS = URMA_SPHERE


def _urma_grid():
    from resdlem_force.grid_cells import Grid
    return Grid.from_projection({'wkt': URMA_WKT,
                                 'transform': URMA_TRANSFORM})


def _urma_cells(nx, ny, col=0):
    """An nx x ny block of URMA grid cells (EPSG:4326 polygon), near
    -110, 40; `col` shifts the block east by whole cells."""
    import pyproj
    g = _urma_grid()
    fwd = pyproj.Transformer.from_crs('EPSG:4326', g.crs, always_xy=True)
    c0, r0 = g.col_row(*fwd.transform(-110, 40))
    c0 += col * 10
    sq = shapely.union_all(g.squares(
        np.repeat(np.arange(c0, c0 + nx), ny),
        np.tile(np.arange(r0, r0 + ny), nx)))
    return reproject(np.array([sq], dtype=object), g.crs, 'EPSG:4326')[0]


def _two_geom_inputs():
    """Same 5 reservoirs as geopandas (URMA cells EPSG:4326, PRISM cells
    EPSG:4269 -- different CRSs on purpose) and as polars (WKB). No
    reservoir polygon: extraction only takes the cells.
      a : URMA cells 2x2, PRISM cells 2x2
      b : NO URMA cells                     -> excluded
      c : URMA cells 1 cell, PRISM 1 cell
      d : NO URMA cells                     -> excluded
      e : URMA cells 1 cell, PRISM 1 cell
    Returns (gdf, pdf, urma_cells_ll)."""
    gpd = pytest.importorskip('geopandas')
    from resdlem_force import grid_cells as gc
    urma_ll = [_urma_cells(2, 2, 0), None, _urma_cells(1, 1, 2), None,
               _urma_cells(1, 1, 4)]
    pg = _prism_grid()
    c0, r0 = pg.col_row(-110.0, 40.0)
    prism = [gc.cell_geometry([(c0, r0), (c0 + 1, r0), (c0, r0 + 1),
                               (c0 + 1, r0 + 1)], pg), None,
             gc.cell_geometry([(c0 + 20, r0)], pg), None,
             gc.cell_geometry([(c0 + 40, r0)], pg)]
    ids = ['a', 'b', 'c', 'd', 'e']
    gdf = gpd.GeoDataFrame({'OWEN_ID': ids},
                           geometry=gpd.GeoSeries(urma_ll, crs='EPSG:4326'))
    gdf = gdf.rename_geometry('urma_cells')
    gdf['prism_cells'] = gpd.GeoSeries(prism, crs='EPSG:4269')
    pdf = pl.DataFrame({
        'OWEN_ID': ids,
        'urma_cells': [None if g is None else shapely.to_wkb(g)
                       for g in urma_ll],
        'prism_cells': [None if g is None else shapely.to_wkb(g)
                        for g in prism]})
    return gdf, pdf, urma_ll


# PRISM 800 m grid as Earth Engine reports it (checked 2026-10-13)
PRISM800 = dict(crs='EPSG:4269', a=0.008333333333, c=-125.02083333333351,
                e=-0.008333333333, f=49.9374999999995)


def _prism_grid():
    from resdlem_force.grid_cells import Grid
    return Grid(**PRISM800)


def _stub_ee(monkeypatch):
    # resdlem_force.extract is also the name of the public function, which
    # shadows the submodule as a package attribute -- fetch the module itself.
    import importlib
    ex = importlib.import_module('resdlem_force.extract')
    calls = []
    monkeypatch.setattr(ex, 'init_ee', lambda *a, **k: None)
    monkeypatch.setattr(ex.grids, 'grid_from_ee',
                        lambda source, cfg: calls.append(source)
                        or {'urma': _urma_grid(),
                            'prism800': _prism_grid()}[source])
    ex._grid_calls = calls
    return ex


def test_extract_prepare_geopandas_and_polars_agree(monkeypatch):
    ex = _stub_ee(monkeypatch)
    gdf, pdf, urma_ll = _two_geom_inputs()
    cfg = RunConfig(project='x', features_per_request=1)
    kw = dict(id_col='OWEN_ID', urma_col='urma_cells',
              prism_col='prism_cells', dates=['2020-07-15', '2020-07-16'],
              start=None, end=None, config=cfg)
    a = ex._prepare(gdf, urma_crs=None, prism_crs=None, **kw)
    b = ex._prepare(pdf, urma_crs='EPSG:4326', prism_crs='EPSG:4269', **kw)

    for plan in (a, b):
        assert plan.excluded_ids == ['b', 'd']      # no URMA cells
        sent = sorted({i for j in plan.jobs for i, _ in j['cells']})
        assert sent == ['a', 'c', 'e']
        # 2 dates fit one block (days_per_request=5) x 3 chunks
        assert len(plan.jobs) == 3
        assert all(j['labels'] == ['20200715', '20200716'] for j in plan.jobs)
    ja = {i: g for j in a.jobs for i, g in j['cells']}
    jb = {i: g for j in b.jobs for i, g in j['cells']}
    assert ja.keys() == jb.keys() == {'a', 'c', 'e'}
    for i, cell in (('a', urma_ll[0]), ('c', urma_ll[2]), ('e', urma_ll[4])):
        assert ja[i]['type'] == 'Polygon'           # sent as the cells
        assert shapely.geometry.shape(ja[i]).equals_exact(cell, 1e-9)
        assert shapely.geometry.shape(ja[i]).equals_exact(
            shapely.geometry.shape(jb[i]), 1e-9)
    # precip goes over the given PRISM 800 m cells (EPSG:4269 -> 4326)
    assert set(a.jobs[0]['grid']) == {'prism800'}
    pa = {i: g for j in a.jobs for i, g in j['grid']['prism800']}
    pb = {i: g for j in b.jobs for i, g in j['grid']['prism800']}
    prism = dict(zip(gdf['OWEN_ID'], gdf['prism_cells']))
    for i in ('a', 'c', 'e'):
        assert shapely.geometry.shape(pa[i]).equals_exact(
            shapely.geometry.shape(pb[i]), 1e-9)
        assert shapely.geometry.shape(pa[i]).equals(prism[i])
    assert ex._grid_calls == []      # no grid read: cells used as given


def test_days_per_request_blocks(monkeypatch):
    ex = _stub_ee(monkeypatch)
    gdf, _, _ = _two_geom_inputs()
    kw = dict(id_col='OWEN_ID', urma_col='urma_cells',
              prism_col='prism_cells', start='2020-07-01', end='2020-07-12',
              dates=None, urma_crs=None, prism_crs=None)
    plan = ex._prepare(gdf, config=RunConfig(project='x', days_per_request=5),
                       **kw)
    blocks = sorted({tuple(j['labels']) for j in plan.jobs})
    assert [len(b) for b in blocks] == [5, 5, 2]
    assert blocks[0][0] == '20200701' and blocks[-1][-1] == '20200712'
    assert sum(blocks, ()) == tuple(sorted(sum(blocks, ())))
    one = ex._prepare(gdf, config=RunConfig(project='x', days_per_request=1),
                      **kw)
    assert len({tuple(j['labels']) for j in one.jobs}) == 12
    assert ex._reservoir_days(plan.jobs) == ex._reservoir_days(one.jobs) \
        == 3 * 12


def test_single_cell_sent_as_polygon_with_one_pixel_centre(monkeypatch):
    """1 cell and 2 cells are both sent as polygons; the single cell's
    polygon (in the URMA projection) holds exactly one pixel centre, the
    2-cell polygon exactly two -- what an unweighted reduction uses."""
    ex = _stub_ee(monkeypatch)
    one, two = _urma_cells(1, 1, 0), _urma_cells(2, 1, 3)
    pdf = pl.DataFrame({'id': ['one', 'two'],
                        'cells': [shapely.to_wkb(one), shapely.to_wkb(two)],
                        })
    plan = ex._prepare(pdf, id_col='id', urma_col='cells', prism_col=None,
                       urma_crs='EPSG:4326', prism_crs=None,
                       dates='2020-07-15', start=None, end=None,
                       config=RunConfig(project='x'), precip=False)
    geo = {i: g for j in plan.jobs for i, g in j['cells']}
    assert {i: g['type'] for i, g in geo.items()} == {
        'one': 'Polygon', 'two': 'Polygon'}
    ug = _urma_grid()
    for i, n in (('one', 1), ('two', 2)):
        poly = reproject(np.array([shapely.geometry.shape(geo[i])],
                                  dtype=object), 'EPSG:4326', ug.crs)[0]
        x0, y0, x1, y1 = poly.bounds
        c0, r0 = ug.col_row(x0, y1)
        c1, r1 = ug.col_row(x1, y0)
        centres = [shapely.Point(*ug.centre(c, r))
                   for c in range(c0 - 1, c1 + 2)
                   for r in range(r0 - 1, r1 + 2)]
        assert sum(poly.contains(p) for p in centres) == n


def test_extract_prepare_without_precip(monkeypatch):
    """precip=False: no PRISM cells needed or sent; same met geometry and
    meta as with precip."""
    ex = _stub_ee(monkeypatch)
    gdf, _, _ = _two_geom_inputs()
    kw = dict(id_col='OWEN_ID', urma_col='urma_cells',
              dates='2020-07-15', start=None, end=None,
              urma_crs=None, prism_crs=None, config=RunConfig(project='x'))
    off = ex._prepare(gdf.drop(columns='prism_cells'), prism_col=None,
                      precip=False, **kw)
    assert off.precip is False
    assert all(j['grid'] == {} and j['precip'] is False for j in off.jobs)
    on = ex._prepare(gdf, prism_col='prism_cells', **kw)
    assert on.precip is True
    cells_off = {i: g for j in off.jobs for i, g in j['cells']}
    cells_on = {i: g for j in on.jobs for i, g in j['cells']}
    assert cells_off == cells_on


def test_extract_prepare_needs_prism_cells_for_precip(monkeypatch):
    ex = _stub_ee(monkeypatch)
    gdf, _, _ = _two_geom_inputs()
    kw = dict(id_col='OWEN_ID', urma_col='urma_cells',
              dates='2020-07-15', start=None, end=None,
              urma_crs=None, prism_crs=None, config=RunConfig(project='x'))
    with pytest.raises(ValueError, match='prism_cells_col'):
        ex._prepare(gdf, prism_col=None, **kw)
    bad = gdf.copy()
    bad.loc[bad['OWEN_ID'] == 'c', 'prism_cells'] = None
    with pytest.raises(ValueError, match="no 'prism_cells'"):
        ex._prepare(bad, prism_col='prism_cells', **kw)


def gpd_series(wkb, crs, index):
    gpd = pytest.importorskip('geopandas')
    return gpd.GeoSeries([shapely.from_wkb(w) for w in wkb], crs=crs,
                         index=index)


def test_points_are_sent_without_geodesic(monkeypatch):
    """Earth Engine's Point constructor rejects a geodesic argument; polygons
    keep geodesic=False (planar edges)."""
    import importlib
    ex = importlib.import_module('resdlem_force.extract')
    calls = []
    monkeypatch.setattr(ex.ee, 'Geometry',
                        lambda *a, **k: calls.append((a, k)) or a[0])
    monkeypatch.setattr(ex.ee, 'Feature', lambda g, p: (g, p))
    monkeypatch.setattr(ex.ee, 'FeatureCollection', lambda f: f)
    pt = {'type': 'Point', 'coordinates': [-110.0, 40.0]}
    poly = {'type': 'Polygon',
            'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    ex._fc([('p', pt), ('q', poly)])
    assert calls[0] == ((pt,), {})
    assert calls[1] == ((poly, None, False), {})


def test_extract_prepare_errors(monkeypatch):
    ex = _stub_ee(monkeypatch)
    gdf, pdf, _ = _two_geom_inputs()
    cfg = RunConfig(project='x')
    kw = dict(id_col='OWEN_ID', urma_col='urma_cells',
              prism_col='prism_cells', dates='2020-07-15', start=None, end=None,
              config=cfg)
    with pytest.raises(ValueError, match='urma_crs'):      # polars needs crs
        ex._prepare(pdf, urma_crs=None, prism_crs=None, **kw)
    with pytest.raises(KeyError):                          # wrong column name
        ex._prepare(gdf, urma_crs=None, prism_crs=None,
                    **{**kw, 'urma_col': 'nope'})
    with pytest.raises(TypeError):
        ex._prepare({'not': 'a frame'}, urma_crs=None, prism_crs=None, **kw)
    nan_id = gdf.copy()
    nan_id['OWEN_ID'] = ['a', float('nan'), 'c', 'd', 'e']  # missing ID
    with pytest.raises(ValueError, match='null IDs'):
        ex._prepare(nan_id, urma_crs=None, prism_crs=None, **kw)


# --- PRISM grid cells (no Earth Engine) ---------------------------------------
def _px(grid, col, row, dx0=0.0, dy0=0.0, dx1=1.0, dy1=1.0):
    """Box in pixel units: cols col+dx0..col+dx1, rows row+dy0..row+dy1."""
    x0 = grid.c + (col + dx0) * grid.a
    x1 = grid.c + (col + dx1) * grid.a
    y0 = grid.f + (row + dy0) * grid.e
    y1 = grid.f + (row + dy1) * grid.e
    return shapely.box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def test_prism_cells_rule():
    from resdlem_force import grid_cells as gc
    grid = _prism_grid()
    # exactly 3x2 pixels: those 6 only (neighbours touch, do not count)
    rect = _px(grid, 1000, 500, 0, 0, 3, 2)
    cells, single = gc.select(rect, rect.centroid, grid)
    assert not single and sorted(cells) == sorted(
        (c, r) for c in range(1000, 1003) for r in range(500, 502))
    g = gc.cell_geometry(cells, grid)
    assert g.geom_type == 'Polygon' and g.equals(rect)
    # 2.3 x 0.6 pixels = 1.38 pixels of area -> 2 pixels: the two 0.6-overlap
    # pixels, not the 0.18-overlap third column
    wide = _px(grid, 1000, 500, 0.0, 0.2, 2.3, 0.8)
    cells, single = gc.select(wide, wide.centroid, grid)
    assert not single and sorted(cells) == [(1000, 500), (1001, 500)]
    # area exactly 2 pixels -> exactly 2 pixels (no extra one)
    two = _px(grid, 1000, 500, 0.0, 0.0, 2.0, 1.0)
    assert len(gc.select(two, two.centroid, grid)[0]) == 2
    # 2.01 pixels of area -> 3 pixels (always add one while short)
    over = _px(grid, 1000, 500, 0.0, 0.0, 2.01, 1.0)
    cells, _ = gc.select(over, over.centroid, grid)
    assert sorted(cells) == [(1000, 500), (1001, 500), (1002, 500)]
    # smaller than one pixel, straddling 4 pixels -> the pixel it overlaps
    # most: x 0.6..1.3, y 0.6..1.3 -> (1000,500) 0.16, (1001,500) 0.12,
    # (1000,501) 0.12, (1001,501) 0.09
    small = _px(grid, 1000, 500, 0.6, 0.6, 1.3, 1.3)
    cells, single = gc.select(small, small.centroid, grid)
    assert single and cells == [(1000, 500)]
    sq = gc.cell_geometry(cells, grid)               # stored: the square
    assert sq.geom_type == 'Polygon' and sq.equals(_px(grid, 1000, 500))
    # sent: the square, which holds exactly one pixel centre (its own)
    sent = gc.to_wgs84_geojson(np.array([sq], dtype=object), grid)[0]
    assert sent['type'] == 'Polygon'
    back = shapely.geometry.shape(sent)
    centres = [shapely.Point(*grid.centre(c, r)) for c in range(999, 1002)
               for r in range(499, 502)]
    assert [back.contains(p) for p in centres].count(True) == 1
    assert back.contains(shapely.Point(*grid.centre(1000, 500)))


def test_prism_overlaps_and_ring():
    """overlaps(): full pixels get the pixel area, edge pixels their
    intersection; total = polygon area. A ring (lake with an island) never
    takes the island pixel ahead of water pixels."""
    from resdlem_force import grid_cells as gc
    grid = _prism_grid()
    poly = _px(grid, 1000, 500, 0.25, 0.25, 4.5, 3.75)
    cc, rr, ov = gc.overlaps(poly, grid)
    assert ov.sum() == pytest.approx(poly.area)
    assert ov.max() == pytest.approx(grid.pixel_area)
    cells, _ = gc.select(poly, poly.centroid, grid)
    assert len(cells) == math.ceil(poly.area / grid.pixel_area - 1e-9)
    # every fully covered pixel is selected
    full = {(c, r) for c, r, o in zip(cc, rr, ov)
            if o == pytest.approx(grid.pixel_area)}
    assert full <= set(cells)
    # 5x5 pixels with the centre pixel removed: 24 pixels of area, 24 cells
    ring = _px(grid, 1000, 500, 0, 0, 5, 5).difference(_px(grid, 1002, 502))
    cells, _ = gc.select(ring, ring.representative_point(), grid)
    assert len(cells) == 24 and (1002, 502) not in cells


def test_prism_cells_build_and_wgs84(monkeypatch):
    """build(): polygons follow pixel edges; sent in EPSG:4326 as
    polygons, single cells included."""
    from resdlem_force import grid_cells as gc
    from resdlem_force.geometry import internal_points
    grid = _prism_grid()
    big = _px(grid, 1000, 500, 0.1, 0.1, 3.9, 2.9)           # 4x3 pixels
    tiny = _px(grid, 1010, 500, 0.2, 0.2, 0.4, 0.4)
    gs = load_geometries(_df([big, tiny], ids=['big', 'tiny']), 'id',
                         'geometry', 'EPSG:4269')
    geoms, n, is_pt = gc.build(gs, internal_points(gs), grid)
    # big: 3.8 x 2.8 = 10.64 pixels of area -> 11 of its 12 pixels
    assert n.tolist() == [11, 1] and is_pt.tolist() == [False, True]
    gj = gc.to_wgs84_geojson(geoms, grid)               # stored and sent
    assert [g['type'] for g in gj] == ['Polygon', 'Polygon']
    assert shapely.geometry.shape(gj[1]).equals(_px(grid, 1010, 500))


def test_select_prism_cells_output(tmp_path, monkeypatch):
    """Path and GeoDataFrame input -> the same GeoDataFrame: originals +
    N_CELLS; geometry = the cell polygons in the input CRS; no files."""
    gpd = pytest.importorskip('geopandas')
    import importlib
    import resdlem_force
    pc = importlib.import_module('resdlem_force.prism_cells')
    monkeypatch.setattr(pc, 'grid_from_ee', lambda s, cfg: _prism_grid())
    grid = _prism_grid()
    shapes = [_px(grid, 1000, 500, 0.1, 0.1, 3.9, 2.9),
              _px(grid, 1010, 500, 0.2, 0.2, 0.4, 0.4)]
    src = gpd.GeoDataFrame({'OWEN_ID': ['big', 'tiny'], 'NAME': ['x', 'y']},
                           geometry=shapes, crs='EPSG:4269').to_crs(5070)
    path = tmp_path / 'res.gpkg'
    src.to_file(path, driver='GPKG')
    cfg = RunConfig(project='x')
    out = resdlem_force.select_prism_cells(path, id_col='OWEN_ID', config=cfg)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['res.gpkg']
    assert list(out.columns) == ['OWEN_ID', 'NAME', 'N_CELLS', 'geometry']
    assert out['N_CELLS'].tolist() == [11, 1]     # 10.64 px of area
    assert out.crs.equals(src.crs)
    assert out.geometry.geom_type.tolist() == ['Polygon', 'Polygon']
    again = resdlem_force.select_prism_cells(gpd.read_file(path),
                                            id_col='OWEN_ID', config=cfg)
    assert again.drop(columns='geometry').equals(out.drop(columns='geometry'))
    assert all(a.equals(b) for a, b in zip(again.geometry, out.geometry))


# --- worker: per-(reservoir, day) precip fallback (no Earth Engine) -----------
def test_extract_chunk_precip_fallback(monkeypatch):
    import importlib
    ex = importlib.import_module('resdlem_force.extract')
    from resdlem_force.config import ID_PROP
    days = ['20200715', '20200716']
    ids = ['a', 'b', 'c']
    met_geo = {i: {'type': 'Point', 'coordinates': [0, 0], 'm': i} for i in ids}
    cells = {i: {'type': 'Point', 'coordinates': [0, 0], 'p': i} for i in ids}

    def fake_met(d, features, cfg):
        return pl.DataFrame({ID_PROP: [i for i, _ in features for _ in d],
                             'date': [x for _ in features for x in d]})
    calls = []
    returns = {
        # prism800: a both days, b first day
        'p800': [('a', days[0], 1.0), ('a', days[1], 2.0),
                 ('b', days[0], 3.0)],
        # prism4k: b second day (+ a value for a/day0 that must be ignored)
        'p4k': [('b', days[1], 4.0), ('a', days[0], 99.0)],
        # urma: c first day only
        'urma': [('c', days[0], 5.0)],
    }
    from resdlem_force.config import PRISM_800_ID, PRISM_4K_ID, URMA_ID
    key = {PRISM_800_ID: 'p800', PRISM_4K_ID: 'p4k', URMA_ID: 'urma'}

    def fake_precip(coll_id, band, labels, features, cfg, grid_proj=None):
        k = key[coll_id]
        calls.append((k, list(labels), [i for i, _ in features],
                      [('p' if 'p' in g else 'm') for _, g in features],
                      grid_proj))
        rows = [r for r in returns[k] if r[1] in labels]
        return pl.DataFrame({ID_PROP: [r[0] for r in rows],
                             'date': [r[1] for r in rows],
                             'pr': [r[2] for r in rows]},
                            schema={ID_PROP: pl.Utf8, 'date': pl.Utf8,
                                    'pr': pl.Float64})
    monkeypatch.setattr(ex, '_reduce_met', fake_met)
    monkeypatch.setattr(ex, '_reduce_precip', fake_precip)
    monkeypatch.setattr(ex, '_prism800_proj', lambda: 'PRISM800_GRID')
    monkeypatch.setattr(ex.grids, 'ee_projection',
                        lambda source: f'{source.upper()}_GRID')
    info = {'urma': days, 'prism800': days, 'prism4k': days, 'urma_pcp': days}
    info['urma'] = days
    out, n_missing = ex._extract_chunk(
        RunConfig(project='x'), info, {days[0]: 'permanent'}, met_geo,
        {'prism800': cells}, ids)
    got = {(r[ID_PROP], r['date']): (r['pr'], r['pr_source'])
           for r in out.iter_rows(named=True)}
    assert got[('a', days[0])] == (1.0, 'prism800')
    assert got[('a', days[1])] == (2.0, 'prism800')
    assert got[('b', days[0])] == (3.0, 'prism800')
    assert got[('b', days[1])] == (4.0, 'prism4k')     # filled by 4 km
    assert got[('c', days[0])] == (5.0, 'urma')
    assert got[('c', days[1])] == (None, None)          # nothing anywhere
    assert n_missing == 1
    st = dict(zip(out['date'], out['urma_status']))
    assert st == {days[0]: 'permanent', days[1]: 'unknown'}
    # 800 m: all; 4 km: only what is still empty, same 800 m cells + grid;
    # urma: only c, on the URMA met geometry and the URMA reference grid
    assert calls[0] == ('p800', days, ids, ['p'] * 3, 'PRISM800_GRID')
    assert calls[1] == ('p4k', days, ['b', 'c'], ['p'] * 2, 'PRISM800_GRID')
    assert calls[2] == ('urma', days, ['c'], ['m'], 'URMA_GRID')


def test_met_and_precip_long():
    import importlib
    ex = importlib.import_module('resdlem_force.extract')
    from resdlem_force.config import ID_PROP, URMA_OUTPUT
    days = ['20200715', '20200716']
    wide = {ID_PROP: ['a', 'b']}
    for d in days:
        for k in URMA_OUTPUT:
            wide[f'd{d}_{k}'] = ['1.5', None]
    met = ex._met_long(pl.DataFrame(wide), days)
    assert met.height == 4 and set(met['date']) == set(days)
    assert met.columns == [ID_PROP, 'date'] + list(URMA_OUTPUT.values())
    assert met['tmin'].dtype == pl.Float64
    pr = ex._precip_long(pl.DataFrame({ID_PROP: ['a', 'b'],
                                       f'p{days[0]}': [1.0, None]}), days)
    assert pr.rows() == [('a', days[0], 1.0)]          # nulls dropped,
    with pytest.raises(KeyError):                      # missing day -> none
        ex._met_long(pl.DataFrame({ID_PROP: ['a']}), days)


# --- request size: chunking and splitting (no Earth Engine) -----------------
def test_chunk_by_size():
    from resdlem_force.geometry import chunk_by_size
    ids = list('abcdef')
    sizes = {'a': (3, 1), 'b': (3, 1), 'c': (5, 1), 'd': (1, 9), 'e': (1, 1),
             'f': (20, 1)}
    chunks = chunk_by_size(ids, sizes, max_items=10, max_bytes=10)
    assert chunks == [['a', 'b'], ['c', 'd'], ['e'], ['f']]
    # order kept, nothing lost, every chunk within budget (or a single id)
    assert sum(chunks, []) == ids
    for c in chunks:
        assert len(c) == 1 or all(sum(sizes[i][j] for i in c) <= 10
                                  for j in range(2))
    assert chunk_by_size(ids, {i: 1 for i in ids}, 4, 100) == [
        list('abcd'), list('ef')]                       # count cap still applies


def test_extract_job_splits_and_reports_missing_days(monkeypatch):
    """A request with more than 2 reservoirs fails with Earth Engine's
    payload-size error; the job must split down and still return every
    reservoir. A day with no URMA image is reported, not fatal."""
    import importlib
    ex = importlib.import_module('resdlem_force.extract')
    from resdlem_force.config import ID_PROP
    labels = ['20200715', '20200716']
    monkeypatch.setattr(ex, '_block_info', lambda lab, precip, cfg: (
        {'urma': ['20200715'], 'prism800': labels, 'prism4k': labels,
         'urma_pcp': labels}, {'20200715': 'permanent'}))
    sizes_seen, succeeded = [], []

    def fake_chunk(cfg, info, status, met_geo, grid_geo, ids, precip=True):
        sizes_seen.append(len(ids))
        if len(ids) > 2:
            raise RuntimeError('EEException: Request payload size exceeds '
                               'the limit: 10485760 bytes.')
        succeeded.append(len(ids))
        df = pl.DataFrame({ID_PROP: ids, 'date': [info['urma'][0]] * len(ids)})
        return df, 0

    monkeypatch.setattr(ex, '_extract_chunk', fake_chunk)
    ids = [f'r{i}' for i in range(7)]
    job = {'cfg': RunConfig(project='x'), 'labels': labels, 'k': 0,
           'cells': [(i, {}) for i in ids], 'grid': {}, 'precip': True}
    status, name, k, df, n_no_pr, bad = ex._extract_job(job)
    assert status == 'ok' and name == '20200715-20200716'
    assert bad == [('* 20200716', 'no URMA image for this date')]
    assert sorted(df[ID_PROP].to_list()) == ids          # nothing lost
    assert sizes_seen[0] == 7                            # tried whole first
    assert max(succeeded) <= 2 and sum(succeeded) == 7   # split until it fit


# --- quota estimate / check (no Cloud Monitoring needed) --------------------
_STATUS = {'limit_eecu_h': 1000.0, 'used_eecu_h': 15.0,
           'remaining_eecu_h': 985.0, 'reads_per_minute_limit': 6000,
           'usage_as_of': 0}


def test_estimate_and_check_quota():
    from resdlem_force.quota import (QuotaExceededError, check_quota,
                                  estimate_eecu_hours)
    assert estimate_eecu_hours(36_000, 0.1) == pytest.approx(1.0)
    cfg = RunConfig(project='x', eecu_s_per_reservoir_day=0.15)
    # 44,749 reservoirs x 1 day ~ 1.9 EECU-h: fits in 985
    info = check_quota(44_749, cfg, status=_STATUS)
    assert info['estimated_eecu_h'] == pytest.approx(44_749 * 0.15 / 3600)
    # 44,749 x 17,000 days ~ 31,700 EECU-h: does not fit
    with pytest.raises(QuotaExceededError, match='stopped'):
        check_quota(44_749 * 17_000, cfg, status=_STATUS)
    # quota_fraction lowers the allowance
    half = RunConfig(project='x', eecu_s_per_reservoir_day=1.0,
                     quota_fraction=0.5)
    check_quota(int(0.49 * 985 * 3600), half, status=_STATUS)
    with pytest.raises(QuotaExceededError):
        check_quota(int(0.51 * 985 * 3600), half, status=_STATUS)


def test_quota_gate_prints_and_skips_when_unavailable(monkeypatch, capsys):
    import importlib
    ex = importlib.import_module('resdlem_force.extract')
    from resdlem_force.quota import QuotaExceededError, QuotaUnavailableError
    jobs = [{'cells': [('a', {}), ('b', {})], 'labels': ['20200715']},
            {'cells': [('c', {})], 'labels': ['20200715']}]
    cfg = RunConfig(project='x')

    def unavailable(rd, config):
        raise QuotaUnavailableError('no monitoring')
    monkeypatch.setattr(ex, 'check_quota', unavailable)
    ex._quota_gate(jobs, cfg)                         # must not raise
    assert 'unavailable' in capsys.readouterr().out

    def broken(rd, config):
        raise ValueError('something odd')
    monkeypatch.setattr(ex, 'check_quota', broken)
    ex._quota_gate(jobs, cfg)                         # any error -> skip
    assert 'unavailable' in capsys.readouterr().out

    def ok(rd, config):
        assert rd == 3                                # reservoir-days counted
        return {**_STATUS, 'estimated_eecu_h': 0.1, 'allowed_eecu_h': 985.0}
    monkeypatch.setattr(ex, 'check_quota', ok)
    ex._quota_gate(jobs, cfg)
    out = capsys.readouterr().out
    assert 'QUOTA: estimated 0.10 EECU-h' in out and '985.0 of 1,000' in out

    def too_big(rd, config):
        raise QuotaExceededError('Run stopped before any data request')
    monkeypatch.setattr(ex, 'check_quota', too_big)
    with pytest.raises(QuotaExceededError):           # stop on exceed
        ex._quota_gate(jobs, cfg)
    assert 'stopped' in capsys.readouterr().out

    ex._quota_gate(jobs, RunConfig(project='x', check_quota=False))  # off


def test_quota_status_without_monitoring_library(monkeypatch):
    import builtins
    from resdlem_force.quota import QuotaUnavailableError, quota_status
    real_import = builtins.__import__

    def no_monitoring(name, *a, **k):
        if name.startswith('google.cloud'):
            raise ImportError('not installed')
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', no_monitoring)
    with pytest.raises(QuotaUnavailableError, match='google-cloud-monitoring'):
        quota_status('nasa-revap')


def test_parse_profile_eecu():
    from resdlem_force.quota import parse_profile_eecu
    table = """ EECU·s PeakMem Count  Description
     10.981    101M    68  Algorithm Image.exp computing pixels
      9.523    162M    64  Loading assets: x
      0.280    338k  4796  (plumbing)
       -        50k     1  Loading assets: y"""
    assert parse_profile_eecu(table) == pytest.approx(20.784)

if __name__ == '__main__':
    # Running the file directly (e.g. Spyder) only defines the tests above;
    # hand it to pytest so they actually run. -p no:cacheprovider avoids
    # writing a .pytest_cache folder next to the source.
    raise SystemExit(pytest.main([__file__, '-v', '-p', 'no:cacheprovider']))


# --- end to end with Earth Engine replaced (no network) -----------------------
def _fake_ee_run(monkeypatch, missing_urma=()):
    ex = _stub_ee(monkeypatch)
    import resdlem_force.ee_utils as eu
    monkeypatch.setattr(eu, 'init_ee', lambda *a, **k: None)
    from resdlem_force.config import ID_PROP, URMA_OUTPUT

    def block_info(labels, precip, cfg):
        days = [d for d in labels if d not in missing_urma]
        return ({'urma': days, 'prism800': labels, 'prism4k': labels},
                {d: 'permanent' for d in days})

    def met(days, features, cfg):
        rows = {ID_PROP: [], 'date': []}
        for i, _ in features:
            for d in days:
                rows[ID_PROP].append(i)
                rows['date'].append(d)
        df = pl.DataFrame(rows)
        return df.with_columns([pl.lit(1.0).alias(c)
                                for c in URMA_OUTPUT.values()])

    def precip(coll_id, band, labels, features, cfg, grid_proj=None):
        return pl.DataFrame({ID_PROP: [i for i, _ in features
                                       for _ in labels],
                             'date': [d for _ in features for d in labels],
                             'pr': [2.0] * (len(features) * len(labels))})
    monkeypatch.setattr(ex, '_block_info', block_info)
    monkeypatch.setattr(ex, '_reduce_met', met)
    monkeypatch.setattr(ex, '_reduce_precip', precip)
    monkeypatch.setattr(ex, '_prism800_proj', lambda: None)
    return ex


def test_extract_urma_end_to_end_fake_ee(monkeypatch):
    """extract_urma returns a pandas DataFrame: output columns in order,
    numbers float64, text object, sorted by id and date."""
    import pandas as pd
    from resdlem_force.config import OUTPUT_COLS_AFTER_ID
    ex = _fake_ee_run(monkeypatch)
    gdf, _, _ = _two_geom_inputs()
    cfg = RunConfig(project='x', n_workers=1, check_quota=False,
                    days_per_request=2)
    df = ex.extract_urma(gdf, id_col='OWEN_ID', config=cfg,
                         start='2020-07-15', end='2020-07-17')
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['OWEN_ID'] + OUTPUT_COLS_AFTER_ID
    assert len(df) == 3 * 3                           # 3 reservoirs x 3 days
    assert set(df['pr_source']) == {'prism800'}
    assert df['pr'].tolist() == [2.0] * 9
    assert df['tmax'].dtype == 'float64'
    assert pd.api.types.is_string_dtype(df['date'])   # object (pandas 2)
    #                                                   or str (pandas 3)
    assert not df.isna().any().any()
    assert list(zip(df['OWEN_ID'], df['date'])) == sorted(
        zip(df['OWEN_ID'], df['date']))
    # without precip: no pr / pr_source, no PRISM cells needed
    met = ex.extract_urma(gdf.drop(columns='prism_cells'), id_col='OWEN_ID',
                          config=cfg, dates='2020-07-15', precip=False,
                          prism_cells_col=None)
    assert 'pr' not in met.columns and 'pr_source' not in met.columns
    met2 = ex.extract_urma(gdf.drop(columns='prism_cells'), id_col='OWEN_ID',
                           config=cfg, dates='2020-07-15', precip=False)
    assert list(met2.columns) == list(met.columns)


def test_extract_missing_urma_day_is_reported(monkeypatch):
    ex = _fake_ee_run(monkeypatch, missing_urma=('20200716',))
    gdf, _, _ = _two_geom_inputs()
    cfg = RunConfig(project='x', n_workers=1, check_quota=False)
    with pytest.raises(RuntimeError, match='no URMA image'):
        ex.extract_urma(gdf, id_col='OWEN_ID', config=cfg,
                        dates=['2020-07-15', '2020-07-16'])
    df = ex.extract_urma(gdf, id_col='OWEN_ID', config=cfg,
                         dates=['2020-07-15', '2020-07-16'],
                         allow_partial=True)
    assert set(df['date']) == {'20200715'}


# --- reference grids and cell-value download (Earth Engine replaced) -------
def test_grid_info_and_cell_value_tiles(monkeypatch):
    import resdlem_force.cells as cells_mod
    import resdlem_force.grids as grids
    grids._CACHE.clear()
    transform = URMA_TRANSFORM
    answers = iter([
        {'proj': {'type': 'Projection', 'wkt': URMA_WKT,
                  'transform': transform}, 'index': '20260101'},
        {'bands': [{'id': 'TMAX', 'dimensions': [7, 600]}]},
    ])

    def fake_get_info(obj, cfg):
        if isinstance(obj, list):            # the CRS check: Earth Engine's
            return [_sphere_xy(*ll) for ll in obj]   # answer = the sphere
        return next(answers)
    monkeypatch.setattr(grids, 'init_ee', lambda *a, **k: None)
    monkeypatch.setattr(grids, 'reference_image', lambda s: _FakeImg())
    monkeypatch.setattr(grids, 'get_info', fake_get_info)
    monkeypatch.setattr(grids.ee, 'Dictionary', lambda d: d)
    monkeypatch.setattr(grids.ee, 'List', lambda x: x)
    monkeypatch.setattr(grids.ee, 'Projection', lambda t: t)
    monkeypatch.setattr(grids.ee, 'Geometry', _FakeGeometry)
    cfg = RunConfig(project='x')
    info = grids.grid_info('urma', cfg)
    assert info['image'] == '20260101' and (info['width'], info['height']) \
        == (7, 600) and (info['col0'], info['row0']) == (0, 0)
    assert info['grid'].a == 2539.703
    assert info['grid'].ee_crs.startswith('PROJCS')     # EE's text kept
    assert grids.grid_info('urma', cfg) is info            # cached

    # computePixels replaced: elevation = 1000*row + col, water at col 3,
    # elevation masked at row 10
    reqs = []

    def compute_pixels(req):
        g = req['grid']
        h, w = g['dimensions']['height'], g['dimensions']['width']
        r0 = round((g['affineTransform']['translateY'] - transform[5])
                   / transform[4])
        rows = np.arange(r0, r0 + h)[:, None] + np.zeros((1, w))
        cols = np.zeros((h, 1)) + np.arange(w)[None, :]
        arr = np.zeros((h, w), dtype=[('elev', 'f4'), ('elev_ok', 'f4'),
                                      ('wm', 'f4'), ('wm_ok', 'f4')])
        arr['elev'] = 1000 * rows + cols
        arr['elev_ok'] = (rows != 10)
        arr['wm'] = (cols == 3)
        arr['wm_ok'] = 1
        reqs.append((r0, h, 'crsWkt' in g))
        return arr
    monkeypatch.setattr(cells_mod, 'init_ee', lambda *a, **k: None)
    monkeypatch.setattr(cells_mod, '_value_image', lambda: None)
    monkeypatch.setattr(cells_mod.ee.data, 'computePixels', compute_pixels)
    v = cells_mod.fetch_cell_values(cfg)
    assert [r[:2] for r in reqs] == [(0, 256), (256, 256), (512, 88)]
    assert all(r[2] for r in reqs)                       # WKT CRS sent
    e, w, ok = v.lookup([2, 3, 2, 9], [5, 300, 10, 1])
    assert e[0] == 5002 and e[1] == 300003
    assert np.isnan(e[2])                                # masked -> NaN
    assert w[1] == 1 and w[0] == 0
    assert not ok[3] and np.isnan(e[3])                  # outside the image
    grids._CACHE.clear()


class _FakeImg:
    def projection(self):
        return 'proj'

    def get(self, k):
        return k


def _sphere_xy(lon, lat):
    import pyproj
    return list(pyproj.Transformer.from_crs('EPSG:4326', URMA_SPHERE,
                                            always_xy=True).transform(lon, lat))


class _FakeGeometry:
    """ee.Geometry stand-in: .Point(ll).transform(...).coordinates() -> ll"""
    @staticmethod
    def Point(ll):
        class P:
            def transform(self, proj, err):
                return self

            def coordinates(self):
                return tuple(ll)
        return P()


def test_local_crs_uses_the_sphere_earth_engine_uses():
    """pyproj alone reads the URMA WKT as WGS 84 (up to 8 km off);
    local_crs() must give the 6,371,200 m sphere, which matched Earth
    Engine to 0.000 m (probe_urma_crs.py)."""
    import pyproj
    from resdlem_force.grid_cells import local_crs
    crs = local_crs(URMA_WKT)
    pts = [(-125, 25), (-117, 31), (-101, 37), (-85, 43), (-69, 49)]
    mine = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    ref = pyproj.Transformer.from_crs('EPSG:4326', URMA_SPHERE,
                                      always_xy=True)
    naive = pyproj.Transformer.from_crs('EPSG:4326', URMA_WKT, always_xy=True)
    for lo, la in pts:
        a, b, c = mine.transform(lo, la), ref.transform(lo, la), \
            naive.transform(lo, la)
        assert abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6
    # the naive reading really is far off (this is the bug being avoided)
    worst = max(np.hypot(*(np.subtract(naive.transform(lo, la),
                                       ref.transform(lo, la))))
                for lo, la in pts)
    assert worst > 5000                                # metres
    assert local_crs('EPSG:4269') == 'EPSG:4269'       # others unchanged


def test_grid_check_stops_on_a_crs_mismatch(monkeypatch):
    import resdlem_force.grids as grids
    from resdlem_force.grid_cells import Grid
    naive = Grid('+proj=lcc +lat_1=25 +lat_0=25 +lon_0=-95 +datum=WGS84 '
                 '+units=m', *URMA_TRANSFORM[::2][:1], URMA_TRANSFORM[2],
                 URMA_TRANSFORM[4], URMA_TRANSFORM[5], URMA_WKT)
    monkeypatch.setattr(grids, 'get_info',
                        lambda obj, cfg: [_sphere_xy(*ll) for ll in obj])
    monkeypatch.setattr(grids.ee, 'List', lambda x: x)
    monkeypatch.setattr(grids.ee, 'Projection', lambda t: t)
    monkeypatch.setattr(grids.ee, 'Geometry', _FakeGeometry)
    with pytest.raises(RuntimeError, match='does not match Earth Engine'):
        grids._verify('urma', naive, [2345, 1597], [0, 0],
                      RunConfig(project='x'))
    good = Grid.from_projection({'wkt': URMA_WKT,
                                 'transform': URMA_TRANSFORM})
    grids._verify('urma', good, [2345, 1597], [0, 0], RunConfig(project='x'))


# --- figures (no Earth Engine, no imagery) ------------------------------------
def test_plot_cells(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    gpd = pytest.importorskip('geopandas')
    import pyproj
    import resdlem_force
    from resdlem_force import figures
    from resdlem_force.cells import CellValues
    ug, pg = _urma_grid(), _prism_grid()
    monkeypatch.setattr(figures.grids, 'grid_from_ee',
                        lambda s, cfg: pg if s == 'prism800' else ug)
    x, y = pyproj.Transformer.from_crs(4326, 5070, always_xy=True).transform(
        -119.06, 39.45)
    lake = shapely.box(x, y, x + 6000, y + 3000)
    gdf = gpd.GeoDataFrame({'OWEN_ID': ['A', 'B'], 'NAME': ['Lake A', 'B']},
                           geometry=[lake, shapely.affinity.translate(
                               lake, 20000, 0)], crs='EPSG:5070')
    # values around the lake: all water -> water_mask method
    ux, uy = pyproj.Transformer.from_crs(4326, ug.crs, always_xy=True
                                         ).transform(-119.06, 39.45)
    c0, r0 = ug.col_row(ux, uy)
    values = CellValues(ug, c0 - 20, r0 - 20, np.zeros((40, 40)),
                        np.ones((40, 40)))
    out = resdlem_force.plot_cells(gdf, id_col='OWEN_ID', ids=['A'],
                                title_col='NAME', basemap=None,
                                config=RunConfig(project='x'), values=values,
                                out_dir=tmp_path, formats=('png', 'pdf'),
                                return_figures=False)
    assert list(out) == ['A']
    assert [p.name for p in out['A']] == ['A_cells.png', 'A_cells.pdf']
    assert all(p.stat().st_size > 1000 for p in out['A'])
    figs = resdlem_force.plot_cells(gdf, id_col='OWEN_ID', ids=['A'],
                                 basemap=None, config=RunConfig(project='x'),
                                 values=values, return_figures=True)
    labels = [t.get_text() for t in figs['A'].legends[0].get_texts()]
    assert labels == ['Reservoir', 'URMA 2.5 km cells', 'PRISM 800 m cells']
    texts = [t.get_text() for t in figs['A'].axes[0].texts]
    assert 'URMA water mask: yes' in texts
    # no water anywhere -> overlap/elevation cells, drawn dashed
    dry = CellValues(ug, c0 - 20, r0 - 20, np.zeros((40, 40)),
                     np.zeros((40, 40)))
    figs = resdlem_force.plot_cells(gdf, id_col='OWEN_ID', ids=['A'],
                                 basemap=None, config=RunConfig(project='x'),
                                 values=dry, return_figures=True)
    texts = [t.get_text() for t in figs['A'].axes[0].texts]
    assert 'URMA water mask: no' in texts
    import matplotlib.pyplot as plt
    plt.close('all')
    with pytest.raises(ValueError):
        resdlem_force.plot_cells(gdf, id_col='OWEN_ID', basemap='osm',
                              config=RunConfig(project='x'), values=values)


def test_repair_log_names_the_geometry(caplog):
    """The repair log says which geometries were repaired; repaired cell
    polygons (always built valid) are a warning."""
    import logging
    bow = shapely.Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])     # bow-tie
    df = pl.DataFrame({'id': ['x'], 'geometry': [shapely.to_wkb(bow)]})
    with caplog.at_level(logging.DEBUG, logger='resdlem_force'):
        gs = load_geometries(df, 'id', 'geometry', 'EPSG:5070',
                             label='native reservoir')
        load_geometries(df, 'id', 'geometry', 'EPSG:5070',
                        label='URMA cell', expect_valid=True)
    assert gs.repaired_ids == ['x'] and gs.geoms[0].is_valid
    recs = [(r.levelname, r.getMessage()) for r in caplog.records
            if 'repaired' in r.getMessage()]
    assert recs[0][0] == 'DEBUG' and 'invalid native reservoir' in recs[0][1]
    assert recs[1][0] == 'WARNING' and 'invalid URMA cell' in recs[1][1]


def test_built_cell_polygons_are_valid():
    """Scattered cells (diagonal neighbours, holes) on both grids give
    valid polygons, also after reprojection to EPSG:4326 / 5070."""
    from resdlem_force import grid_cells as gc
    rng = np.random.default_rng(0)
    for grid, (c0, r0) in ((_urma_grid(), (1500, 800)),
                           (_prism_grid(), (3000, 1500))):
        for _ in range(300):
            cells = list({(c0 + int(rng.integers(0, 7)),
                           r0 + int(rng.integers(0, 7)))
                          for _ in range(int(rng.integers(1, 30)))})
            g = gc.cell_geometry(cells, grid)
            assert g.is_valid
            for crs in ('EPSG:4326', 'EPSG:5070'):
                assert reproject(np.array([g], dtype=object), grid.crs,
                                 crs)[0].is_valid


def test_cell_geometry_falls_back_when_coverage_union_invalid(monkeypatch):
    """Some GEOS versions give an invalid ring from coverage_union_all for
    cells touching only at a corner around a gap (seen on a real reservoir,
    PRISM cells below); the squares are then joined with union_all."""
    from resdlem_force import grid_cells as gc
    grid = _prism_grid()
    cells = [(3689, 357), (3690, 357), (3686, 358), (3687, 358), (3688, 358),
             (3689, 358), (3690, 358), (3691, 358), (3692, 358), (3686, 359),
             (3687, 359), (3688, 359), (3689, 359), (3690, 359), (3691, 359),
             (3687, 360), (3688, 360), (3685, 359), (3691, 357), (3689, 354),
             (3691, 360), (3686, 360), (3693, 354), (3690, 355), (3685, 358),
             (3687, 356), (3686, 357), (3692, 359), (3683, 359), (3695, 354),
             (3687, 357), (3684, 359), (3683, 360), (3691, 355), (3690, 356),
             (3688, 357), (3692, 355), (3689, 360), (3696, 354), (3689, 356),
             (3690, 354), (3692, 360), (3687, 361), (3688, 354), (3694, 354),
             (3687, 355)]
    ref = gc.cell_geometry(cells, grid)
    assert ref.is_valid
    bow = shapely.Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])    # invalid
    monkeypatch.setattr(gc.shapely, 'coverage_union_all', lambda sq: bow)
    g = gc.cell_geometry(cells, grid)
    assert g.is_valid and g.equals(ref)
    assert g.area == pytest.approx(len(cells) * grid.pixel_area)



def test_cell_geometry_fallback_urma_case(monkeypatch):
    """Same fallback on the URMA grid, with the cells of a real reservoir
    that failed on an older GEOS (258 cells, corner contacts)."""
    from resdlem_force import grid_cells as gc
    URMA_CASE = [(1104, 517), (1105, 518), (1106, 519), (1107, 519),
                 (1107, 520), (1107, 522), (1107, 523), (1106, 526),
                 (1106, 527), (1107, 528), (1107, 529), (1107, 530),
                 (1107, 531), (1107, 532), (1107, 533), (1106, 534),
                 (1107, 536), (1106, 537), (1105, 538), (1104, 539),
                 (1105, 539), (1104, 540), (1105, 540), (1104, 541),
                 (1105, 541), (1104, 542), (1105, 542), (1105, 543),
                 (1105, 544), (1105, 545), (1105, 546), (1106, 546),
                 (1106, 547), (1107, 547), (1107, 548), (1108, 548),
                 (1107, 549), (1108, 549), (1108, 550), (1109, 550),
                 (1110, 550), (1109, 551), (1110, 551), (1115, 551),
                 (1110, 552), (1111, 552), (1111, 553), (1112, 553),
                 (1113, 553), (1112, 554), (1112, 555), (1112, 556),
                 (1112, 557), (1113, 558), (1113, 559), (1114, 559),
                 (1114, 560), (1114, 561), (1114, 562), (1104, 563),
                 (1111, 563), (1112, 563), (1113, 563), (1105, 564),
                 (1106, 564), (1110, 564), (1111, 564), (1112, 564),
                 (1105, 565), (1106, 565), (1108, 565), (1109, 565),
                 (1110, 565), (1106, 566), (1107, 566), (1109, 566),
                 (1110, 566), (1107, 567), (1109, 567), (1108, 568),
                 (1108, 569), (1108, 570), (1109, 570), (1110, 570),
                 (1111, 570), (1109, 571), (1111, 571), (1112, 571),
                 (1112, 572), (1113, 572), (1112, 573), (1113, 573),
                 (1113, 574), (1114, 574), (1114, 575), (1114, 576),
                 (1106, 577), (1107, 577), (1108, 577), (1109, 577),
                 (1111, 577), (1114, 577), (1115, 577), (1112, 578),
                 (1113, 578), (1114, 578), (1112, 579), (1113, 579),
                 (1114, 579), (1112, 580), (1113, 580), (1114, 580),
                 (1114, 581), (1114, 582), (1115, 583), (1114, 584),
                 (1114, 585), (1114, 586), (1114, 587), (1113, 588),
                 (1114, 588), (1112, 589), (1113, 589), (1112, 590),
                 (1113, 590), (1112, 591), (1113, 592), (1109, 593),
                 (1110, 593), (1111, 593), (1112, 593), (1114, 593),
                 (1109, 594), (1108, 595), (1109, 595), (1108, 596),
                 (1109, 596), (1108, 597), (1109, 597), (1109, 598),
                 (1110, 598), (1109, 599), (1110, 599), (1098, 600),
                 (1099, 600), (1109, 600), (1110, 600), (1098, 601),
                 (1099, 601), (1100, 601), (1101, 601), (1102, 601),
                 (1105, 601), (1108, 601), (1109, 601), (1092, 602),
                 (1099, 602), (1100, 602), (1101, 602), (1102, 602),
                 (1105, 602), (1106, 602), (1107, 602), (1108, 602),
                 (1092, 603), (1093, 603), (1094, 603), (1099, 603),
                 (1101, 603), (1102, 603), (1103, 603), (1106, 603),
                 (1107, 603), (1090, 604), (1091, 604), (1092, 604),
                 (1093, 604), (1094, 604), (1095, 604), (1096, 604),
                 (1097, 604), (1098, 604), (1099, 604), (1102, 604),
                 (1103, 604), (1105, 604), (1106, 604), (1107, 604),
                 (1084, 605), (1085, 605), (1086, 605), (1087, 605),
                 (1088, 605), (1089, 605), (1090, 605), (1098, 605),
                 (1099, 605), (1100, 605), (1101, 605), (1103, 605),
                 (1104, 605), (1105, 605), (1087, 606), (1088, 606),
                 (1089, 606), (1090, 606), (1100, 606), (1101, 606),
                 (1102, 606), (1104, 606), (1090, 607), (1100, 607),
                 (1101, 607), (1102, 607), (1101, 608), (1102, 608),
                 (1100, 609), (1101, 609), (1102, 609), (1101, 610),
                 (1102, 610), (1101, 611), (1100, 612), (1101, 612),
                 (1100, 613), (1101, 613), (1102, 613), (1105, 613),
                 (1106, 613), (1101, 614), (1102, 614), (1103, 614),
                 (1104, 614), (1105, 614), (1106, 614), (1101, 615),
                 (1102, 615), (1103, 615), (1104, 615), (1105, 615),
                 (1104, 616), (1105, 616), (1103, 617), (1104, 617),
                 (1102, 618), (1103, 618), (1104, 618), (1106, 618),
                 (1103, 619), (1104, 619), (1105, 619), (1106, 619),
                 (1107, 619), (1108, 619), (1103, 620), (1104, 620),
                 (1105, 620), (1106, 620)]
    grid = _urma_grid()
    bow = shapely.Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])    # invalid
    monkeypatch.setattr(gc.shapely, 'coverage_union_all', lambda sq: bow)
    g = gc.cell_geometry(URMA_CASE, grid)
    assert g.is_valid
    assert g.area == pytest.approx(len(URMA_CASE) * grid.pixel_area)


def test_extract_prepare_database_style_geodataframe(monkeypatch):
    """A GeoDataFrame as a database read gives it: urma_cells is the
    GeoSeries, prism_cells a plain column of hex WKB (PostGIS), hex EWKB
    with an SRID, or shapely objects -- all give the same jobs as two
    GeoSeries."""
    ex = _stub_ee(monkeypatch)
    gdf, _, _ = _two_geom_inputs()
    kw = dict(id_col='OWEN_ID', urma_col='urma_cells',
              prism_col='prism_cells', dates='2020-07-15', start=None, end=None,
              urma_crs=None, config=RunConfig(project='x'))
    ref = ex._prepare(gdf, prism_crs=None, **kw)
    want = {i: g for j in ref.jobs for i, g in j['grid']['prism800']}
    geoms = list(gdf['prism_cells'])
    variants = {
        'hex wkb': [None if g is None else shapely.to_wkb(g, hex=True)
                    for g in geoms],
        'hex ewkb': [None if g is None else shapely.to_wkb(
            shapely.set_srid(g, 4269), hex=True, include_srid=True)
            for g in geoms],
        'shapely': geoms,
    }
    for name, vals in variants.items():
        db = gdf.drop(columns='prism_cells').copy()
        db['prism_cells'] = pd_series(vals, db.index)
        plan = ex._prepare(db, prism_crs='EPSG:4269', **kw)
        got = {i: g for j in plan.jobs for i, g in j['grid']['prism800']}
        assert got.keys() == want.keys(), name
        for i in want:
            assert shapely.geometry.shape(got[i]).equals_exact(
                shapely.geometry.shape(want[i]), 1e-9), name
    db = gdf.drop(columns='prism_cells').copy()
    db['prism_cells'] = pd_series(variants['hex wkb'], db.index)
    with pytest.raises(ValueError, match='prism_crs'):     # CRS needed
        ex._prepare(db, prism_crs=None, **kw)


def pd_series(values, index):
    import pandas as pd
    return pd.Series(values, index=index, dtype=object)


def test_select_urma_cells_geodataframe(tmp_path, monkeypatch):
    """A GeoDataFrame (as read from the GeoPackage) gives the same output as
    the file path."""
    gpd = pytest.importorskip('geopandas')
    import resdlem_force
    import resdlem_force.files as files
    monkeypatch.setattr(files, 'select_cells_frame', _fake_select_cells({}))
    src, path = _reservoir_file(tmp_path, 'gpkg')
    cfg = RunConfig(project='x')
    from_file = resdlem_force.select_urma_cells(
        path, id_col='OWEN_ID', config=cfg, layer='reservoirs')
    gdf = gpd.read_file(path, layer='reservoirs')
    from_gdf = resdlem_force.select_urma_cells(gdf, id_col='OWEN_ID',
                                              config=cfg)
    assert list(from_gdf.columns) == list(from_file.columns)
    assert from_gdf.drop(columns='geometry').equals(
        from_file.drop(columns='geometry'))
    assert all(a is None and b is None or a.equals(b) for a, b in
               zip(from_gdf.geometry, from_file.geometry))
    assert from_gdf.crs.equals(gdf.crs)


def test_selection_info_log_is_three_lines(tmp_path, monkeypatch, caplog):
    """select_urma_cells / select_prism_cells log only started, processing
    N geometries, finished at INFO (warnings only when something is wrong)."""
    import importlib
    import logging
    gpd = pytest.importorskip('geopandas')
    import resdlem_force
    import resdlem_force.files as files
    pc = importlib.import_module('resdlem_force.prism_cells')
    monkeypatch.setattr(files, 'select_cells_frame', _fake_select_cells({}))
    monkeypatch.setattr(pc, 'grid_from_ee', lambda s, cfg: _prism_grid())
    src, _ = _reservoir_file(tmp_path, 'gpkg')
    cfg = RunConfig(project='x')
    for fn, name in [(resdlem_force.select_urma_cells, 'select_urma_cells'),
                     (resdlem_force.select_prism_cells, 'select_prism_cells')]:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger='resdlem_force'):
            fn(src, id_col='OWEN_ID', config=cfg)
        info = [r.getMessage() for r in caplog.records
                if r.levelno == logging.INFO]
        assert info == [f'{name}: started',
                        f'{name}: processing 3 geometries',
                        f'{name}: finished'], info
