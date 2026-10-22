import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import logging
import geopandas as gpd
import resdlem_force

URMA_CELLS_FILE = '/Users/Thomas.Ott@dri.edu/Documents/delete/owen_master_v1_urma_cells.gpkg'
PRISM_CELLS_FILE = '/Users/Thomas.Ott@dri.edu/Documents/delete/owen_master_v1_prism_cells.gpkg'
ID_COL = 'OWEN_ID'
START, END = '2020-07-01', '2020-07-31'
PROJECT_ID = 'nasa-revap'
N_WORKERS = 20
PRECIP = True

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('resdlem_force').setLevel(logging.INFO)
    urma = gpd.read_file(URMA_CELLS_FILE)[[ID_COL, 'geometry']]
    prism = gpd.read_file(PRISM_CELLS_FILE).set_index(ID_COL).geometry
    table = urma.rename_geometry('urma_cells')
    table['prism_cells'] = gpd.GeoSeries(
        prism.reindex(table[ID_COL]).values, index=table.index, crs=prism.crs)

    cfg = resdlem_force.RunConfig(project=PROJECT_ID, n_workers=N_WORKERS)
    df = resdlem_force.extract_urma(table, id_col=ID_COL, start=START, end=END,
                                   config=cfg, precip=PRECIP)
    print(df)
