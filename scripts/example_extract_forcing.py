import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import logging
import geopandas as gpd
import redlem_force

URMA_CELLS_FILE = '/Users/Thomas.Ott@dri.edu/Documents/delete/owen_master_v1_urma_cells.gpkg'
PRISM_CELLS_FILE = '/Users/Thomas.Ott@dri.edu/Documents/delete/owen_master_v1_prism_cells.gpkg'
ID_COL = 'OWEN_ID'
START, END = '2020-07-01', '2020-07-31'
PROJECT_ID = 'nasa-revap'
N_WORKERS = 20

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    urma = gpd.read_file(URMA_CELLS_FILE)[[ID_COL, 'geometry']]
    prism = gpd.read_file(PRISM_CELLS_FILE).set_index(ID_COL).geometry
    table = urma.rename_geometry('urma_cells')
    table['prism_cells'] = gpd.GeoSeries(
        prism.reindex(table[ID_COL]).values, index=table.index, crs=prism.crs)

    cfg = redlem_force.RunConfig(project=PROJECT_ID, n_workers=N_WORKERS,
                                 eecu_s_per_reservoir_day=0.012)
    df = redlem_force.extract_urma(table, id_col=ID_COL, start=START, end=END,
                                   config=cfg)
    print(df)
