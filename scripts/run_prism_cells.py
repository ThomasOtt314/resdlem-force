import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import logging
import geopandas as gpd
import resdlem_force

GEOMETRY_FILE = '/Users/Thomas.Ott@dri.edu/Documents/delete/owen_master_v1.gpkg'
ID_COL = 'OWEN_ID'
OUT_DIR = Path('/Users/Thomas.Ott@dri.edu/Documents/delete')
OUTPUT_NAME = 'owen_master_v1_prism_cells'
PROJECT_ID = 'nasa-revap'

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('resdlem_force').setLevel(logging.INFO)
    gdf = gpd.read_file(GEOMETRY_FILE)
    cells = resdlem_force.select_prism_cells(
        gdf, id_col=ID_COL, config=resdlem_force.RunConfig(project=PROJECT_ID))

    gpkg = OUT_DIR / f'{OUTPUT_NAME}.gpkg'
    gpkg.unlink(missing_ok=True)
    cells.to_file(gpkg, layer=OUTPUT_NAME, driver='GPKG')
