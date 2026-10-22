import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import logging
import resdlem_force

GEOMETRY_FILE = '/Users/Thomas.Ott@dri.edu/Documents/delete/owen_master_v1.gpkg'
ID_COL = 'OWEN_ID'
IDS = ['BORGIS_146', 'BORGIS_64']
OUT_DIR = '/Users/Thomas.Ott@dri.edu/Documents/delete/cell_maps'
PROJECT_ID = 'nasa-revap'

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('resdlem_force').setLevel(logging.INFO)
    resdlem_force.plot_cells(
        GEOMETRY_FILE, id_col=ID_COL, ids=IDS, out_dir=OUT_DIR,
        config=resdlem_force.RunConfig(project=PROJECT_ID))
