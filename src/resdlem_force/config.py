"""Dataset IDs, band/column maps, and the user-facing run configuration."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Earth Engine assets
# ---------------------------------------------------------------------------
URMA_ID = 'projects/climate-engine-pro/assets/noaa-urma/daily'
PRISM_800_ID = 'projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_DAILY'
PRISM_4K_ID = 'OREGONSTATE/PRISM/ANd'

# Per-cell values for URMA cell selection, read on the URMA grid.
# Elevation: URMA ancillary asset, band 'elevation', m; the same data as
# the earlier rtma_elev asset (same lattice as the URMA grid).
# Water mask: same lattice as the URMA grid, 1 = water (checked 2026-10-21).
ELEV_ID = 'projects/climate-engine-pro/assets/noaa-urma/ancillary/elevation'
WM_ID = 'projects/climate-engine-pro/assets/noaa-urma/ancillary/water_mask'

HIGH_VOLUME_URL = 'https://earthengine-highvolume.googleapis.com'

# The ONLY source of grid information (resdlem_force.grids): the reference
# image of each source as (collection, band, year of the first image by
# system:index, or None = first image of the collection).
GRID_REFERENCE = {
    'urma': (URMA_ID, 'TMAX', 2026),
    'prism800': (PRISM_800_ID, 'ppt', None),
}

# Property name used for the reservoir ID inside Earth Engine requests.
# The user's own ID column name is restored on output.
ID_PROP = 'dlem_id'

# ---------------------------------------------------------------------------
# URMA met variables (same bands, reducers and output names as
# urma_download_v2.py)
# ---------------------------------------------------------------------------
MEAN_BANDS = ['PRES', 'TMIN', 'TMAX', 'vs2m', 'SRAD_TCDC', 'vpd', 'SPFH',
              'ETo', 'ETr']
MEDIAN_BANDS = ['WDIR']

# reduceRegions output name -> output column name. Output names follow the
# DLEM variable config (key = output name, band_name = source band):
#   tmax TMAX degC   tmin TMIN degC   vpd vpd kpa (derived)   vs2m m/s
#   (derived from WIND)   srad SRAD_TCDC wm2   etr ETr mm/day   eto ETo
#   mm/day   th WDIR deg (median)   sph SPFH kg/kg   pres PRES pa
# pr (mm) is the precip column: PRISM 800 m ppt, falling back to PRISM 4 km
# ppt, then URMA PCP (pr_source says which).
URMA_OUTPUT = {
    'TMAX_mean': 'tmax',
    'TMIN_mean': 'tmin',
    'vpd_mean': 'vpd',
    'vs2m_mean': 'vs2m',
    'SRAD_TCDC_mean': 'srad',
    'ETr_mean': 'etr',
    'ETo_mean': 'eto',
    'WDIR_median': 'th',
    'SPFH_mean': 'sph',
    'PRES_mean': 'pres',
}

# ---------------------------------------------------------------------------
# Precip fallback chain, in priority order: (name, collection, band).
# All three are matched by date label (system:index == YYYYMMDD), never by
# filterDate: the 4 km collection stamps images at 12:00 UTC on the day
# BEFORE their label. Verified: 4 km vs 800 m corr 0.998 when matched by
# label, ~0.17 when off by a day.
# ---------------------------------------------------------------------------
PRECIP_SOURCES = (
    ('prism800', PRISM_800_ID, 'ppt'),
    ('prism4k', PRISM_4K_ID, 'ppt'),
    ('urma', URMA_ID, 'PCP'),
)

# Output columns, in order. The ID column takes the user's name.
FLOAT_COLS = ['pr'] + list(URMA_OUTPUT.values())
OUTPUT_COLS_AFTER_ID = (['date'] + FLOAT_COLS
                        + ['pr_source', 'urma_status', 'processed_date'])


@dataclass
class RunConfig:
    """Settings for one run.

    project              : Google Cloud project used for Earth Engine.
    n_workers            : parallel worker processes. 1 = run in this
                           process (useful for debugging / Spyder).
    base_url             : Earth Engine endpoint. Default is the high-volume
                           endpoint used by urma_download_v2.py. None = the
                           standard endpoint.
    tile_scale           : reduceRegions tileScale. Raise to 2 or 4 only on
                           an Earth Engine memory error.
    features_per_request : max reservoirs per extraction request.
    days_per_request     : consecutive dates stacked into one request (all
                           days of a block as bands of one image). 5 was
                           benchmarked (stacking 5 days: ~1.35x faster for
                           URMA met, ~2.7x for precip); larger blocks are
                           NOT yet benchmarked.
    max_request_bytes    : max geometry bytes (GeoJSON) per extraction
                           request. Earth Engine rejects requests over
                           10,485,760 bytes ("Request payload size exceeds the
                           limit", seen with 1000 detailed reservoirs); the
                           4 MB default leaves room for request overhead.
    max_retries          : attempts per Earth Engine call / download.
    retry_wait_secs      : base wait for exponential backoff.
    download_timeout_secs: HTTP timeout for CSV downloads.
    check_quota          : before extracting, estimate the run's compute
                           and stop if it exceeds the project's remaining
                           monthly Earth Engine quota (needs
                           google-cloud-monitoring).
    eecu_s_per_reservoir_day : cost estimate used by that check. Default
                           0.012 = measured ~0.0102 (2026-10-15, URMA met +
                           PRISM precip, 5-day blocks), rounded up.
    quota_fraction       : stop if the estimate exceeds this fraction of the
                           remaining quota (1.0 = all of it).
    """
    project: str
    n_workers: int = 8
    base_url: str | None = HIGH_VOLUME_URL
    tile_scale: float = 1
    features_per_request: int = 1000
    days_per_request: int = 5
    max_request_bytes: int = 4_000_000
    max_retries: int = 5
    retry_wait_secs: float = 10.0
    download_timeout_secs: float = 600.0
    check_quota: bool = True
    eecu_s_per_reservoir_day: float = 0.012
    quota_fraction: float = 1.0

    def __post_init__(self):
        if not self.project:
            raise ValueError('project is required')
        for name in ('n_workers', 'features_per_request', 'days_per_request',
                     'max_retries',
                     'max_request_bytes'):
            if int(getattr(self, name)) < 1:
                raise ValueError(f'{name} must be >= 1')
        if self.eecu_s_per_reservoir_day <= 0:
            raise ValueError('eecu_s_per_reservoir_day must be > 0')
        if not 0 < self.quota_fraction <= 1:
            raise ValueError('quota_fraction must be in (0, 1]')
