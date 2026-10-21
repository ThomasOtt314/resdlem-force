"""Earth Engine quota: read what is left, estimate what a run will use.

Nothing here tracks individual requests. Before a run, the remaining
monthly compute quota is read from Cloud Monitoring and compared with an
estimate:

    estimated EECU-s = reservoir-days x RunConfig.eecu_s_per_reservoir_day

If the estimate exceeds the remaining quota (times quota_fraction) the run
stops before any data request is made. Usage from earlier runs is already
included in the remaining quota Google reports, so each new run checks
against up-to-date numbers (Cloud Monitoring reports with some delay).

Quota numbers come from the consumer-quota metrics found earlier
for nasa-revap:
  limit : serviceruntime.googleapis.com/quota/limit,
          limit_name = EarthEngineComputeTimePerMonthPerProject
  usage : serviceruntime.googleapis.com/quota/allocation/usage,
          quota_metric = earthengine.googleapis.com/monthly_eecu_usage_time
Values are read as EECU-seconds (limit 3,600,000 = 1,000 EECU-hours). The
unit is inferred from the metric name; the API does not state it.

Needs the optional dependency google-cloud-monitoring and Application
Default Credentials (gcloud auth application-default login).
"""

from __future__ import annotations

import logging
import re
import time

log = logging.getLogger('redlem_force')

_LIMIT_METRIC = 'serviceruntime.googleapis.com/quota/limit'
_USAGE_METRIC = 'serviceruntime.googleapis.com/quota/allocation/usage'
_MONTHLY_LIMIT_NAME = 'EarthEngineComputeTimePerMonthPerProject'
_MONTHLY_QUOTA_METRIC = 'earthengine.googleapis.com/monthly_eecu_usage_time'
_READS_LIMIT_NAME = 'ReadsPerMinutePerProject'


class QuotaExceededError(RuntimeError):
    """The estimated cost of a run is larger than the quota left."""


class QuotaUnavailableError(RuntimeError):
    """The remaining quota could not be read."""


def _latest(series):
    """Newest numeric value across time series (points are newest-first)."""
    best_t, best_v = None, None
    for ts in series:
        for p in ts.points:
            v = p.value
            val = None
            for attr in ('int64_value', 'double_value'):
                try:
                    if v._pb.HasField(attr):
                        val = getattr(v, attr)
                except Exception:
                    pass
            if val is None:
                continue
            t = p.interval.end_time
            t = t.timestamp() if hasattr(t, 'timestamp') else t.seconds
            if best_t is None or t > best_t:
                best_t, best_v = t, val
            break                       # newest point of this series
    return best_v, best_t


def quota_status(project: str, lookback_hours: int = 48) -> dict:
    """Monthly Earth Engine compute quota of a project.

    Returns {'limit_eecu_h', 'used_eecu_h', 'remaining_eecu_h',
    'reads_per_minute_limit', 'usage_as_of'} (usage_as_of = UNIX time of the
    usage value). Raises QuotaUnavailableError if it cannot be read.
    """
    try:
        from google.cloud import monitoring_v3
    except ImportError as e:
        raise QuotaUnavailableError(
            'google-cloud-monitoring is not installed: '
            'pip install google-cloud-monitoring') from e
    try:
        client = monitoring_v3.MetricServiceClient()
        name = f'projects/{project}'
        now = int(time.time())
        interval = monitoring_v3.TimeInterval({
            'end_time': {'seconds': now},
            'start_time': {'seconds': now - lookback_hours * 3600}})
        view = monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL

        def series(flt):
            return list(client.list_time_series(request={
                'name': name, 'filter': flt, 'interval': interval,
                'view': view}))

        svc = 'resource.labels.service = "earthengine.googleapis.com"'
        limit, _ = _latest(series(
            f'metric.type = "{_LIMIT_METRIC}" AND {svc} AND '
            f'metric.labels.limit_name = "{_MONTHLY_LIMIT_NAME}"'))
        used, used_t = _latest(series(
            f'metric.type = "{_USAGE_METRIC}" AND {svc} AND '
            f'metric.labels.quota_metric = "{_MONTHLY_QUOTA_METRIC}"'))
        reads, _ = _latest(series(
            f'metric.type = "{_LIMIT_METRIC}" AND {svc} AND '
            f'metric.labels.limit_name = "{_READS_LIMIT_NAME}"'))
    except Exception as e:
        raise QuotaUnavailableError(
            f'could not read quota from Cloud Monitoring: '
            f'{type(e).__name__}: {e}. Check Application Default '
            'Credentials (gcloud auth application-default login).') from e

    if limit is None:
        raise QuotaUnavailableError(
            f'no {_MONTHLY_LIMIT_NAME} value found for project {project}')
    if used is None:
        # No usage reported in the lookback window: treat as nothing used.
        log.warning('no monthly EECU usage reported in the last %d h; '
                    'treating usage as 0', lookback_hours)
        used = 0
    return {
        'limit_eecu_h': limit / 3600,
        'used_eecu_h': used / 3600,
        'remaining_eecu_h': max(limit - used, 0) / 3600,
        'reads_per_minute_limit': reads,
        'usage_as_of': used_t,
    }


def estimate_eecu_hours(reservoir_days: int,
                        eecu_s_per_reservoir_day: float) -> float:
    """Estimated compute of a run, in EECU-hours."""
    return reservoir_days * eecu_s_per_reservoir_day / 3600


def check_quota(reservoir_days: int, config, status: dict | None = None
                ) -> dict:
    """Stop (QuotaExceededError) if the estimated run cost is larger than
    config.quota_fraction x the remaining monthly quota. Returns the numbers
    used for the decision."""
    est = estimate_eecu_hours(reservoir_days, config.eecu_s_per_reservoir_day)
    st = status if status is not None else quota_status(config.project)
    allowed = st['remaining_eecu_h'] * config.quota_fraction
    info = {**st, 'reservoir_days': reservoir_days,
            'estimated_eecu_h': est, 'allowed_eecu_h': allowed}
    log.info('quota: estimated %.2f EECU-h for %d reservoir-days '
             '(%.4f EECU-s each); remaining %.2f of %.2f EECU-h this month',
             est, reservoir_days, config.eecu_s_per_reservoir_day,
             st['remaining_eecu_h'], st['limit_eecu_h'])
    if est > allowed:
        raise QuotaExceededError(
            f'Run stopped before any data request: estimated '
            f'{est:,.1f} EECU-h for {reservoir_days:,} reservoir-days, but '
            f'only {st["remaining_eecu_h"]:,.1f} EECU-h of the monthly '
            f'{st["limit_eecu_h"]:,.0f} remain'
            + (f' (x quota_fraction {config.quota_fraction})'
               if config.quota_fraction != 1 else '')
            + '. Run fewer dates or reservoirs, or wait for the monthly reset.')
    return info


def parse_profile_eecu(profile_text: str) -> float:
    """Total EECU-s from an ee.profilePrinting table (sum of the EECU-s
    column; rows showing '-' had no compute). Used for calibration only."""
    total = 0.0
    for line in profile_text.splitlines():
        m = re.match(r'\s*([0-9]*\.?[0-9]+)\s+\S+\s+\d+\s', line)
        if m:
            total += float(m.group(1))
    return total
