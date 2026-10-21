"""Earth Engine session, retries, downloads and the worker pool."""

from __future__ import annotations

import io
import logging
import multiprocessing
import random
import time

import ee
import polars as pl
import requests

from .config import ID_PROP

log = logging.getLogger('redlem_force')

# Substrings that mark an error worth retrying as-is (quota, transient server
# or network trouble). Network timeouts are caught by type below.
_RETRYABLE = ('Too Many Requests', '429', 'Quota exceeded',
              'Service Unavailable', '503', '502', 'Internal error',
              'Connection reset', 'Connection aborted')

# Substrings that mean the REQUEST IS TOO BIG. Retrying the same request
# will fail the same way; the remedy is a smaller request.
_TOO_BIG = ('Computation timed out', 'User memory limit exceeded',
            'Too many concurrent aggregations', 'payload size',
            'Request payload')


def is_too_big(exc: Exception) -> bool:
    msg = str(exc)
    return any(tok in msg for tok in _TOO_BIG)


def solve_split(fn, items: list):
    """Call fn(items). If it fails because the request is too big, split
    items in half and recurse, down to single items.

    Returns (results, failures): results is the concatenation of fn's list
    results; failures is [(item, message)] for items that still failed.
    """
    try:
        return fn(items), []
    except Exception as e:
        msg = f'{type(e).__name__}: {e}'
        if len(items) > 1 and is_too_big(e):
            mid = len(items) // 2
            r1, f1 = solve_split(fn, items[:mid])
            r2, f2 = solve_split(fn, items[mid:])
            return r1 + r2, f1 + f2
        return [], [(it, msg) for it in items]


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    msg = str(exc)
    return any(tok in msg for tok in _RETRYABLE)


def with_retries(fn, max_retries: int = 5, wait: float = 10.0):
    """Call fn(); on a retryable error wait (exponential backoff with jitter)
    and try again, up to max_retries attempts in total."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries or not is_retryable(e):
                raise
            sleep = wait * 2 ** (attempt - 1) * (0.5 + random.random())
            log.warning('retryable error (attempt %d/%d), sleeping %.0fs: %s',
                        attempt, max_retries, sleep, e)
            time.sleep(sleep)


def init_ee(project: str, base_url: str | None) -> None:
    """Initialize Earth Engine (used as the worker-pool initializer)."""
    def go():
        if base_url:
            ee.Initialize(project=project, opt_url=base_url)
        else:
            ee.Initialize(project=project)
    with_retries(go)


def get_info(obj, cfg):
    return with_retries(obj.getInfo, cfg.max_retries, cfg.retry_wait_secs)


def download_csv(fc, selectors: list, cfg) -> pl.DataFrame:
    """Download a FeatureCollection's properties as CSV into polars.

    ID_PROP is always read as text. Other columns are left for the caller to
    cast; empty CSV fields become nulls.
    """
    def go():
        url = fc.getDownloadURL(filetype='CSV', selectors=selectors)
        r = requests.get(url, timeout=cfg.download_timeout_secs)
        r.raise_for_status()
        return r.content

    content = with_retries(go, cfg.max_retries, cfg.retry_wait_secs)
    if not content.strip():
        return pl.DataFrame(schema={c: pl.Utf8 for c in selectors})
    return pl.read_csv(io.BytesIO(content), schema_overrides={ID_PROP: pl.Utf8},
                       infer_schema_length=None)


def run_jobs(fn, jobs: list, cfg):
    """Yield fn(job) for each job, in completion order.

    n_workers == 1 (or a single job) runs in this process. Otherwise a
    multiprocessing pool is used, each worker initializing Earth Engine once
    -- the same pattern as urma_download_v2.py. Scripts that call this must
    sit under `if __name__ == '__main__':` (required by multiprocessing on
    macOS/Windows).
    """
    if not jobs:
        return
    if cfg.n_workers == 1 or len(jobs) == 1:
        init_ee(cfg.project, cfg.base_url)
        for j in jobs:
            yield fn(j)
        return
    n = min(cfg.n_workers, len(jobs))
    with multiprocessing.Pool(processes=n, initializer=init_ee,
                              initargs=(cfg.project, cfg.base_url)) as pool:
        yield from pool.imap_unordered(fn, jobs)


class Progress:
    """Log progress about every 5% of jobs."""

    def __init__(self, total: int, what: str):
        self.total, self.what = total, what
        self.done = 0
        self.t0 = time.time()
        self.step = max(1, total // 20)

    def tick(self):
        self.done += 1
        if self.done % self.step == 0 or self.done == self.total:
            el = time.time() - self.t0
            rate = self.done / el * 3600 if el > 0 else float('nan')
            log.info('%s: %d/%d done, %.1f min, %.0f jobs/hr', self.what,
                     self.done, self.total, el / 60, rate)
