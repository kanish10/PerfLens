"""SQLite storage layer (WAL mode) for runs, per-rep kernel results, and baselines."""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections.abc import Iterable
from pathlib import Path

from perflens.models import KernelResultRecord, RunInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  git_sha TEXT NOT NULL,
  branch TEXT NOT NULL,
  gpu TEXT NOT NULL,
  driver TEXT NOT NULL,
  cuda_version TEXT NOT NULL,
  suite_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kernel_results(
  run_id INTEGER NOT NULL REFERENCES runs(id),
  entry TEXT NOT NULL,
  kernel TEXT NOT NULL,
  rep INTEGER NOT NULL,
  duration_ns REAL NOT NULL,
  occupancy_pct REAL,
  dram_pct REAL,
  l2_hit_pct REAL,
  regs_per_thread INTEGER,
  block_size INTEGER,
  grid_size INTEGER,
  ipc REAL,
  source_sha TEXT,
  raw_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_kernel_results_run
  ON kernel_results(run_id, entry, kernel);

CREATE TABLE IF NOT EXISTS baselines(
  entry TEXT NOT NULL,
  kernel TEXT NOT NULL,
  metric TEXT NOT NULL,
  median REAL NOT NULL,
  mad REAL NOT NULL,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  set_ts TEXT NOT NULL,
  PRIMARY KEY(entry, kernel, metric)
);
"""

# Secondary (diagnostic-only) metrics tracked alongside the primary duration_ns metric.
SECONDARY_METRICS = (
    "occupancy_pct",
    "dram_pct",
    "l2_hit_pct",
    "regs_per_thread",
    "block_size",
    "grid_size",
    "ipc",
)


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def insert_run(conn: sqlite3.Connection, run: RunInfo) -> int:
    cur = conn.execute(
        """INSERT INTO runs(ts, git_sha, branch, gpu, driver, cuda_version, suite_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (run.ts, run.git_sha, run.branch, run.gpu, run.driver, run.cuda_version, run.suite_hash),
    )
    conn.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


def insert_kernel_results(
    conn: sqlite3.Connection, run_id: int, records: Iterable[KernelResultRecord]
) -> int:
    rows = [
        (
            run_id,
            r.entry,
            r.kernel,
            r.rep,
            r.duration_ns,
            r.occupancy_pct,
            r.dram_pct,
            r.l2_hit_pct,
            r.regs_per_thread,
            r.block_size,
            r.grid_size,
            r.ipc,
            r.source_sha,
            json.dumps(r.raw, sort_keys=True),
        )
        for r in records
    ]
    conn.executemany(
        """INSERT INTO kernel_results(
             run_id, entry, kernel, rep, duration_ns, occupancy_pct, dram_pct,
             l2_hit_pct, regs_per_thread, block_size, grid_size, ipc, source_sha, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()


def get_entry_source_sha(conn: sqlite3.Connection, run_id: int, entry: str) -> str | None:
    """The HEAD of `entry`'s own source repo (SuiteEntry.cwd) recorded at
    collection time for this run -- distinct from runs.git_sha, which is
    this repo's own commit. Different entries can live in different sibling
    repos with independent histories (see suite.yaml).
    """
    row = conn.execute(
        "SELECT source_sha FROM kernel_results WHERE run_id = ? AND entry = ? LIMIT 1",
        (run_id, entry),
    ).fetchone()
    return row["source_sha"] if row is not None else None


def get_latest_run(conn: sqlite3.Connection, branch: str | None = None) -> sqlite3.Row | None:
    if branch is not None:
        return conn.execute(
            "SELECT * FROM runs WHERE branch = ? ORDER BY id DESC LIMIT 1", (branch,)
        ).fetchone()
    return conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()


def get_kernel_results(
    conn: sqlite3.Connection, run_id: int, entry: str | None = None
) -> list[sqlite3.Row]:
    if entry is not None:
        return conn.execute(
            "SELECT * FROM kernel_results WHERE run_id = ? AND entry = ? ORDER BY kernel, rep",
            (run_id, entry),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM kernel_results WHERE run_id = ? ORDER BY entry, kernel, rep", (run_id,)
    ).fetchall()


def list_entry_kernels(conn: sqlite3.Connection, run_id: int) -> list[tuple[str, str]]:
    rows = conn.execute(
        "SELECT DISTINCT entry, kernel FROM kernel_results WHERE run_id = ? ORDER BY entry, kernel",
        (run_id,),
    ).fetchall()
    return [(r["entry"], r["kernel"]) for r in rows]


def _median_mad(values: list[float]) -> tuple[float, float]:
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values]) if len(values) > 1 else 0.0
    return med, mad


def compute_medians(
    conn: sqlite3.Connection, run_id: int, entry: str, kernel: str
) -> dict[str, tuple[float, float]]:
    """Median + MAD per metric for one (entry, kernel) within a single run's reps."""
    rows = conn.execute(
        """SELECT duration_ns, occupancy_pct, dram_pct, l2_hit_pct, regs_per_thread,
                  block_size, grid_size, ipc
           FROM kernel_results WHERE run_id = ? AND entry = ? AND kernel = ?""",
        (run_id, entry, kernel),
    ).fetchall()
    if not rows:
        return {}
    out: dict[str, tuple[float, float]] = {}
    for metric in ("duration_ns", *SECONDARY_METRICS):
        values = [r[metric] for r in rows if r[metric] is not None]
        if values:
            out[metric] = _median_mad([float(v) for v in values])
    return out


def set_baseline(
    conn: sqlite3.Connection,
    entry: str,
    kernel: str,
    metric: str,
    median: float,
    mad: float,
    run_id: int,
    set_ts: str,
) -> None:
    conn.execute(
        """INSERT INTO baselines(entry, kernel, metric, median, mad, run_id, set_ts)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(entry, kernel, metric) DO UPDATE SET
             median=excluded.median, mad=excluded.mad,
             run_id=excluded.run_id, set_ts=excluded.set_ts""",
        (entry, kernel, metric, median, mad, run_id, set_ts),
    )
    conn.commit()


def get_baseline(
    conn: sqlite3.Connection, entry: str, kernel: str, metric: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM baselines WHERE entry = ? AND kernel = ? AND metric = ?",
        (entry, kernel, metric),
    ).fetchone()


def get_metric_history(
    conn: sqlite3.Connection, entry: str, kernel: str, metric: str, n: int = 10
) -> list[tuple[str, float]]:
    """Recent per-run medians for one metric, most recent first."""
    if metric not in ("duration_ns", *SECONDARY_METRICS):
        raise ValueError(f"unknown metric: {metric}")
    rows = conn.execute(
        f"""SELECT r.ts as ts, kr.{metric} as val
            FROM kernel_results kr JOIN runs r ON r.id = kr.run_id
            WHERE kr.entry = ? AND kr.kernel = ? AND kr.{metric} IS NOT NULL
            ORDER BY r.id DESC""",
        (entry, kernel),
    ).fetchall()
    by_run: dict[str, list[float]] = {}
    order: list[str] = []
    for r in rows:
        if r["ts"] not in by_run:
            by_run[r["ts"]] = []
            order.append(r["ts"])
        by_run[r["ts"]].append(float(r["val"]))
    out = [(ts, statistics.median(by_run[ts])) for ts in order[:n]]
    return out
