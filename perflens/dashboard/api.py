"""Read-only FastAPI service over the same SQLite store `perflens run` writes to.

Run it with `perflens dashboard` (see cli.py) or directly:
    uvicorn perflens.dashboard.api:app --reload
Configure the DB path with the PERFLENS_DB env var (default: perflens.db).
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from perflens import detect, store
from perflens.models import Finding, KernelResultRecord

DB_PATH = os.environ.get("PERFLENS_DB", "perflens.db")

app = FastAPI(title="PerfLens Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "PERFLENS_DASHBOARD_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


# Schema setup is DDL (CREATE TABLE/INDEX) plus a commit -- run it once per
# DB path the process has seen, not on every request. A concurrent CLI
# writer (`perflens run`, `baseline set`) can hold the WAL write lock at any
# moment; re-running DDL on every GET risks a transient "database is locked"
# on requests that would otherwise be a plain, harmless read.
_initialized_dbs: set[str] = set()


def get_conn() -> Iterator[sqlite3.Connection]:
    conn = store.get_connection(DB_PATH)
    if DB_PATH not in _initialized_dbs:
        store.init_db(conn)
        _initialized_dbs.add(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


class RunOut(BaseModel):
    id: int
    ts: str
    git_sha: str
    branch: str
    gpu: str
    driver: str
    cuda_version: str
    suite_hash: str


class EntryKernel(BaseModel):
    entry: str
    kernel: str


class HistoryPoint(BaseModel):
    ts: str
    value: float


def _get_run_or_404(conn: sqlite3.Connection, run_id: int) -> dict[str, object]:
    row = store.get_run(conn, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return dict(row)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs", response_model=list[RunOut])
def list_runs(
    branch: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[dict[str, object]]:
    if branch:
        rows = conn.execute(
            "SELECT * FROM runs WHERE branch = ? ORDER BY id DESC LIMIT ?", (branch, limit)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/runs/{run_id}", response_model=RunOut)
def get_run(run_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, object]:
    return _get_run_or_404(conn, run_id)


@app.get("/api/runs/{run_id}/findings", response_model=list[Finding])
def get_findings(
    run_id: int,
    threshold: float = detect.DEFAULT_THRESHOLD,
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[Finding]:
    _get_run_or_404(conn, run_id)
    return detect.detect_regressions(conn, run_id, threshold=threshold)


@app.get("/api/runs/{run_id}/kernel-results", response_model=list[KernelResultRecord])
def get_kernel_results(
    run_id: int,
    entry: str | None = None,
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[KernelResultRecord]:
    _get_run_or_404(conn, run_id)
    rows = store.get_kernel_results(conn, run_id, entry=entry)
    return [
        KernelResultRecord(
            entry=r["entry"],
            kernel=r["kernel"],
            rep=r["rep"],
            duration_ns=r["duration_ns"],
            occupancy_pct=r["occupancy_pct"],
            dram_pct=r["dram_pct"],
            l2_hit_pct=r["l2_hit_pct"],
            regs_per_thread=r["regs_per_thread"],
            block_size=r["block_size"],
            grid_size=r["grid_size"],
            ipc=r["ipc"],
        )
        for r in rows
    ]


@app.get("/api/entries", response_model=list[EntryKernel])
def list_entries(
    run_id: int | None = None, conn: sqlite3.Connection = Depends(get_conn)
) -> list[dict[str, str]]:
    if run_id is not None:
        _get_run_or_404(conn, run_id)
        pairs = store.list_entry_kernels(conn, run_id)
    else:
        rows = conn.execute(
            "SELECT DISTINCT entry, kernel FROM kernel_results ORDER BY entry, kernel"
        ).fetchall()
        pairs = [(r["entry"], r["kernel"]) for r in rows]
    return [{"entry": e, "kernel": k} for e, k in pairs]


@app.get("/api/metrics/history", response_model=list[HistoryPoint])
def metric_history(
    entry: str,
    kernel: str,
    metric: str = "duration_ns",
    n: int = Query(20, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[dict[str, float | str]]:
    try:
        points = store.get_metric_history(conn, entry, kernel, metric, n=n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [{"ts": ts, "value": v} for ts, v in reversed(points)]  # oldest first, for charting
