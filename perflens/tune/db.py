"""SQLite tables for tuning runs, alongside PerfLens's existing schema (same WAL database)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from perflens.tune.strategies import EvalRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS tune_runs(
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  kernel TEXT NOT NULL,
  batch INTEGER NOT NULL,
  strategy TEXT NOT NULL,
  seed INTEGER NOT NULL,
  budget INTEGER NOT NULL,
  gpu TEXT NOT NULL,
  driver TEXT NOT NULL,
  engine_sha TEXT,
  model TEXT,
  space_json TEXT NOT NULL,
  stop_reason TEXT,
  invalid_proposals INTEGER NOT NULL DEFAULT 0,
  finished_ts TEXT
);

CREATE TABLE IF NOT EXISTS tune_evals(
  run_id INTEGER NOT NULL REFERENCES tune_runs(id),
  idx INTEGER NOT NULL,
  config_key TEXT NOT NULL,
  config_json TEXT NOT NULL,
  build_ok INTEGER NOT NULL,
  cached_build INTEGER NOT NULL,
  compile_ms REAL,
  registers INTEGER,
  spill_bytes INTEGER,
  smem_bytes INTEGER,
  ptxas_warnings TEXT,
  correct INTEGER NOT NULL,
  max_abs_err REAL,
  median_us REAL,
  mad_us REAL,
  rationale TEXT,
  error TEXT,
  PRIMARY KEY(run_id, idx)
);
"""


def init_tune_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def insert_run(
    conn: sqlite3.Connection,
    *,
    kernel: str,
    batch: int,
    strategy: str,
    seed: int,
    budget: int,
    gpu: str,
    driver: str,
    engine_sha: str | None,
    model: str | None,
    space: dict[str, Any],
) -> int:
    cur = conn.execute(
        """INSERT INTO tune_runs(ts, kernel, batch, strategy, seed, budget, gpu, driver,
                                 engine_sha, model, space_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(UTC).isoformat(),
            kernel,
            batch,
            strategy,
            seed,
            budget,
            gpu,
            driver,
            engine_sha,
            model,
            json.dumps(space, sort_keys=True),
        ),
    )
    conn.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


def insert_eval(conn: sqlite3.Connection, run_id: int, record: EvalRecord) -> None:
    r = record.result
    conn.execute(
        """INSERT INTO tune_evals(run_id, idx, config_key, config_json, build_ok, cached_build,
                                  compile_ms, registers, spill_bytes, smem_bytes, ptxas_warnings,
                                  correct, max_abs_err, median_us, mad_us, rationale, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            record.idx,
            r.config.key(),
            json.dumps(r.config.model_dump(), sort_keys=True),
            int(r.build_ok),
            int(r.cached_build),
            r.compile_ms,
            r.registers,
            r.spill_bytes,
            r.smem_bytes,
            json.dumps(r.ptxas_warnings),
            int(r.correct),
            r.max_abs_err,
            r.median_us,
            r.mad_us,
            record.rationale,
            r.error,
        ),
    )
    conn.commit()


def finish_run(conn: sqlite3.Connection, run_id: int, stop_reason: str, invalid_proposals: int) -> None:
    conn.execute(
        "UPDATE tune_runs SET stop_reason = ?, invalid_proposals = ?, finished_ts = ? WHERE id = ?",
        (stop_reason, invalid_proposals, datetime.now(UTC).isoformat(), run_id),
    )
    conn.commit()


def list_runs(conn: sqlite3.Connection, finished_only: bool = True) -> list[sqlite3.Row]:
    sql = "SELECT * FROM tune_runs"
    if finished_only:
        sql += " WHERE finished_ts IS NOT NULL"
    return list(conn.execute(sql + " ORDER BY id"))


def get_evals(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM tune_evals WHERE run_id = ? ORDER BY idx", (run_id,)))
