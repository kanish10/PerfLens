from __future__ import annotations

import sqlite3

import pytest

from perflens import store
from perflens.models import RunInfo


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    store.init_db(c)
    yield c
    c.close()


@pytest.fixture
def make_run(conn: sqlite3.Connection):
    def _make(
        ts: str = "2026-01-01T00:00:00+00:00",
        git_sha: str = "abc123",
        branch: str = "main",
        gpu: str = "RTX 3060",
        driver: str = "550.54.14",
        cuda_version: str = "12.4",
        suite_hash: str = "deadbeef",
    ) -> int:
        run = RunInfo(
            ts=ts,
            git_sha=git_sha,
            branch=branch,
            gpu=gpu,
            driver=driver,
            cuda_version=cuda_version,
            suite_hash=suite_hash,
        )
        return store.insert_run(conn, run)

    return _make
