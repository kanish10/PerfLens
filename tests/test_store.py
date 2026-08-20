from __future__ import annotations

import sqlite3

from perflens import store
from perflens.models import KernelResultRecord


def _records(entry: str, kernel: str, durations: list[float]) -> list[KernelResultRecord]:
    return [
        KernelResultRecord(
            entry=entry,
            kernel=kernel,
            rep=i,
            duration_ns=d,
            occupancy_pct=50.0 + i,
            regs_per_thread=32,
        )
        for i, d in enumerate(durations)
    ]


def test_insert_and_get_run(conn: sqlite3.Connection, make_run) -> None:
    run_id = make_run(git_sha="deadbeef", branch="main")
    row = store.get_run(conn, run_id)
    assert row is not None
    assert row["git_sha"] == "deadbeef"
    assert row["branch"] == "main"


def test_get_latest_run_filters_by_branch(conn: sqlite3.Connection, make_run) -> None:
    make_run(branch="main", ts="2026-01-01T00:00:00+00:00")
    second = make_run(branch="feature", ts="2026-01-02T00:00:00+00:00")
    latest_main = store.get_latest_run(conn, branch="main")
    latest_feature = store.get_latest_run(conn, branch="feature")
    assert latest_main["branch"] == "main"
    assert latest_feature["id"] == second


def test_insert_kernel_results_and_list_entry_kernels(conn: sqlite3.Connection, make_run) -> None:
    run_id = make_run()
    records = _records("inference.paged_attention", "attn_kernel", [100, 102, 98, 101, 99])
    n = store.insert_kernel_results(conn, run_id, records)
    assert n == 5
    assert store.list_entry_kernels(conn, run_id) == [("inference.paged_attention", "attn_kernel")]


def test_compute_medians(conn: sqlite3.Connection, make_run) -> None:
    run_id = make_run()
    store.insert_kernel_results(
        conn, run_id, _records("entry", "kernel", [100.0, 102.0, 98.0, 101.0, 99.0])
    )
    medians = store.compute_medians(conn, run_id, "entry", "kernel")
    median, mad = medians["duration_ns"]
    assert median == 100.0
    assert mad == 1.0  # |100-100|,|102-100|,|98-100|,|101-100|,|99-100| -> [0,2,2,1,1] -> median 1


def test_compute_medians_missing_entry_returns_empty(conn: sqlite3.Connection, make_run) -> None:
    run_id = make_run()
    assert store.compute_medians(conn, run_id, "nope", "nope") == {}


def test_set_and_get_baseline(conn: sqlite3.Connection, make_run) -> None:
    run_id = make_run()
    ts = "2026-01-01T00:00:00+00:00"
    store.set_baseline(conn, "entry", "kernel", "duration_ns", 100.0, 2.0, run_id, ts)
    row = store.get_baseline(conn, "entry", "kernel", "duration_ns")
    assert row is not None
    assert row["median"] == 100.0
    assert row["mad"] == 2.0


def test_set_baseline_upsert_overwrites(conn: sqlite3.Connection, make_run) -> None:
    run_id = make_run()
    store.set_baseline(
        conn, "entry", "kernel", "duration_ns", 100.0, 2.0, run_id, "2026-01-01T00:00:00+00:00"
    )
    store.set_baseline(
        conn, "entry", "kernel", "duration_ns", 150.0, 3.0, run_id, "2026-01-02T00:00:00+00:00"
    )
    row = store.get_baseline(conn, "entry", "kernel", "duration_ns")
    assert row["median"] == 150.0
    assert row["mad"] == 3.0


def test_get_baseline_missing_returns_none(conn: sqlite3.Connection) -> None:
    assert store.get_baseline(conn, "x", "y", "duration_ns") is None


def test_get_metric_history_orders_most_recent_first(conn: sqlite3.Connection, make_run) -> None:
    run1 = make_run(ts="2026-01-01T00:00:00+00:00")
    store.insert_kernel_results(conn, run1, _records("entry", "kernel", [100.0] * 5))
    run2 = make_run(ts="2026-01-02T00:00:00+00:00")
    store.insert_kernel_results(conn, run2, _records("entry", "kernel", [110.0] * 5))

    history = store.get_metric_history(conn, "entry", "kernel", "duration_ns", n=10)
    assert [v for _, v in history] == [110.0, 100.0]


def test_get_metric_history_unknown_metric_raises(conn: sqlite3.Connection) -> None:
    import pytest

    with pytest.raises(ValueError):
        store.get_metric_history(conn, "entry", "kernel", "not_a_metric")


def test_wal_mode_on_file_db(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    c = store.get_connection(db_path)
    store.init_db(c)
    mode = c.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    c.close()
