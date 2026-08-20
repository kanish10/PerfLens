from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from perflens import store
from perflens.models import KernelResultRecord, RunInfo


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("PERFLENS_DB", str(db_path))

    conn = store.get_connection(db_path)
    store.init_db(conn)

    base_run = store.insert_run(
        conn,
        RunInfo(
            ts="2026-01-01T00:00:00+00:00", git_sha="abc123", branch="main",
            gpu="RTX 3060", driver="550.54.14", cuda_version="12.4", suite_hash="h1",
        ),
    )
    store.insert_kernel_results(
        conn,
        base_run,
        [
            KernelResultRecord(
                entry="inference.paged_attention", kernel="paged_attention_kernel",
                rep=i, duration_ns=100.0, occupancy_pct=80.0, regs_per_thread=32,
            )
            for i in range(5)
        ],
    )
    medians = store.compute_medians(conn, base_run, "inference.paged_attention", "paged_attention_kernel")
    median, mad = medians["duration_ns"]
    store.set_baseline(
        conn, "inference.paged_attention", "paged_attention_kernel", "duration_ns",
        median, mad, base_run, "2026-01-01T00:00:00+00:00",
    )

    new_run = store.insert_run(
        conn,
        RunInfo(
            ts="2026-01-02T00:00:00+00:00", git_sha="def456", branch="main",
            gpu="RTX 3060", driver="550.54.14", cuda_version="12.4", suite_hash="h1",
        ),
    )
    store.insert_kernel_results(
        conn,
        new_run,
        [
            KernelResultRecord(
                entry="inference.paged_attention", kernel="paged_attention_kernel",
                rep=i, duration_ns=130.0, occupancy_pct=60.0, regs_per_thread=48,
            )
            for i in range(5)
        ],
    )
    conn.close()

    # Import after PERFLENS_DB is set so the module picks it up at import time,
    # then override the module-level DB_PATH directly for extra safety.
    from perflens.dashboard import api

    api.DB_PATH = str(db_path)
    return TestClient(api.app)


def test_health(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_runs(client: TestClient) -> None:
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert len(runs) == 2
    assert runs[0]["git_sha"] == "def456"  # most recent first


def test_list_runs_filters_by_branch(client: TestClient) -> None:
    resp = client.get("/api/runs", params={"branch": "nonexistent"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_run(client: TestClient) -> None:
    resp = client.get("/api/runs/1")
    assert resp.status_code == 200
    assert resp.json()["git_sha"] == "abc123"


def test_get_run_404(client: TestClient) -> None:
    resp = client.get("/api/runs/999")
    assert resp.status_code == 404


def test_get_findings(client: TestClient) -> None:
    resp = client.get("/api/runs/2/findings")
    assert resp.status_code == 200
    findings = resp.json()
    assert len(findings) == 1
    assert findings[0]["kind"] == "regression"
    assert findings[0]["severity"] == "major"


def test_get_findings_404(client: TestClient) -> None:
    resp = client.get("/api/runs/999/findings")
    assert resp.status_code == 404


def test_get_kernel_results(client: TestClient) -> None:
    resp = client.get("/api/runs/1/kernel-results")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 5
    assert results[0]["duration_ns"] == 100.0


def test_list_entries(client: TestClient) -> None:
    resp = client.get("/api/entries")
    assert resp.status_code == 200
    assert resp.json() == [{"entry": "inference.paged_attention", "kernel": "paged_attention_kernel"}]


def test_metric_history_oldest_first(client: TestClient) -> None:
    resp = client.get(
        "/api/metrics/history",
        params={
            "entry": "inference.paged_attention",
            "kernel": "paged_attention_kernel",
            "metric": "duration_ns",
        },
    )
    assert resp.status_code == 200
    points = resp.json()
    assert [p["value"] for p in points] == [100.0, 130.0]


def test_metric_history_invalid_metric(client: TestClient) -> None:
    resp = client.get(
        "/api/metrics/history",
        params={"entry": "x", "kernel": "y", "metric": "not_a_real_metric"},
    )
    assert resp.status_code == 400
