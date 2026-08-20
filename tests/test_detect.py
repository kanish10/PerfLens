"""Synthetic-distribution tests for the median + 3*MAD regression detector.

These are the tests CLAUDE.md calls out explicitly for M3: a clean regression
is caught, small noise is not flagged, a high-MAD kernel needs a bigger delta
to trip the gate, and the improvement path is exercised too.
"""

from __future__ import annotations

import sqlite3

import pytest

from perflens import detect, store
from perflens.models import KernelResultRecord

ENTRY = "inference.paged_attention"
KERNEL = "paged_attention_kernel"


def _seed_baseline(conn: sqlite3.Connection, make_run, durations: list[float]) -> int:
    run_id = make_run(ts="2026-01-01T00:00:00+00:00")
    records = [
        KernelResultRecord(entry=ENTRY, kernel=KERNEL, rep=i, duration_ns=d)
        for i, d in enumerate(durations)
    ]
    store.insert_kernel_results(conn, run_id, records)
    median, mad = store.compute_medians(conn, run_id, ENTRY, KERNEL)["duration_ns"]
    store.set_baseline(conn, ENTRY, KERNEL, "duration_ns", median, mad, run_id, "2026-01-01T00:00:00+00:00")
    return run_id


def _seed_new_run(conn: sqlite3.Connection, make_run, durations: list[float]) -> int:
    run_id = make_run(ts="2026-01-02T00:00:00+00:00")
    records = [
        KernelResultRecord(entry=ENTRY, kernel=KERNEL, rep=i, duration_ns=d)
        for i, d in enumerate(durations)
    ]
    store.insert_kernel_results(conn, run_id, records)
    return run_id


def test_clean_20pct_regression_is_caught(conn: sqlite3.Connection, make_run) -> None:
    _seed_baseline(conn, make_run, [100.0, 101.0, 99.0, 100.0, 100.0])  # tight, low MAD
    new_run = _seed_new_run(conn, make_run, [120.0, 121.0, 119.0, 120.0, 120.0])  # +20%

    findings = detect.detect_regressions(conn, new_run)

    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "regression"
    assert f.severity == "major"
    assert f.delta_pct > 15.0


def test_3pct_noise_not_flagged(conn: sqlite3.Connection, make_run) -> None:
    _seed_baseline(conn, make_run, [100.0, 101.0, 99.0, 100.0, 100.0])
    new_run = _seed_new_run(conn, make_run, [103.0, 102.0, 104.0, 103.0, 103.0])  # ~3%

    findings = detect.detect_regressions(conn, new_run)

    assert findings == []


def test_high_mad_kernel_needs_bigger_delta(conn: sqlite3.Connection, make_run) -> None:
    # Wide baseline spread -> large MAD. A modest 6% median shift should NOT
    # clear the 3*MAD noise floor even though it clears the 5% threshold.
    _seed_baseline(conn, make_run, [80.0, 90.0, 100.0, 110.0, 120.0])  # median 100, MAD 20
    small_shift_run = _seed_new_run(conn, make_run, [86.0, 96.0, 106.0, 116.0, 126.0])  # median 106, +6%
    findings = detect.detect_regressions(conn, small_shift_run)
    assert findings == []  # 6 < 3*20=60 noise floor

    # A much larger delta on the same noisy kernel DOES clear both gates.
    big_shift_run_id = make_run(ts="2026-01-03T00:00:00+00:00")
    big_records = [
        KernelResultRecord(entry=ENTRY, kernel=KERNEL, rep=i, duration_ns=d)
        for i, d in enumerate([200.0, 210.0, 220.0, 230.0, 240.0])  # median 220, +120%
    ]
    store.insert_kernel_results(conn, big_shift_run_id, big_records)
    findings = detect.detect_regressions(conn, big_shift_run_id)
    assert len(findings) == 1
    assert findings[0].kind == "regression"
    assert findings[0].severity == "major"


def test_improvement_path(conn: sqlite3.Connection, make_run) -> None:
    _seed_baseline(conn, make_run, [100.0, 101.0, 99.0, 100.0, 100.0])
    new_run = _seed_new_run(conn, make_run, [70.0, 71.0, 69.0, 70.0, 70.0])  # -30%

    findings = detect.detect_regressions(conn, new_run)

    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "improvement"
    assert f.severity is None
    assert f.delta_pct < 0
    assert f.suggested_command is not None
    assert "baseline set" in f.suggested_command


def test_minor_vs_major_severity_boundary(conn: sqlite3.Connection, make_run) -> None:
    _seed_baseline(conn, make_run, [1000.0, 1001.0, 999.0, 1000.0, 1000.0])

    minor_run = _seed_new_run(conn, make_run, [1080.0, 1081.0, 1079.0, 1080.0, 1080.0])  # +8%
    findings = detect.detect_regressions(conn, minor_run)
    assert findings[0].severity == "minor"

    major_run_id = make_run(ts="2026-01-04T00:00:00+00:00")
    records = [
        KernelResultRecord(entry=ENTRY, kernel=KERNEL, rep=i, duration_ns=d)
        for i, d in enumerate([1200.0, 1201.0, 1199.0, 1200.0, 1200.0])  # +20%
    ]
    store.insert_kernel_results(conn, major_run_id, records)
    findings = detect.detect_regressions(conn, major_run_id)
    assert findings[0].severity == "major"


def test_no_baseline_means_no_finding(conn: sqlite3.Connection, make_run) -> None:
    run_id = _seed_new_run(conn, make_run, [100.0, 100.0, 100.0, 100.0, 100.0])
    assert detect.detect_regressions(conn, run_id) == []


def test_secondary_metrics_attached_as_evidence_never_gate_alone(
    conn: sqlite3.Connection, make_run
) -> None:
    base_run = make_run(ts="2026-01-01T00:00:00+00:00")
    base_records = [
        KernelResultRecord(
            entry=ENTRY, kernel=KERNEL, rep=i, duration_ns=100.0,
            occupancy_pct=80.0, regs_per_thread=32,
        )
        for i in range(5)
    ]
    store.insert_kernel_results(conn, base_run, base_records)
    for metric in ("duration_ns", "occupancy_pct", "regs_per_thread"):
        median, mad = store.compute_medians(conn, base_run, ENTRY, KERNEL)[metric]
        store.set_baseline(conn, ENTRY, KERNEL, metric, median, mad, base_run, "2026-01-01T00:00:00+00:00")

    # Occupancy craters and regs spike, but duration_ns stays flat -> must NOT
    # be flagged as a regression on its own (secondary metrics are diagnostic only).
    flat_duration_run = make_run(ts="2026-01-02T00:00:00+00:00")
    flat_records = [
        KernelResultRecord(
            entry=ENTRY, kernel=KERNEL, rep=i, duration_ns=100.0,
            occupancy_pct=20.0, regs_per_thread=96,
        )
        for i in range(5)
    ]
    store.insert_kernel_results(conn, flat_duration_run, flat_records)
    assert detect.detect_regressions(conn, flat_duration_run) == []

    # Now duration_ns also regresses -- evidence dict should carry the
    # secondary deltas alongside the primary-metric finding.
    real_regression_run = make_run(ts="2026-01-03T00:00:00+00:00")
    reg_records = [
        KernelResultRecord(
            entry=ENTRY, kernel=KERNEL, rep=i, duration_ns=130.0,
            occupancy_pct=20.0, regs_per_thread=96,
        )
        for i in range(5)
    ]
    store.insert_kernel_results(conn, real_regression_run, reg_records)
    findings = detect.detect_regressions(conn, real_regression_run)
    assert len(findings) == 1
    evidence = findings[0].evidence
    assert evidence["occupancy_pct_delta_pct"] < 0
    assert evidence["regs_per_thread_delta_pct"] > 0


def test_gpu_driver_mismatch_skips_comparison(conn: sqlite3.Connection, make_run) -> None:
    base_run = make_run(ts="2026-01-01T00:00:00+00:00", gpu="RTX 3060", driver="550.54.14")
    records = [
        KernelResultRecord(entry=ENTRY, kernel=KERNEL, rep=i, duration_ns=100.0) for i in range(5)
    ]
    store.insert_kernel_results(conn, base_run, records)
    median, mad = store.compute_medians(conn, base_run, ENTRY, KERNEL)["duration_ns"]
    store.set_baseline(conn, ENTRY, KERNEL, "duration_ns", median, mad, base_run, "2026-01-01T00:00:00+00:00")

    other_gpu_run = make_run(ts="2026-01-02T00:00:00+00:00", gpu="RTX 4090", driver="550.54.14")
    big_records = [
        KernelResultRecord(entry=ENTRY, kernel=KERNEL, rep=i, duration_ns=500.0) for i in range(5)
    ]
    store.insert_kernel_results(conn, other_gpu_run, big_records)

    with pytest.warns(UserWarning, match="RTX 3060.*RTX 4090"):
        assert detect.detect_regressions(conn, other_gpu_run) == []
