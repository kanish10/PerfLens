"""Fixture-based tests for ncu CSV parsing (M1). No GPU or `ncu` binary required."""

from __future__ import annotations

from pathlib import Path

import pytest

from perflens.collect import CollectionError, parse_ncu_csv, to_kernel_results

FIXTURE = Path(__file__).parent / "fixtures" / "sample_ncu.csv"


def test_parse_ncu_csv_groups_by_launch_id() -> None:
    rows = parse_ncu_csv(FIXTURE.read_text())
    assert len(rows) == 3  # 2 paged_attention launches + 1 bias_gelu launch
    assert rows[0].kernel.startswith("paged_attention_kernel")
    assert rows[2].kernel.startswith("bias_gelu_kernel")


def test_parse_ncu_csv_converts_duration_to_ns() -> None:
    rows = parse_ncu_csv(FIXTURE.read_text())
    # first row: 12.450 usecond -> 12450.0 ns
    assert float(rows[0].metrics["gpu__time_duration.sum"]) == pytest.approx(12450.0)


def test_parse_ncu_csv_missing_columns_raises() -> None:
    with pytest.raises(CollectionError):
        parse_ncu_csv("a,b,c\n1,2,3\n")


def test_to_kernel_results_assigns_rep_index_per_kernel() -> None:
    rows = parse_ncu_csv(FIXTURE.read_text())
    records = to_kernel_results("inference.paged_attention", rows)
    reps = [r.rep for r in records if r.kernel.startswith("paged_attention_kernel")]
    assert reps == [0, 1]
    gelu_reps = [r.rep for r in records if r.kernel.startswith("bias_gelu_kernel")]
    assert gelu_reps == [0]


def test_to_kernel_results_maps_all_metric_fields() -> None:
    rows = parse_ncu_csv(FIXTURE.read_text())
    records = to_kernel_results("inference.paged_attention", rows)
    r = records[0]
    assert r.duration_ns == pytest.approx(12450.0)
    assert r.occupancy_pct == pytest.approx(78.32)
    assert r.dram_pct == pytest.approx(41.10)
    assert r.l2_hit_pct == pytest.approx(62.75)
    assert r.regs_per_thread == 32
    assert r.block_size == 256
    assert r.grid_size == 64
    assert r.ipc == pytest.approx(1.85)
    assert r.entry == "inference.paged_attention"


def test_to_kernel_results_preserves_raw_metrics() -> None:
    rows = parse_ncu_csv(FIXTURE.read_text())
    records = to_kernel_results("inference.paged_attention", rows)
    assert "gpu__time_duration.sum" in records[0].raw


def test_to_kernel_results_skips_rows_without_primary_metric() -> None:
    from perflens.collect import RawKernelRow

    rows = [RawKernelRow(kernel="orphan_kernel", metrics={"launch__block_size": "128"})]
    records = to_kernel_results("entry", rows)
    assert records == []
