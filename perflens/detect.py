"""Noise-aware regression detector: median + 3*MAD gate on gpu__time_duration.sum,
with secondary metrics attached as diagnostic-only evidence (never gate alone).
"""

from __future__ import annotations

import sqlite3
import warnings

from perflens import store
from perflens.models import Finding, Severity

DEFAULT_THRESHOLD = 0.05  # 5% relative-delta gate
MAD_GUARD_K = 3.0
MAJOR_SEVERITY_PCT = 15.0  # >15% delta -> major, [5, 15]% -> minor

PRIMARY_METRIC = "duration_ns"

# For each secondary metric: True if "up" is the regression-consistent direction
# (i.e. worth flagging as corroborating evidence for a duration_ns regression).
_SECONDARY_UP_IS_BAD = {
    "regs_per_thread": True,
    "dram_pct": True,
    "occupancy_pct": False,  # occupancy DOWN is bad
    "l2_hit_pct": False,  # L2 hit rate DOWN is bad
}


def _pct_delta(new: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (new - base) / base * 100.0


def _secondary_evidence(
    conn: sqlite3.Connection, run_id: int, entry: str, kernel: str, base_metrics: dict
) -> dict[str, float]:
    new_metrics = store.compute_medians(conn, run_id, entry, kernel)
    evidence: dict[str, float] = {}
    for metric in _SECONDARY_UP_IS_BAD:
        if metric not in new_metrics or metric not in base_metrics:
            continue
        new_med, _ = new_metrics[metric]
        base_med, _ = base_metrics[metric]
        evidence[f"{metric}_delta_pct"] = round(_pct_delta(new_med, base_med), 3)
    return evidence


def detect_regressions(
    conn: sqlite3.Connection,
    run_id: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Finding]:
    """Compare `run_id`'s kernel medians against stored baselines for every
    (entry, kernel) present in the run. Returns regression AND improvement findings.
    """
    run = store.get_run(conn, run_id)
    if run is None:
        raise ValueError(f"no such run: {run_id}")

    findings: list[Finding] = []
    for entry, kernel in store.list_entry_kernels(conn, run_id):
        baseline = store.get_baseline(conn, entry, kernel, PRIMARY_METRIC)
        if baseline is None:
            continue  # nothing to compare against yet

        baseline_run = store.get_run(conn, baseline["run_id"])
        if baseline_run is not None and (
            baseline_run["gpu"] != run["gpu"] or baseline_run["driver"] != run["driver"]
        ):
            # Discipline: never compare across different gpu/driver strings.
            warnings.warn(
                f"skipping {entry}/{kernel}: baseline was set on "
                f"{baseline_run['gpu']!r}/{baseline_run['driver']!r}, "
                f"this run is on {run['gpu']!r}/{run['driver']!r} -- "
                "run `perflens baseline set` on this GPU/driver to compare",
                stacklevel=2,
            )
            continue

        new_metrics = store.compute_medians(conn, run_id, entry, kernel)
        if PRIMARY_METRIC not in new_metrics:
            continue
        new_median, _ = new_metrics[PRIMARY_METRIC]
        base_median = baseline["median"]
        base_mad = baseline["mad"]

        delta_pct = _pct_delta(new_median, base_median)

        is_regression = new_median > base_median * (1 + threshold) and new_median > (
            base_median + MAD_GUARD_K * base_mad
        )
        is_improvement = new_median < base_median * (1 - threshold) and new_median < (
            base_median - MAD_GUARD_K * base_mad
        )

        if not (is_regression or is_improvement):
            continue

        base_metrics_for_evidence = {PRIMARY_METRIC: (base_median, base_mad)}
        for metric in _SECONDARY_UP_IS_BAD:
            b = store.get_baseline(conn, entry, kernel, metric)
            if b is not None:
                base_metrics_for_evidence[metric] = (b["median"], b["mad"])
        evidence = _secondary_evidence(conn, run_id, entry, kernel, base_metrics_for_evidence)

        if is_regression:
            severity: Severity = "major" if abs(delta_pct) > MAJOR_SEVERITY_PCT else "minor"
            findings.append(
                Finding(
                    entry=entry,
                    kernel=kernel,
                    metric=PRIMARY_METRIC,
                    base_median=base_median,
                    new_median=new_median,
                    delta_pct=round(delta_pct, 3),
                    evidence=evidence,
                    severity=severity,
                    kind="regression",
                )
            )
        else:
            findings.append(
                Finding(
                    entry=entry,
                    kernel=kernel,
                    metric=PRIMARY_METRIC,
                    base_median=base_median,
                    new_median=new_median,
                    delta_pct=round(delta_pct, 3),
                    evidence=evidence,
                    severity=None,
                    kind="improvement",
                    suggested_command=f"perflens baseline set --run {run_id}",
                )
            )

    return findings
