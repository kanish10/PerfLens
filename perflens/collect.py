"""Nsight Compute (ncu) invocation and CSV parsing.

ncu serializes and replays kernels; durations reported here are kernel-level,
not application wall-clock. That is exactly what we want for kernel regression
detection -- app-level throughput regressions are a different signal and are
out of scope for this channel (see CLAUDE.md M7 for the wall-clock ingestion
stretch goal).
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
from dataclasses import dataclass, field

from perflens.models import KernelResultRecord, SuiteEntry

METRIC_SET = (
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "launch__registers_per_thread",
    "launch__block_size",
    "launch__grid_size",
    "smsp__inst_executed.avg.per_cycle_active",
)

# ncu Metric Name -> KernelResultRecord field
_METRIC_FIELD_MAP = {
    "gpu__time_duration.sum": "duration_ns",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_pct",
    "launch__registers_per_thread": "regs_per_thread",
    "launch__block_size": "block_size",
    "launch__grid_size": "grid_size",
    "smsp__inst_executed.avg.per_cycle_active": "ipc",
}
_INT_FIELDS = {"regs_per_thread", "block_size", "grid_size"}

_TIME_UNIT_TO_NS = {
    "ns": 1.0,
    "nsecond": 1.0,
    "us": 1_000.0,
    "usecond": 1_000.0,
    "ms": 1_000_000.0,
    "msecond": 1_000_000.0,
    "s": 1_000_000_000.0,
    "second": 1_000_000_000.0,
}


class CollectionError(RuntimeError):
    """Raised when the `ncu` invocation fails or its output can't be parsed."""


PERMISSION_FIX_HINT = (
    "ncu exited nonzero -- this usually means the profiling counters are locked down.\n"
    "Fix: either run with elevated privileges once to confirm, then set persistent access via\n"
    "  sudo modprobe -r nvidia_uvm nvidia && \\\n"
    "  sudo sh -c 'echo \"options nvidia NVreg_RestrictProfilingToAdminUsers=0\" "
    "> /etc/modprobe.d/nvidia-profiling.conf' && sudo update-initramfs -u\n"
    "or add the invoking user to a sudo-less profiling group per your distro's NVIDIA driver docs,\n"
    "then reboot (or reload the nvidia kernel modules) for the setting to take effect."
)


def build_ncu_command(entry: SuiteEntry, reps: int, metrics: tuple[str, ...] = METRIC_SET) -> list[str]:
    return [
        "ncu",
        "--csv",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        f"regex:{entry.kernel_regex}",
        "--launch-skip",
        "2",
        "--launch-count",
        str(reps),
        "--metrics",
        ",".join(metrics),
        "--",
        *entry.cmd.split(),
    ]


@dataclass
class RawKernelRow:
    kernel: str
    metrics: dict[str, str] = field(default_factory=dict)


def parse_ncu_csv(csv_text: str) -> list[RawKernelRow]:
    """Parse ncu's long-format CSV (one row per kernel-launch x metric) into
    one RawKernelRow per launch, preserving file order.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    required = {"ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"}
    fieldnames = set(reader.fieldnames or [])
    missing = required - fieldnames
    if missing:
        raise CollectionError(f"ncu CSV missing expected columns: {sorted(missing)}")

    rows_by_id: dict[str, RawKernelRow] = {}
    order: list[str] = []
    for row in reader:
        launch_id = row["ID"]
        if launch_id not in rows_by_id:
            rows_by_id[launch_id] = RawKernelRow(kernel=row["Kernel Name"])
            order.append(launch_id)
        metric_name = row["Metric Name"]
        unit = (row.get("Metric Unit") or "").strip()
        value = row["Metric Value"]
        if metric_name == "gpu__time_duration.sum" and unit:
            scale = _TIME_UNIT_TO_NS.get(unit.lower())
            if scale is not None:
                value = str(float(value.replace(",", "")) * scale)
        rows_by_id[launch_id].metrics[metric_name] = value

    return [rows_by_id[i] for i in order]


def to_kernel_results(entry_name: str, rows: list[RawKernelRow]) -> list[KernelResultRecord]:
    """Group raw per-launch rows by kernel name and assign rep indices in
    file order (launch-skip already dropped warmup iterations upstream).
    """
    rep_counters: dict[str, int] = {}
    out: list[KernelResultRecord] = []
    for row in rows:
        rep = rep_counters.get(row.kernel, 0)
        rep_counters[row.kernel] = rep + 1

        floats: dict[str, float] = {}
        ints: dict[str, int] = {}
        for metric_name, value in row.metrics.items():
            field_name = _METRIC_FIELD_MAP.get(metric_name)
            if field_name is None:
                continue
            try:
                num = float(str(value).replace(",", ""))
            except ValueError:
                continue
            if field_name in _INT_FIELDS:
                ints[field_name] = int(num)
            else:
                floats[field_name] = num

        if "duration_ns" not in floats:
            continue  # can't build a record without the primary metric

        out.append(
            KernelResultRecord(
                entry=entry_name,
                kernel=row.kernel,
                rep=rep,
                duration_ns=floats.pop("duration_ns"),
                occupancy_pct=floats.get("occupancy_pct"),
                dram_pct=floats.get("dram_pct"),
                l2_hit_pct=floats.get("l2_hit_pct"),
                ipc=floats.get("ipc"),
                regs_per_thread=ints.get("regs_per_thread"),
                block_size=ints.get("block_size"),
                grid_size=ints.get("grid_size"),
                raw=dict(row.metrics),
            )
        )
    return out


def run_entry(
    entry: SuiteEntry,
    reps: int,
    metrics: tuple[str, ...] = METRIC_SET,
    ncu_path: str = "ncu",
) -> list[KernelResultRecord]:
    """Invoke ncu for one suite entry and return parsed per-rep kernel results."""
    if shutil.which(ncu_path) is None:
        raise CollectionError(
            f"'{ncu_path}' not found on PATH. Nsight Compute must be installed on the "
            "self-hosted GPU runner to collect real measurements."
        )

    cmd = build_ncu_command(entry, reps, metrics)
    cmd[0] = ncu_path
    proc = subprocess.run(
        cmd,
        cwd=entry.cwd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise CollectionError(
            f"ncu failed for entry '{entry.name}' (exit {proc.returncode}).\n"
            f"stderr:\n{proc.stderr}\n\n{PERMISSION_FIX_HINT}"
        )

    rows = parse_ncu_csv(proc.stdout)
    return to_kernel_results(entry.name, rows)
