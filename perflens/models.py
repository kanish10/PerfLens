"""Shared pydantic data models used across the collect -> detect -> triage -> report pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["minor", "major"]
FindingKind = Literal["regression", "improvement"]
Confidence = Literal["low", "medium", "high"]


class SuiteEntry(BaseModel):
    """One benchmark entry from suite.yaml."""

    name: str
    cwd: str
    cmd: str
    kernel_regex: str
    source_paths: list[str] = Field(default_factory=list)


class Suite(BaseModel):
    gpu: str
    reps: int = 5
    entries: list[SuiteEntry]


class KernelResultRecord(BaseModel):
    """One (entry, kernel, rep) measurement parsed from an ncu CSV."""

    entry: str
    kernel: str
    rep: int
    duration_ns: float
    occupancy_pct: float | None = None
    dram_pct: float | None = None
    l2_hit_pct: float | None = None
    regs_per_thread: int | None = None
    block_size: int | None = None
    grid_size: int | None = None
    ipc: float | None = None
    # HEAD of the entry's own source repo (SuiteEntry.cwd) at collection time --
    # NOT this repo's git_sha. Different entries can live in different sibling
    # repos (see suite.yaml), each with its own independent commit history.
    source_sha: str | None = None
    raw: dict[str, str] = Field(default_factory=dict)


class RunInfo(BaseModel):
    id: int | None = None
    ts: str
    git_sha: str
    branch: str
    gpu: str
    driver: str
    cuda_version: str
    suite_hash: str


class Finding(BaseModel):
    """Detector output for one (entry, kernel) whose primary metric moved significantly."""

    entry: str
    kernel: str
    metric: str = "duration_ns"
    base_median: float
    new_median: float
    delta_pct: float
    # A metric that legitimately baselines at 0 (e.g. l2_hit_pct) makes a
    # percent delta undefined rather than 0 -- None means "undefined", not
    # "unchanged". See detect._pct_delta.
    evidence: dict[str, float | None] = Field(default_factory=dict)
    severity: Severity | None = None
    kind: FindingKind
    suggested_command: str | None = None


class FindingTriage(BaseModel):
    """Agent-produced root-cause analysis for a single Finding."""

    entry: str
    kernel: str
    severity: Severity
    evidence_summary: str
    root_cause_hypothesis: str
    confidence: Confidence
    next_diagnostic_step: str


class TriageReport(BaseModel):
    findings: list[FindingTriage] = Field(default_factory=list)
    model: str
    generated_at: str
