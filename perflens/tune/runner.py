"""Runners turn one (kernel, batch, config) into a measured `EvalResult`.

`LocalGpuRunner` drives the gpu-engine harness on a real GPU box: build the
candidate (bench/build_candidate.sh), parse its ptxas log, then run the
resulting kernel_bench binary. `FakeRunner` is a deterministic synthetic
landscape so every strategy, the agent loop, storage, and reporting can be
tested without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from perflens.tune import ptxas
from perflens.tune.space import SearchSpace, TuneConfig


class EvalResult(BaseModel):
    config: TuneConfig
    build_ok: bool
    cached_build: bool = False
    compile_ms: float | None = None
    registers: int | None = None
    spill_bytes: int | None = None
    smem_bytes: int | None = None
    ptxas_warnings: list[str] = Field(default_factory=list)
    correct: bool = False
    max_abs_err: float | None = None
    median_us: float | None = None
    mad_us: float | None = None
    error: str | None = None

    @property
    def fitness(self) -> float:
        """Lower is better. Anything that failed to build or is incorrect is never best."""
        if self.build_ok and self.correct and self.median_us is not None:
            return self.median_us
        return math.inf

    def summary(self) -> dict[str, object]:
        """Compact view for the agent and for logs."""
        return {
            "config": self.config.model_dump(),
            "build_ok": self.build_ok,
            "compile_ms": self.compile_ms,
            "registers": self.registers,
            "spill_bytes": self.spill_bytes,
            "ptxas_warnings": self.ptxas_warnings,
            "correct": self.correct,
            "max_abs_err": self.max_abs_err,
            "median_us": self.median_us,
            "mad_us": self.mad_us,
            "error": self.error,
        }


class Runner(Protocol):
    def evaluate(self, space: SearchSpace, batch: int, cfg: TuneConfig) -> EvalResult: ...

    def ptxas_log(self, space: SearchSpace, cfg: TuneConfig) -> str: ...

    def kernel_source(self, space: SearchSpace) -> str: ...


class RunnerError(RuntimeError):
    pass


class LocalGpuRunner:
    """Runs candidates through gpu-engine/bench on this machine's GPU."""

    def __init__(
        self,
        engine_dir: str | Path,
        samples: int = 200,
        inner: int = 10,
        warmup: int = 20,
        timeout_s: int = 600,
    ) -> None:
        self.engine_dir = Path(engine_dir)
        self.samples = samples
        self.inner = inner
        self.warmup = warmup
        self.timeout_s = timeout_s
        if not (self.engine_dir / "bench" / "build_candidate.sh").exists():
            raise RunnerError(
                f"{self.engine_dir}/bench/build_candidate.sh not found -- point --engine-dir at a "
                "gpu-engine checkout that includes the tuning harness"
            )
        self._builds: dict[tuple[str, str], dict[str, object]] = {}

    def _build(self, space: SearchSpace, cfg: TuneConfig) -> dict[str, object]:
        key = (space.kernel, cfg.key())
        if key in self._builds:
            return self._builds[key]
        proc = subprocess.run(
            [
                "bash",
                "bench/build_candidate.sh",
                space.kernel,
                json.dumps(space.build_config(cfg)),
            ],
            cwd=self.engine_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        try:
            record: dict[str, object] = json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            record = {"ok": False, "error": f"build script gave no JSON (stderr: {proc.stderr[-300:]})"}
        self._builds[key] = record
        return record

    def binary_for(self, space: SearchSpace, cfg: TuneConfig) -> Path:
        """Build (or reuse) a candidate and return its kernel_bench binary."""
        record = self._build(space, cfg)
        if not record.get("ok"):
            raise RunnerError(f"build failed for {cfg.short()}: {record.get('error')}")
        return Path(str(record["binary"]))

    def ptxas_log(self, space: SearchSpace, cfg: TuneConfig) -> str:
        record = self._build(space, cfg)
        path = record.get("ptxas_log")
        return Path(str(path)).read_text() if path else str(record.get("error", ""))

    def kernel_source(self, space: SearchSpace) -> str:
        parts = []
        for rel in space.source_files:
            parts.append(f"// ===== {rel} =====\n{(self.engine_dir / rel).read_text()}")
        return "\n".join(parts)

    def evaluate(self, space: SearchSpace, batch: int, cfg: TuneConfig) -> EvalResult:
        record = self._build(space, cfg)
        if not record.get("ok"):
            return EvalResult(
                config=cfg,
                build_ok=False,
                compile_ms=_float_or_none(record.get("compile_ms")),
                error=str(record.get("error") or "build failed"),
            )
        info = ptxas.parse(self.ptxas_log(space, cfg), function_regex=space.kernel_regex)
        result = EvalResult(
            config=cfg,
            build_ok=True,
            cached_build=bool(record.get("cached")),
            compile_ms=_float_or_none(record.get("compile_ms")),
            registers=info.registers,
            spill_bytes=info.spill_bytes,
            smem_bytes=info.smem_bytes,
            ptxas_warnings=info.warnings,
        )
        proc = subprocess.run(
            [
                str(record["binary"]),
                "--batch",
                str(batch),
                "--samples",
                str(self.samples),
                "--inner",
                str(self.inner),
                "--warmup",
                str(self.warmup),
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        try:
            run = json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return result.model_copy(
                update={"error": f"kernel_bench gave no JSON (exit {proc.returncode}): {proc.stderr[-300:]}"}
            )
        return result.model_copy(
            update={
                "correct": bool(run.get("correct")),
                "max_abs_err": _float_or_none(run.get("max_abs_err")),
                "median_us": _float_or_none(run.get("median_us")),
                "mad_us": _float_or_none(run.get("mad_us")),
                "error": run.get("error"),
            }
        )


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Synthetic landscape for tests and dry runs. The numbers mean nothing about
# real hardware; they only give every strategy something with structure to
# search (an optimum away from the default, an incorrect region, noise).
# ---------------------------------------------------------------------------

_THREAD_EFF = {32: 0.55, 64: 0.75, 128: 0.92, 256: 1.0, 512: 0.97, 1024: 0.88}
FAKE_SM_COUNT = 28


class FakeRunner:
    def __init__(self, seed: int = 0, noise: float = 0.01, incorrect_fast_math: bool = False) -> None:
        self.seed = seed
        self.noise = noise
        self.incorrect_fast_math = incorrect_fast_math
        self.calls: list[tuple[str, int, str]] = []

    def _rng(self, *parts: object) -> random.Random:
        digest = hashlib.sha256(repr((self.seed, *parts)).encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def kernel_source(self, space: SearchSpace) -> str:
        return f"// synthetic source for {space.kernel}\n__global__ void {space.kernel}_kernel() {{}}\n"

    def ptxas_log(self, space: SearchSpace, cfg: TuneConfig) -> str:
        regs = 14 + (8 if cfg.vec == 2 else 0) - (1 if cfg.fast_math else 0)
        return (
            f"ptxas info    : Compiling entry function '_Z{space.kernel}_kernel' for 'sm_86'\n"
            f"ptxas info    : Function properties for _Z{space.kernel}_kernel\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            f"ptxas info    : Used {regs} registers, used 0 barriers, 380 bytes cmem[0]\n"
        )

    def runtime_us(self, space: SearchSpace, batch: int, cfg: TuneConfig) -> float:
        blocks = batch * cfg.blocks_per_row
        sm_util = min(1.0, blocks / FAKE_SM_COUNT)
        work_items = (3072 if space.kernel == "bias_gelu" else 768) / cfg.vec
        idle = max(0.0, cfg.threads * cfg.blocks_per_row - work_items) / (cfg.threads * cfg.blocks_per_row)
        us = 4.0 / (sm_util * _THREAD_EFF[cfg.threads])
        us *= 0.8 if cfg.vec == 2 else 1.0
        us *= 0.97 if cfg.fast_math else 1.0
        us += 2.0 * idle
        return us

    def evaluate(self, space: SearchSpace, batch: int, cfg: TuneConfig) -> EvalResult:
        self.calls.append((space.kernel, batch, cfg.key()))
        info = ptxas.parse(self.ptxas_log(space, cfg))
        rng = self._rng("noise", space.kernel, batch, cfg.key(), len(self.calls))
        jitter = 1.0 + rng.gauss(0, self.noise)
        correct = not (self.incorrect_fast_math and cfg.fast_math)
        return EvalResult(
            config=cfg,
            build_ok=True,
            compile_ms=3000.0 + 10 * cfg.threads / 32,
            registers=info.registers,
            spill_bytes=info.spill_bytes,
            smem_bytes=info.smem_bytes,
            correct=correct,
            max_abs_err=0.004 if correct else 0.07,
            median_us=round(self.runtime_us(space, batch, cfg) * jitter, 4),
            mad_us=0.02,
        )
