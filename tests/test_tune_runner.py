"""LocalGpuRunner against a stand-in gpu-engine dir (fake build script + fake kernel_bench)."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from perflens.tune.runner import FakeRunner, LocalGpuRunner, RunnerError
from perflens.tune.space import TuneConfig, get_space

FIXTURES = Path(__file__).parent / "fixtures"


def _make_engine(tmp_path: Path, bench_stdout: str, build_ok: bool = True) -> Path:
    engine = tmp_path / "gpu-engine"
    (engine / "bench").mkdir(parents=True)
    (engine / "src" / "kernels").mkdir(parents=True)
    (engine / "include" / "engine").mkdir(parents=True)
    (engine / "src" / "kernels" / "gelu.cu").write_text("__global__ void bias_gelu_kernel() {}\n")
    (engine / "include" / "engine" / "tune_params.h").write_text("#define GELU_THREADS 256\n")

    log = tmp_path / "ptxas.log"
    log.write_text((FIXTURES / "ptxas_bias_gelu_vec2.log").read_text())
    binary = tmp_path / "kernel_bench"
    binary.write_text(f"#!/usr/bin/env bash\necho 'noise on stdout'\necho '{bench_stdout}'\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)

    record = {
        "ok": build_ok,
        "compile_ms": 3456.7,
        "cached": False,
        "binary": str(binary),
        "ptxas_log": str(log),
        "error": None if build_ok else "error: static assertion failed",
    }
    script = engine / "bench" / "build_candidate.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> {tmp_path}/build_args.txt\n'
        f"echo '{json.dumps(record)}'\n"
    )
    return engine


def test_evaluate_combines_build_ptxas_and_bench(tmp_path: Path) -> None:
    bench = json.dumps({"correct": True, "max_abs_err": 0.0039, "median_us": 3.21, "mad_us": 0.02})
    runner = LocalGpuRunner(_make_engine(tmp_path, bench))
    space = get_space("bias_gelu")
    cfg = TuneConfig(threads=512, blocks_per_row=4, vec=2)

    result = runner.evaluate(space, 8, cfg)

    assert result.build_ok and result.correct
    assert (result.registers, result.spill_bytes, result.compile_ms) == (22, 0, 3456.7)
    assert result.median_us == 3.21 and result.fitness == 3.21
    args = (tmp_path / "build_args.txt").read_text().split(" ", 1)
    assert args[0] == "bias_gelu"
    assert json.loads(args[1]) == {"threads": 512, "fast_math": False, "blocks_per_row": 4, "vec": 2}

    runner.evaluate(space, 32, cfg)  # second batch size reuses the build
    assert len((tmp_path / "build_args.txt").read_text().splitlines()) == 1
    assert "gelu.cu" in runner.kernel_source(space)


def test_incorrect_result_is_never_fit(tmp_path: Path) -> None:
    bench = json.dumps({"correct": False, "max_abs_err": 0.09, "median_us": 1.0, "mad_us": 0.01})
    result = LocalGpuRunner(_make_engine(tmp_path, bench)).evaluate(
        get_space("bias_gelu"), 8, TuneConfig(threads=256)
    )
    assert result.correct is False
    assert result.fitness == float("inf")


def test_build_failure_skips_the_bench(tmp_path: Path) -> None:
    runner = LocalGpuRunner(_make_engine(tmp_path, "{}", build_ok=False))
    result = runner.evaluate(get_space("bias_gelu"), 8, TuneConfig(threads=256))
    assert result.build_ok is False
    assert result.error == "error: static assertion failed"
    with pytest.raises(RunnerError, match="build failed"):
        runner.binary_for(get_space("bias_gelu"), TuneConfig(threads=256))


def test_bench_without_json_is_reported(tmp_path: Path) -> None:
    result = LocalGpuRunner(_make_engine(tmp_path, "segfault")).evaluate(
        get_space("bias_gelu"), 8, TuneConfig(threads=256)
    )
    assert result.build_ok is True and result.correct is False
    assert result.error is not None and "no JSON" in result.error


def test_missing_harness_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(RunnerError, match="build_candidate.sh not found"):
        LocalGpuRunner(tmp_path)


def test_fake_runner_is_deterministic_per_seed() -> None:
    space, cfg = get_space("bias_gelu"), TuneConfig(threads=128, blocks_per_row=4)
    a = FakeRunner(seed=1).evaluate(space, 8, cfg)
    b = FakeRunner(seed=1).evaluate(space, 8, cfg)
    assert a == b
    assert FakeRunner(seed=1, incorrect_fast_math=True).evaluate(
        space, 8, cfg.model_copy(update={"fast_math": True})
    ).correct is False
