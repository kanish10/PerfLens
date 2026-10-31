"""ptxas -v parsing, against real nvcc 12.8 (sm_86) output captured from gpu-engine builds."""

from __future__ import annotations

from pathlib import Path

from perflens.tune import ptxas

FIXTURES = Path(__file__).parent / "fixtures"


def test_real_gelu_vec2_build() -> None:
    info = ptxas.parse((FIXTURES / "ptxas_bias_gelu_vec2.log").read_text(), "bias_gelu_kernel")
    assert info.function is not None and "bias_gelu_kernel" in info.function
    assert info.registers == 22
    assert info.spill_bytes == 0
    assert info.stack_bytes == 0
    assert info.warnings == []


def test_real_clamped_maxrregcount_warning_is_surfaced_once() -> None:
    info = ptxas.parse((FIXTURES / "ptxas_bias_gelu_maxrreg_clamped.log").read_text(), "bias_gelu_kernel")
    assert info.registers == 13
    assert info.warnings == [
        "For profile sm_86 adjusting per thread register count of 16 to lower bound of 24"
    ]


def test_real_two_kernel_log_selects_by_regex() -> None:
    text = (FIXTURES / "ptxas_attention_demo_regression.log").read_text()
    assert ptxas.parse(text, "paged_attention_kernel").registers == 155  # the PerfLens case study
    assert ptxas.parse(text, "kv_append_kernel").registers == 22
    assert ptxas.parse(text).registers == 22  # no regex: last entry function


def test_spills_and_static_smem() -> None:
    text = (
        "ptxas info    : Compiling entry function '_Z3fooPf' for 'sm_86'\n"
        "ptxas info    : Function properties for _Z3fooPf\n"
        "    48 bytes stack frame, 40 bytes spill stores, 36 bytes spill loads\n"
        "ptxas info    : Used 24 registers, used 1 barriers, 4096 bytes smem, 360 bytes cmem[0]\n"
    )
    info = ptxas.parse(text, "foo")
    assert (info.registers, info.stack_bytes, info.spill_bytes, info.smem_bytes) == (24, 48, 76, 4096)


def test_unmatched_or_empty_log() -> None:
    assert ptxas.parse("", "anything").registers is None
    text = (FIXTURES / "ptxas_bias_gelu_vec2.log").read_text()
    assert ptxas.parse(text, "argmax_kernel").registers is None
