"""Parse `nvcc -Xptxas -v` output into per-kernel resource usage."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ENTRY = re.compile(r"(?:Compiling entry function|Function properties for)\s+'?([^'\s]+)'?")
_FRAME = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
)
_USED = re.compile(r"Used (\d+) registers")
_SMEM = re.compile(r"(\d+) bytes smem")


@dataclass
class PtxasInfo:
    function: str | None = None
    registers: int | None = None
    stack_bytes: int = 0
    spill_store_bytes: int = 0
    spill_load_bytes: int = 0
    smem_bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def spill_bytes(self) -> int:
        return self.spill_store_bytes + self.spill_load_bytes


def parse(text: str, function_regex: str | None = None) -> PtxasInfo:
    """Resource usage for one entry function from a ptxas -v log.

    `function_regex` picks the function (searched in its mangled name, e.g.
    "bias_gelu_kernel"). Without it, the last entry function in the log wins.
    ptxas warnings (e.g. a -maxrregcount being raised to the architecture's
    floor) are returned for every function, since they apply to the build.
    """
    infos: dict[str, PtxasInfo] = {}
    order: list[str] = []
    warnings: list[str] = []
    current: PtxasInfo | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if "ptxas warning" in line:
            warning = line.split(":", 1)[-1].strip()
            if warning not in warnings:  # host and device passes repeat the same warning
                warnings.append(warning)
            continue
        m = _ENTRY.search(line)
        if m:
            name = m.group(1)
            if name not in infos:
                infos[name] = PtxasInfo(function=name)
                order.append(name)
            current = infos[name]
            continue
        if current is None:
            continue
        m = _FRAME.search(line)
        if m:
            current.stack_bytes, current.spill_store_bytes, current.spill_load_bytes = (
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
            )
            continue
        m = _USED.search(line)
        if m:
            current.registers = int(m.group(1))
            s = _SMEM.search(line)
            current.smem_bytes = int(s.group(1)) if s else 0

    chosen: PtxasInfo | None = None
    if function_regex is not None:
        pattern = re.compile(function_regex)
        for name in order:
            if pattern.search(name):
                chosen = infos[name]
    elif order:
        chosen = infos[order[-1]]

    result = chosen if chosen is not None else PtxasInfo()
    result.warnings = warnings
    return result
