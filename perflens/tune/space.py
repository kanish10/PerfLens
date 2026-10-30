"""Search spaces: which compile/launch settings each kernel can be tuned over.

A `TuneConfig` is one point in a kernel's space. The space for each kernel
lists every dimension and its allowed values; anything outside those values
is rejected by `SearchSpace.validate`, which is also how the agent's
proposals are checked.

`-maxrregcount` and `__launch_bounds__` are supported by the gpu-engine
harness but left out of these default spaces: under nvcc 12.8 / sm_86 every
kernel here uses at most 22 registers, and ptxas raises any -maxrregcount
below 24 to 24, so neither knob can change the compiled kernel (see
gpu-engine/bench/TUNING.md).
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


class SpaceError(ValueError):
    """A config that is not a point in the kernel's search space."""


class TuneConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    threads: int
    blocks_per_row: int = 1
    vec: int = 1
    fast_math: bool = False

    def key(self) -> str:
        """Canonical, order-independent identity used for de-duplication."""
        return json.dumps(self.model_dump(), sort_keys=True)

    def short(self) -> str:
        parts = [f"t{self.threads}"]
        if self.blocks_per_row != 1:
            parts.append(f"bpr{self.blocks_per_row}")
        if self.vec != 1:
            parts.append(f"vec{self.vec}")
        if self.fast_math:
            parts.append("fastmath")
        return "/".join(parts)


THREADS = (32, 64, 128, 256, 512, 1024)


@dataclass(frozen=True)
class SearchSpace:
    kernel: str
    dims: dict[str, tuple[Any, ...]]
    default: TuneConfig
    kernel_regex: str  # Nsight Compute kernel-name filter
    source_files: tuple[str, ...]  # relative to the gpu-engine dir; shown to the agent

    @property
    def size(self) -> int:
        n = 1
        for values in self.dims.values():
            n *= len(values)
        return n

    def all(self) -> list[TuneConfig]:
        names = list(self.dims)
        return [
            TuneConfig(**dict(zip(names, combo, strict=True)))
            for combo in itertools.product(*(self.dims[n] for n in names))
        ]

    def validate(self, raw: dict[str, Any]) -> TuneConfig:
        """Parse a raw dict (e.g. an agent proposal) into a config in this space.

        Every dimension must be given, with one of its allowed values. Booleans
        and ints are not interchangeable (``True`` is not ``1`` here).
        """
        if not isinstance(raw, dict):
            raise SpaceError(f"config must be an object, got {type(raw).__name__}")
        unknown = sorted(set(raw) - set(self.dims))
        if unknown:
            raise SpaceError(f"unknown dimension(s) for {self.kernel}: {unknown}")
        missing = sorted(set(self.dims) - set(raw))
        if missing:
            raise SpaceError(f"missing dimension(s): {missing}")
        for name, allowed in self.dims.items():
            value = raw[name]
            want_bool = isinstance(allowed[0], bool)
            if isinstance(value, bool) != want_bool or value not in allowed:
                raise SpaceError(f"{name}={value!r} not in allowed values {list(allowed)}")
        return TuneConfig(**raw)

    def build_config(self, cfg: TuneConfig) -> dict[str, Any]:
        """The JSON the gpu-engine build script takes for this kernel."""
        out: dict[str, Any] = {"threads": cfg.threads, "fast_math": cfg.fast_math}
        if "blocks_per_row" in self.dims:
            out["blocks_per_row"] = cfg.blocks_per_row
        if "vec" in self.dims:
            out["vec"] = cfg.vec
        return out

    def sample_unseen(self, rng: random.Random, seen: set[str]) -> TuneConfig | None:
        remaining = [c for c in self.all() if c.key() not in seen]
        return rng.choice(remaining) if remaining else None

    def mutate(self, cfg: TuneConfig, rng: random.Random) -> TuneConfig:
        """Change exactly one dimension (with more than one value) to a different allowed value."""
        mutable = [n for n, v in self.dims.items() if len(v) > 1]
        name = rng.choice(mutable)
        current = getattr(cfg, name)
        choices = [v for v in self.dims[name] if v != current]
        return cfg.model_copy(update={name: rng.choice(choices)})

    def describe(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "dimensions": {n: list(v) for n, v in self.dims.items()},
            "default": self.default.model_dump(include=set(self.dims)),
            "size": self.size,
        }

    def config_schema(self) -> dict[str, Any]:
        """JSON schema for one config in this space (used in the agent's tool)."""
        return {
            "type": "object",
            "properties": {n: {"enum": list(v)} for n, v in self.dims.items()},
            "required": list(self.dims),
            "additionalProperties": False,
        }


SPACES: dict[str, SearchSpace] = {
    "bias_gelu": SearchSpace(
        kernel="bias_gelu",
        dims={
            "threads": THREADS,
            "blocks_per_row": (1, 2, 4, 8, 12),
            "vec": (1, 2),
            "fast_math": (False, True),
        },
        default=TuneConfig(threads=256),
        kernel_regex="bias_gelu_kernel.*",
        source_files=("src/kernels/gelu.cu", "include/engine/tune_params.h"),
    ),
    "add_layernorm": SearchSpace(
        kernel="add_layernorm",
        dims={"threads": THREADS, "fast_math": (False, True)},
        default=TuneConfig(threads=256),
        kernel_regex="add_layernorm_kernel.*",
        source_files=("src/kernels/layernorm.cu", "include/engine/tune_params.h"),
    ),
    "argmax": SearchSpace(
        kernel="argmax",
        dims={"threads": THREADS, "fast_math": (False, True)},
        default=TuneConfig(threads=256),
        kernel_regex="argmax_kernel.*",
        source_files=("src/kernels/logits.cu", "include/engine/tune_params.h"),
    ),
}


def get_space(kernel: str) -> SearchSpace:
    try:
        return SPACES[kernel]
    except KeyError:
        raise SpaceError(f"unknown kernel {kernel!r}; expected one of {sorted(SPACES)}") from None
