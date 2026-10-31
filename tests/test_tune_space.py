"""Search spaces: membership, validation of agent proposals, and the mutation operator."""

from __future__ import annotations

import random

import pytest

from perflens.tune.space import SPACES, SpaceError, TuneConfig, get_space


def test_space_sizes_and_defaults_match_shipped_launch_config() -> None:
    assert {k: s.size for k, s in SPACES.items()} == {"bias_gelu": 120, "add_layernorm": 12, "argmax": 12}
    for space in SPACES.values():
        assert space.default == TuneConfig(threads=256)
        assert space.default.key() in {c.key() for c in space.all()}


def test_all_enumerates_each_config_exactly_once() -> None:
    for space in SPACES.values():
        keys = [c.key() for c in space.all()]
        assert len(keys) == len(set(keys)) == space.size


def test_validate_accepts_a_full_in_space_config() -> None:
    cfg = get_space("bias_gelu").validate(
        {"threads": 512, "blocks_per_row": 4, "vec": 2, "fast_math": True}
    )
    assert cfg == TuneConfig(threads=512, blocks_per_row=4, vec=2, fast_math=True)


@pytest.mark.parametrize(
    "raw, reason",
    [
        ({"threads": 96, "blocks_per_row": 1, "vec": 1, "fast_math": False}, "not in allowed"),
        ({"threads": 256, "blocks_per_row": 1, "vec": 1}, "missing"),
        ({"threads": 256, "blocks_per_row": 1, "vec": 1, "fast_math": False, "unroll": 2}, "unknown"),
        ({"threads": 256, "blocks_per_row": 1, "vec": 1, "fast_math": 1}, "not in allowed"),
        ({"threads": 256, "blocks_per_row": True, "vec": 1, "fast_math": False}, "not in allowed"),
        ({"threads": 256, "blocks_per_row": 3, "vec": 1, "fast_math": False}, "not in allowed"),
    ],
)
def test_validate_rejects_out_of_space_proposals(raw: dict[str, object], reason: str) -> None:
    with pytest.raises(SpaceError, match=reason):
        get_space("bias_gelu").validate(raw)


def test_validate_rejects_dimensions_a_kernel_does_not_have() -> None:
    with pytest.raises(SpaceError, match="unknown"):
        get_space("argmax").validate({"threads": 256, "fast_math": False, "vec": 2})
    with pytest.raises(SpaceError, match="must be an object"):
        get_space("argmax").validate([256])  # type: ignore[arg-type]


def test_build_config_only_sends_dimensions_the_kernel_has() -> None:
    cfg = TuneConfig(threads=128, blocks_per_row=8, vec=2, fast_math=True)
    assert get_space("bias_gelu").build_config(cfg) == {
        "threads": 128,
        "fast_math": True,
        "blocks_per_row": 8,
        "vec": 2,
    }
    assert get_space("argmax").build_config(TuneConfig(threads=64)) == {"threads": 64, "fast_math": False}


def test_mutate_changes_exactly_one_dimension_and_stays_in_space() -> None:
    space = get_space("bias_gelu")
    rng = random.Random(0)
    cfg = space.default
    for _ in range(300):
        child = space.mutate(cfg, rng)
        diffs = [n for n in space.dims if getattr(child, n) != getattr(cfg, n)]
        assert len(diffs) == 1
        space.validate(child.model_dump())
        cfg = child


def test_sample_unseen_returns_none_once_everything_is_seen() -> None:
    space = get_space("argmax")
    rng = random.Random(0)
    seen = {c.key() for c in space.all()}
    assert space.sample_unseen(rng, seen) is None
    seen.discard(space.default.key())
    assert space.sample_unseen(rng, seen) == space.default


def test_config_schema_enumerates_allowed_values() -> None:
    schema = get_space("bias_gelu").config_schema()
    assert schema["properties"]["blocks_per_row"] == {"enum": [1, 2, 4, 8, 12]}
    assert schema["required"] == ["threads", "blocks_per_row", "vec", "fast_math"]
    assert schema["additionalProperties"] is False


def test_unknown_kernel() -> None:
    with pytest.raises(SpaceError, match="unknown kernel"):
        get_space("softmax")


def test_short_names() -> None:
    assert TuneConfig(threads=256).short() == "t256"
    assert TuneConfig(threads=64, blocks_per_row=4, vec=2, fast_math=True).short() == "t64/bpr4/vec2/fastmath"
