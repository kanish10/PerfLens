"""Summarize tuning runs into results/tune/summary.csv and REPORT.md.

Ground truth per (kernel, batch) comes from the latest exhaustive run. To keep
timing noise from deciding the comparison, every strategy's choices are
scored on the exhaustive run's measurement of the same config ("canonical"
runtime). Each strategy still made its decisions from its own measurements.

- evals_to_5pct: how many evaluations (counting the default) a run needed
  before it had evaluated a config within 5% of the exhaustive best
- regret_pct: how far the run's final pick is from the exhaustive best
- compile_s_total: nvcc time the run's evaluations cost (original build
  times, even when a build was served from cache)
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from perflens.tune import db
from perflens.tune.space import TuneConfig

NEAR_BEST = 1.05


@dataclass
class GroundTruth:
    run_id: int
    n_configs: int
    canonical_us: dict[str, float]
    best_key: str | None
    best_us: float | None
    default_us: float | None
    rejected: list[dict[str, object]] = field(default_factory=list)
    registers: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class RunSummary:
    run_id: int
    kernel: str
    batch: int
    strategy: str
    seed: int
    budget: int
    evals: int
    invalid_proposals: int
    stop_reason: str | None
    default_us: float | None
    best_config: str | None
    best_us_own: float | None
    best_us_canonical: float | None
    exhaustive_best_us: float | None
    regret_pct: float | None
    evals_to_5pct: int | None
    compile_s_total: float
    model: str | None
    gpu: str
    driver: str
    engine_sha: str | None


def _short(config_json: str) -> str:
    return TuneConfig(**json.loads(config_json)).short()


def _fitness(row: sqlite3.Row) -> float:
    if row["build_ok"] and row["correct"] and row["median_us"] is not None:
        return float(row["median_us"])
    return math.inf


def _ground_truth(run: sqlite3.Row, evals: list[sqlite3.Row]) -> GroundTruth:
    canonical = {e["config_key"]: float(e["median_us"]) for e in evals if math.isfinite(_fitness(e))}
    best_key = min(canonical, key=lambda k: canonical[k]) if canonical else None
    rejected = [
        {
            "config": _short(e["config_json"]),
            "build_ok": bool(e["build_ok"]),
            "max_abs_err": e["max_abs_err"],
            "error": e["error"],
        }
        for e in evals
        if not math.isfinite(_fitness(e))
    ]
    registers: dict[str, list[int]] = defaultdict(list)
    for e in evals:
        if e["registers"] is not None:
            cfg = json.loads(e["config_json"])
            variant = f"vec={cfg.get('vec', 1)}, fast_math={cfg.get('fast_math', False)}"
            registers[variant].append(int(e["registers"]))
    return GroundTruth(
        run_id=run["id"],
        n_configs=len(evals),
        canonical_us=canonical,
        best_key=best_key,
        best_us=canonical[best_key] if best_key else None,
        default_us=_fitness(evals[0]) if evals and math.isfinite(_fitness(evals[0])) else None,
        rejected=rejected,
        registers={k: sorted(set(v)) for k, v in registers.items()},
    )


def summarize(conn: sqlite3.Connection) -> tuple[list[RunSummary], dict[tuple[str, int], GroundTruth]]:
    runs = db.list_runs(conn)
    evals_by_run = {r["id"]: db.get_evals(conn, r["id"]) for r in runs}

    truth: dict[tuple[str, int], GroundTruth] = {}
    for r in runs:  # ordered by id, so the latest exhaustive run wins
        if r["strategy"] == "exhaustive":
            truth[(r["kernel"], r["batch"])] = _ground_truth(r, evals_by_run[r["id"]])

    summaries: list[RunSummary] = []
    for r in runs:
        evals = evals_by_run[r["id"]]
        gt = truth.get((r["kernel"], r["batch"]))
        scored = [e for e in evals if math.isfinite(_fitness(e))]
        best = min(scored, key=_fitness) if scored else None
        best_canon = None
        regret = None
        to_5pct = None
        if gt is not None and gt.best_us is not None:
            if best is not None:
                best_canon = gt.canonical_us.get(best["config_key"], _fitness(best))
                regret = 100.0 * (best_canon - gt.best_us) / gt.best_us
            if r["strategy"] != "exhaustive":
                for i, e in enumerate(evals, start=1):
                    canon = gt.canonical_us.get(e["config_key"])
                    if canon is not None and canon <= NEAR_BEST * gt.best_us:
                        to_5pct = i
                        break
        summaries.append(
            RunSummary(
                run_id=r["id"],
                kernel=r["kernel"],
                batch=r["batch"],
                strategy=r["strategy"],
                seed=r["seed"],
                budget=r["budget"],
                evals=len(evals),
                invalid_proposals=r["invalid_proposals"],
                stop_reason=r["stop_reason"],
                default_us=_fitness(evals[0]) if evals and math.isfinite(_fitness(evals[0])) else None,
                best_config=_short(best["config_json"]) if best is not None else None,
                best_us_own=_fitness(best) if best is not None else None,
                best_us_canonical=best_canon,
                exhaustive_best_us=gt.best_us if gt else None,
                regret_pct=regret,
                evals_to_5pct=to_5pct,
                compile_s_total=sum(float(e["compile_ms"] or 0.0) for e in evals) / 1000.0,
                model=r["model"],
                gpu=r["gpu"],
                driver=r["driver"],
                engine_sha=r["engine_sha"],
            )
        )
    return summaries, truth


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def write_csv(summaries: list[RunSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(s) for s in summaries]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RunSummary.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(
    conn: sqlite3.Connection,
    summaries: list[RunSummary],
    truth: dict[tuple[str, int], GroundTruth],
    db_path: str,
) -> str:
    out: list[str] = ["# PerfLens Tune results", ""]
    if not summaries:
        return "\n".join([*out, "No finished tuning runs in the database yet."]) + "\n"

    if any(s.gpu == "fake" for s in summaries):
        out += [
            "> **SYNTHETIC DATA.** Some runs in this database used `--runner fake`, a made-up "
            "landscape for testing. None of these numbers are GPU measurements; do not cite them.",
            "",
        ]
    gpus = sorted({f"{s.gpu} (driver {s.driver})" for s in summaries})
    shas = sorted({s.engine_sha or "unknown" for s in summaries})
    models = sorted({s.model for s in summaries if s.strategy == "agent" and s.model})
    out += [
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by `perflens tune report` "
        f"from `{db_path}`.",
        "",
        f"- GPU: {', '.join(gpus)}",
        f"- gpu-engine commit: {', '.join(s[:12] for s in shas)}",
        f"- Agent model: {', '.join(models) if models else 'n/a'}",
        "- Runtimes are per-launch medians from kernel_bench (cudaEvent timing, 200 samples of 10 "
        "launches). Strategy picks are scored on the exhaustive run's measurement of the same config.",
        "",
    ]

    groups = sorted({(s.kernel, s.batch) for s in summaries})
    ledger: list[tuple[str, str, str]] = []
    for kernel, batch in groups:
        gt = truth.get((kernel, batch))
        group = [s for s in summaries if (s.kernel, s.batch) == (kernel, batch)]
        out += [f"## {kernel}, batch {batch}", ""]
        if gt is None:
            out += ["No exhaustive run for this kernel and batch, so there is no ground truth.", ""]
        elif gt.best_us is None or gt.best_key is None:
            out += [
                f"Exhaustive run #{gt.run_id} found no configuration that built and passed the "
                f"correctness check ({len(gt.rejected)} of {gt.n_configs} rejected), so there is "
                "no ground truth.",
                "",
            ]
            for rej in gt.rejected[:3]:
                out.append(f"- `{rej['config']}`: build_ok={rej['build_ok']}, error={rej['error']}")
            out.append("")
        else:
            best_cfg = TuneConfig(**json.loads(gt.best_key)).short()
            speedup = gt.default_us / gt.best_us if gt.default_us else None
            out += [
                f"Exhaustive search (run #{gt.run_id}, {gt.n_configs} configs): default "
                f"{_fmt(gt.default_us, 3)} us, best `{best_cfg}` {_fmt(gt.best_us, 3)} us "
                f"({_fmt(speedup, 2, 'x')} vs. default).",
                "",
            ]
            if speedup is not None:
                ledger.append(
                    (
                        f"{kernel} b{batch}: best config vs. default",
                        f"{speedup:.2f}x ({best_cfg})",
                        f"summary.csv run_id={gt.run_id}: default_us / best_us_own",
                    )
                )
            if gt.rejected:
                out.append(f"Rejected by the correctness gate or the build: {len(gt.rejected)} configs.")
                for rej in gt.rejected[:6]:
                    out.append(
                        f"- `{rej['config']}`: build_ok={rej['build_ok']}, "
                        f"max_abs_err={rej['max_abs_err']}, error={rej['error']}"
                    )
                out.append("")
            if gt.registers:
                regs = "; ".join(f"{k}: {v}" for k, v in sorted(gt.registers.items()))
                out += [f"ptxas registers/thread by variant: {regs}.", ""]

        if not any(s.strategy != "exhaustive" for s in group):
            continue
        out += [
            "| strategy | runs | evals to within 5% of best: median [min-max] | runs that got there "
            "| final regret (median) | invalid proposals | compile time per run (median) |",
            "|---|---|---|---|---|---|---|",
        ]
        for strategy in ("random", "evo", "agent"):
            runs = [s for s in group if s.strategy == strategy]
            if not runs:
                continue
            reached = [s.evals_to_5pct for s in runs if s.evals_to_5pct is not None]
            regrets = [s.regret_pct for s in runs if s.regret_pct is not None]
            med_reach = _median([float(x) for x in reached])
            spread = f" [{min(reached)}-{max(reached)}]" if reached else ""
            out.append(
                f"| {strategy} | {len(runs)} | {_fmt(med_reach, 1)}{spread} | {len(reached)}/{len(runs)} "
                f"| {_fmt(_median(regrets), 1, '%')} | {sum(s.invalid_proposals for s in runs)} "
                f"| {_fmt(_median([s.compile_s_total for s in runs]), 0, ' s')} |"
            )
            if med_reach is not None:
                ledger.append(
                    (
                        f"{kernel} b{batch}: {strategy} evals to within 5% (median of {len(reached)} runs)",
                        f"{med_reach:.1f} (budget {runs[0].budget})",
                        f"summary.csv strategy={strategy},kernel={kernel},batch={batch}: "
                        "median evals_to_5pct",
                    )
                )
        out.append("")

        agent_runs = sorted((s for s in group if s.strategy == "agent"), key=lambda s: s.seed)
        if agent_runs:
            first = agent_runs[0]
            rationales = [
                (e["idx"], _short(e["config_json"]), e["rationale"])
                for e in db.get_evals(conn, first.run_id)
                if e["idx"] > 0 and e["rationale"]
            ][:3]
            if rationales:
                out += [f"Agent rationales (run #{first.run_id}, seed {first.seed}), verbatim:", ""]
                out += [f"- #{i} `{cfg}`: {text}" for i, cfg, text in rationales]
                out.append("")

    out += ["## Claims ledger", "", "| Claim | Value | Source |", "|---|---|---|"]
    out += [f"| {c} | {v} | {src} |" for c, v, src in ledger]
    return "\n".join(out) + "\n"


def write_report(conn: sqlite3.Connection, out_dir: Path, db_path: str) -> tuple[Path, Path]:
    summaries, truth = summarize(conn)
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "REPORT.md"
    write_csv(summaries, csv_path)
    md_path.write_text(render_markdown(conn, summaries, truth, db_path))
    return csv_path, md_path
