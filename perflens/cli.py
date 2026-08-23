"""perflens CLI: run | baseline set | detect | triage | report | gate | dashboard."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import click
import yaml

from perflens import collect, detect, report, store, triage
from perflens.models import Finding, RunInfo, Suite, TriageReport

DEFAULT_DB = "perflens.db"
DEFAULT_SUITE = "suite.yaml"


def _load_suite(path: str) -> Suite:
    data = yaml.safe_load(Path(path).read_text())
    return Suite(**data)


def _suite_hash(path: str) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(["git", *args], capture_output=True, text=True)
    except FileNotFoundError:
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _nvidia_smi_field(query: str) -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines[0] if lines else "unknown"


def _cuda_version() -> str:
    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    except FileNotFoundError:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    m = re.search(r"CUDA Version:\s*([\d.]+)", proc.stdout)
    return m.group(1) if m else "unknown"


def _print_findings_table(findings: list[Finding]) -> None:
    if not findings:
        click.echo("no findings against baseline")
        return
    for f in findings:
        click.echo(
            f"[{report.severity_label(f)}] {f.entry}/{f.kernel} {f.metric}: "
            f"{f.base_median:.0f} -> {f.new_median:.0f} ({f.delta_pct:+.2f}%)"
        )


def _load_findings(conn, run_id: int, findings_json: str | None) -> list[Finding]:
    if findings_json:
        return [Finding(**d) for d in json.loads(Path(findings_json).read_text())]
    return detect.detect_regressions(conn, run_id)


@click.group()
@click.version_option(package_name="perflens")
def main() -> None:
    """PerfLens -- CUDA kernel performance-regression triage agent."""


@main.command("run")
@click.option("--suite", default=DEFAULT_SUITE, show_default=True)
@click.option("--db", default=DEFAULT_DB, show_default=True)
@click.option("--branch", default=None, help="Override detected git branch")
@click.option("--git-sha", default=None, help="Override detected git sha")
@click.option("--entry", "entry_names", multiple=True, help="Restrict to specific suite entries")
@click.option("--ncu-path", default="ncu", show_default=True)
@click.option(
    "--print-run-id",
    is_flag=True,
    help="Print only the new run id to stdout (for CI: run_id=$(perflens run ... --print-run-id))",
)
def run_cmd(
    suite: str,
    db: str,
    branch: str | None,
    git_sha: str | None,
    entry_names: tuple[str, ...],
    ncu_path: str,
    print_run_id: bool,
) -> None:
    """Collect the kernel suite (or a subset) into SQLite via Nsight Compute."""
    s = _load_suite(suite)
    entries = [e for e in s.entries if not entry_names or e.name in entry_names]
    if not entries:
        raise click.ClickException("no matching suite entries")

    conn = store.get_connection(db)
    store.init_db(conn)

    run_info = RunInfo(
        ts=datetime.now(UTC).isoformat(),
        git_sha=git_sha or _git("rev-parse", "HEAD"),
        branch=branch or _git("rev-parse", "--abbrev-ref", "HEAD"),
        gpu=s.gpu or _nvidia_smi_field("name"),
        driver=_nvidia_smi_field("driver_version"),
        cuda_version=_cuda_version(),
        suite_hash=_suite_hash(suite),
    )
    run_id = store.insert_run(conn, run_info)

    total = 0
    for e in entries:
        if not print_run_id:
            click.echo(f"collecting {e.name} ({s.reps} reps)...")
        records = collect.run_entry(e, s.reps, ncu_path=ncu_path)
        total += store.insert_kernel_results(conn, run_id, records)

    if print_run_id:
        click.echo(str(run_id))
    else:
        click.echo(f"run {run_id}: stored {total} kernel_results rows")


@main.group("baseline")
def baseline_group() -> None:
    """Manage regression baselines."""


@baseline_group.command("set")
@click.option("--run", "run_id", required=True, type=int)
@click.option("--db", default=DEFAULT_DB, show_default=True)
@click.option(
    "--entry",
    "entry_names",
    multiple=True,
    help="Restrict to specific suite entries (default: every entry in the run)",
)
def baseline_set(run_id: int, db: str, entry_names: tuple[str, ...]) -> None:
    """Write medians + MADs for every metric of every (entry, kernel) in a run."""
    conn = store.get_connection(db)
    store.init_db(conn)
    run = store.get_run(conn, run_id)
    if run is None:
        raise click.ClickException(f"no such run: {run_id}")

    entry_kernels = store.list_entry_kernels(conn, run_id)
    if entry_names:
        entry_kernels = [(e, k) for e, k in entry_kernels if e in entry_names]

    now = datetime.now(UTC).isoformat()
    count = 0
    for entry, kernel in entry_kernels:
        for metric, (median, mad) in store.compute_medians(conn, run_id, entry, kernel).items():
            store.set_baseline(conn, entry, kernel, metric, median, mad, run_id, now)
            count += 1
    click.echo(f"baseline set: {count} (entry, kernel, metric) row(s) from run {run_id}")


@main.command("detect")
@click.option("--run", "run_id", required=True, type=int)
@click.option("--db", default=DEFAULT_DB, show_default=True)
@click.option("--threshold", default=detect.DEFAULT_THRESHOLD, show_default=True, type=float)
@click.option("--json-out", default=None, type=click.Path())
def detect_cmd(run_id: int, db: str, threshold: float, json_out: str | None) -> None:
    """Print (and optionally save) findings for a run against its baseline."""
    conn = store.get_connection(db)
    store.init_db(conn)
    findings = detect.detect_regressions(conn, run_id, threshold=threshold)
    _print_findings_table(findings)
    if json_out:
        Path(json_out).write_text(json.dumps([f.model_dump() for f in findings], indent=2))
        click.echo(f"wrote {json_out}")


@main.command("triage")
@click.option("--run", "run_id", required=True, type=int)
@click.option("--db", default=DEFAULT_DB, show_default=True)
@click.option("--suite", default=DEFAULT_SUITE, show_default=True)
@click.option("--findings-json", default=None, type=click.Path(exists=True))
@click.option(
    "--base-sha",
    default=None,
    help="Override: normally each entry's source commits are resolved per-entry "
    "from recorded history (see TriageContext.resolve_shas); only set this "
    "together with --head-sha to force one pair for every entry",
)
@click.option("--head-sha", default=None, help="See --base-sha")
@click.option(
    "--model",
    default=triage.DEFAULT_MODEL,
    show_default=True,
    help="A 'claude-*' model routes to Anthropic (needs ANTHROPIC_API_KEY); any other "
    "model name routes to Groq's OpenAI-compatible API via groq_adapter (needs "
    "GROQ_API_KEY and the 'groq' extra installed).",
)
@click.option("--out", default=None, type=click.Path())
def triage_cmd(
    run_id: int,
    db: str,
    suite: str,
    findings_json: str | None,
    base_sha: str | None,
    head_sha: str | None,
    model: str,
    out: str | None,
) -> None:
    """Run the triage agent over a run's findings. Requires ANTHROPIC_API_KEY (default
    model) or GROQ_API_KEY (--model <a non-claude-* model>)."""
    conn = store.get_connection(db)
    store.init_db(conn)
    s = _load_suite(suite)
    entries = {e.name: e for e in s.entries}
    findings = _load_findings(conn, run_id, findings_json)

    ctx = triage.TriageContext(
        conn=conn,
        entries=entries,
        findings=findings,
        run_id=run_id,
        base_sha=base_sha,
        head_sha=head_sha,
    )
    try:
        result = triage.run_triage(ctx, model=model)
    except triage.TriageError as exc:
        raise click.ClickException(str(exc)) from exc

    out_path = out or "reports/triage.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(result.model_dump_json(indent=2))
    click.echo(f"wrote {out_path} ({len(result.findings)} triaged finding(s))")


@main.command("report")
@click.option("--run", "run_id", required=True, type=int)
@click.option("--db", default=DEFAULT_DB, show_default=True)
@click.option("--findings-json", default=None, type=click.Path(exists=True))
@click.option("--triage-json", default=None, type=click.Path(exists=True))
@click.option("--nightly", is_flag=True)
@click.option("--pr", "pr_number", default=None, type=int)
@click.option("--out", default=None, type=click.Path())
def report_cmd(
    run_id: int,
    db: str,
    findings_json: str | None,
    triage_json: str | None,
    nightly: bool,
    pr_number: int | None,
    out: str | None,
) -> None:
    """Render a Markdown report; commit it nightly, or post it as a PR comment."""
    conn = store.get_connection(db)
    store.init_db(conn)
    run = store.get_run(conn, run_id)
    if run is None:
        raise click.ClickException(f"no such run: {run_id}")

    findings = _load_findings(conn, run_id, findings_json)
    triage_report = (
        TriageReport(**json.loads(Path(triage_json).read_text())) if triage_json else None
    )

    md = report.render_markdown(dict(run), findings, triage_report)

    if out:
        out_path = out
    elif nightly:
        out_path = f"reports/{run['ts'][:10]}.md"
    else:
        out_path = "reports/pr-preview.md"
    report.write_report(out_path, md)
    click.echo(f"wrote {out_path}")

    if pr_number is not None:
        report.post_pr_comment(pr_number, md)
        click.echo(f"posted comment on PR #{pr_number}")


@main.command("gate")
@click.option("--run", "run_id", required=True, type=int)
@click.option("--db", default=DEFAULT_DB, show_default=True)
@click.option("--suite", default=DEFAULT_SUITE, show_default=True)
@click.option("--base-sha", default=None)
@click.option("--head-sha", default=None)
@click.option("--pr", "pr_number", default=None, type=int)
@click.option("--skip-triage", is_flag=True, help="Detect + report only, skip the agent call")
@click.option("--out", default=None, type=click.Path())
def gate_cmd(
    run_id: int,
    db: str,
    suite: str,
    base_sha: str | None,
    head_sha: str | None,
    pr_number: int | None,
    skip_triage: bool,
    out: str | None,
) -> None:
    """detect -> triage -> report -> (PR comment + commit status).

    Exits nonzero if any `major` regression is found -- this is the PR gate.
    """
    conn = store.get_connection(db)
    store.init_db(conn)
    run = store.get_run(conn, run_id)
    if run is None:
        raise click.ClickException(f"no such run: {run_id}")

    findings = detect.detect_regressions(conn, run_id)
    _print_findings_table(findings)

    triage_report = None
    if findings and not skip_triage:
        s = _load_suite(suite)
        entries = {e.name: e for e in s.entries}
        # base_sha/head_sha default to None -- each entry's source commits
        # are then resolved per-entry from recorded history (see
        # TriageContext.resolve_shas), since entries can live in different
        # sibling repos with independent commit histories from this repo's.
        ctx = triage.TriageContext(
            conn=conn,
            entries=entries,
            findings=findings,
            run_id=run_id,
            base_sha=base_sha,
            head_sha=head_sha,
        )
        try:
            triage_report = triage.run_triage(ctx)
        except triage.TriageError as exc:
            click.echo(f"warning: triage failed, continuing without narrative: {exc}", err=True)

    md = report.render_markdown(dict(run), findings, triage_report)
    out_path = out or (f"reports/pr-{pr_number}.md" if pr_number else f"reports/{run['ts'][:10]}.md")
    report.write_report(out_path, md)
    click.echo(f"wrote {out_path}")

    major = [f for f in findings if f.kind == "regression" and f.severity == "major"]

    if pr_number is not None:
        try:
            report.post_pr_comment(pr_number, md)
            report.set_commit_status(
                run["git_sha"],
                "failure" if major else "success",
                f"{len(major)} major regression(s)" if major else "no major regressions",
            )
        except RuntimeError as exc:
            click.echo(f"warning: gh posting failed: {exc}", err=True)

    if major:
        raise SystemExit(1)


@main.command("dashboard")
@click.option("--db", default=DEFAULT_DB, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (development only)")
def dashboard_cmd(db: str, host: str, port: int, reload: bool) -> None:
    """Serve the read-only dashboard API (M7 stretch) over the SQLite store."""
    try:
        import uvicorn
    except ImportError as exc:
        raise click.ClickException(
            "the dashboard needs the optional deps: pip install -e '.[dashboard]'"
        ) from exc

    os.environ["PERFLENS_DB"] = db
    uvicorn.run("perflens.dashboard.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
