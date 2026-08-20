"""Diffs and commit logs scoped to a suite entry's kernel source paths.

Used by the triage agent to correlate metric-delta findings with the code
that plausibly caused them. Every call is scoped to `source_paths` so the
agent never sees an unrelated diff and can't hallucinate a cause outside it.
"""

from __future__ import annotations

import subprocess


class GitContextError(RuntimeError):
    pass


def _run_git(repo_path: str, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", "-C", repo_path, *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise GitContextError(f"git {' '.join(args)} failed in {repo_path}: {proc.stderr.strip()}")
    return proc.stdout


def get_kernel_diff(
    repo_path: str, base_sha: str, head_sha: str, source_paths: list[str]
) -> str:
    """Unified diff between base_sha and head_sha, limited to source_paths."""
    if not source_paths:
        return ""
    out = _run_git(repo_path, ["diff", f"{base_sha}..{head_sha}", "--", *source_paths])
    return out if out.strip() else "(no changes in source_paths between base and head)"


def get_commit_log(
    repo_path: str, base_sha: str, head_sha: str, source_paths: list[str]
) -> list[dict[str, str]]:
    """Commits (sha, author, date, message) touching source_paths in (base, head]."""
    if not source_paths:
        return []
    fmt = "%H%x1f%an%x1f%aI%x1f%s"
    out = _run_git(
        repo_path,
        ["log", f"{base_sha}..{head_sha}", f"--pretty=format:{fmt}", "--", *source_paths],
    )
    commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, author, date, subject = line.split("\x1f", 3)
        commits.append({"sha": sha, "author": author, "date": date, "message": subject})
    return commits
