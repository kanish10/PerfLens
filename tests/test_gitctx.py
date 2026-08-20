from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from perflens import gitctx


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "test@example.com")
    _git(r, "config", "user.name", "Test")

    (r / "src").mkdir()
    (r / "src" / "attention.cu").write_text("// v1\n__global__ void k() {}\n")
    (r / "README.md").write_text("hello\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "initial")

    return r


def test_get_kernel_diff_scoped_to_source_paths(repo: Path) -> None:
    base_sha = _git(repo, "rev-parse", "HEAD").strip()

    (repo / "src" / "attention.cu").write_text("// v2 -- added locals\n__global__ void k() { int x[64]; }\n")
    (repo / "README.md").write_text("hello world\n")  # unrelated change
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "regress attention kernel")
    head_sha = _git(repo, "rev-parse", "HEAD").strip()

    diff = gitctx.get_kernel_diff(str(repo), base_sha, head_sha, ["src/attention.cu"])

    assert "attention.cu" in diff
    assert "README.md" not in diff
    assert "int x[64]" in diff


def test_get_kernel_diff_empty_range_says_so(repo: Path) -> None:
    sha = _git(repo, "rev-parse", "HEAD").strip()
    diff = gitctx.get_kernel_diff(str(repo), sha, sha, ["src/attention.cu"])
    assert "no changes" in diff


def test_get_kernel_diff_no_source_paths_returns_empty(repo: Path) -> None:
    sha = _git(repo, "rev-parse", "HEAD").strip()
    assert gitctx.get_kernel_diff(str(repo), sha, sha, []) == ""


def test_get_commit_log_scoped_and_ordered(repo: Path) -> None:
    base_sha = _git(repo, "rev-parse", "HEAD").strip()

    (repo / "README.md").write_text("unrelated\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "unrelated readme change")

    (repo / "src" / "attention.cu").write_text("// v2\n__global__ void k() { int x[64]; }\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "regress attention kernel")
    head_sha = _git(repo, "rev-parse", "HEAD").strip()

    commits = gitctx.get_commit_log(str(repo), base_sha, head_sha, ["src/attention.cu"])

    assert len(commits) == 1
    assert commits[0]["message"] == "regress attention kernel"
    assert commits[0]["author"] == "Test"
    assert len(commits[0]["sha"]) == 40


def test_get_commit_log_invalid_repo_raises() -> None:
    with pytest.raises(gitctx.GitContextError):
        gitctx.get_commit_log("/nonexistent/path/xyz", "a", "b", ["x.cu"])
