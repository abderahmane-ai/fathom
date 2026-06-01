"""Smoke tests for scripts/render_summary.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_render_summary_writes_top_level_md(tmp_path: Path) -> None:
    """Build two per-benchmark SUMMARY.md files, run render_summary, verify output."""
    # Build a results dir with two per-benchmark summaries.
    results = tmp_path / "results"
    for bench in ("depth_preservation", "lm_quality"):
        d = results / bench
        d.mkdir(parents=True)
        (d / "SUMMARY.md").write_text(
            f"# {bench}\n\n| Mode | value |\n|------|-------|\n| a | 1 |\n",
            encoding="utf-8",
        )

    out = results / "SUMMARY.md"
    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "scripts.render_summary", "--results", str(results), "--out", str(out)],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"render_summary failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    # Both per-benchmark names should appear as section headings.
    assert "## depth_preservation" in text
    assert "## lm_quality" in text
    # And in the TOC.
    assert "Table of Contents" in text
    # The pre-benchmark content should be preserved.
    assert "Mode" in text
    assert "value" in text


def test_render_summary_no_benchmarks(tmp_path: Path) -> None:
    """render_summary returns 0 even with no per-benchmark dirs, but logs a warning."""
    results = tmp_path / "results"
    results.mkdir()
    out = results / "SUMMARY.md"
    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "scripts.render_summary", "--results", str(results), "--out", str(out)],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # 0 benchmarks → exit 1
    assert result.returncode == 1
    assert out.is_file()
    # Should still have the top-level H1 + TOC.
    text = out.read_text(encoding="utf-8")
    assert "# Project Summary" in text
    assert "Table of Contents" in text
