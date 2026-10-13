"""Artifact extraction for shell-produced files (execute / execute_shell_command)."""

from __future__ import annotations

from pathlib import Path

from octop.infra.agents.threads.artifact import (
    extract_artifact_paths,
    is_artifact_tool_name,
    normalize_artifact_path,
)


def _ws(tmp_path: Path) -> Path:
    ws = tmp_path / "agents" / "a1"
    (ws / "generated").mkdir(parents=True)
    return ws


def test_shell_tools_are_artifact_tools() -> None:
    assert is_artifact_tool_name("execute")
    assert is_artifact_tool_name("execute_shell_command")
    assert is_artifact_tool_name("ns/execute")
    # existing tools must keep working
    assert is_artifact_tool_name("write_file")
    assert is_artifact_tool_name("send_file_to_user")
    # non-producers stay excluded
    assert not is_artifact_tool_name("read_file")
    assert not is_artifact_tool_name("ls")


def test_shell_output_records_existing_file(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "generated" / "report.docx").write_bytes(b"x")

    got = extract_artifact_paths(
        tool_name="execute",
        args={"command": "node gen.js"},
        result="wrote generated/report.docx",
        workspace_dir=ws,
    )
    assert got == [f"{ws.as_posix()}/generated/report.docx"]


def test_shell_output_ignores_nonexistent_path(tmp_path: Path) -> None:
    """A path printed by a failing command must not mint an artifact."""
    ws = _ws(tmp_path)

    got = extract_artifact_paths(
        tool_name="execute",
        args={"command": "node gen.js"},
        result="Error: cannot find generated/ghost.docx",
        workspace_dir=ws,
    )
    assert got == []


def test_shell_output_without_path_records_nothing(tmp_path: Path) -> None:
    """Regression: `Document created successfully.` carries no path at all."""
    ws = _ws(tmp_path)
    (ws / "generated" / "real.docx").write_bytes(b"x")

    got = extract_artifact_paths(
        tool_name="execute",
        args={"command": "node gen.js"},
        result="Document created successfully.",
        workspace_dir=ws,
    )
    assert got == []


def test_shell_paths_outside_workspace_are_rejected(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    outside = tmp_path / "etc" / "passwd"
    outside.parent.mkdir(parents=True)
    outside.write_text("x")

    got = extract_artifact_paths(
        tool_name="execute",
        args={},
        result=f"cat {outside} -> {outside}",
        workspace_dir=ws,
    )
    assert got == []


def test_file_tools_are_not_existence_gated(tmp_path: Path) -> None:
    """write_file records its target even if the row is written before flush."""
    ws = _ws(tmp_path)

    got = extract_artifact_paths(
        tool_name="write_file",
        args={"path": "generated/pending.md"},
        result="ok",
        workspace_dir=ws,
    )
    assert got == [f"{ws.as_posix()}/generated/pending.md"]


def test_normalize_rejects_extensionless_dirs(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    assert normalize_artifact_path("generated", ws) == ""
    assert normalize_artifact_path("generated/", ws) == ""
