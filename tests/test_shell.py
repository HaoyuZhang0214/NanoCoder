"""Windows shell routing: Git Bash discovery and spawn selection."""

import os
from unittest import mock

from nanocoder.shell import _git_bash, run_shell


def _nt_env(tmp_path):
    bash = tmp_path / "Git" / "bin" / "bash.exe"
    bash.parent.mkdir(parents=True)
    bash.write_text("", encoding="utf-8")
    return bash


def test_git_bash_prefers_program_files(tmp_path):
    bash = _nt_env(tmp_path)
    _git_bash.cache_clear()
    with (
        mock.patch("nanocoder.shell.os.name", "nt"),
        mock.patch.dict(os.environ, {"ProgramFiles": str(tmp_path)}),
    ):
        assert _git_bash() == str(bash)
    _git_bash.cache_clear()


def test_git_bash_none_off_windows():
    _git_bash.cache_clear()
    with mock.patch("nanocoder.shell.os.name", "posix"):
        assert _git_bash() is None
    _git_bash.cache_clear()


def test_git_bash_never_picks_wsl(tmp_path):
    _git_bash.cache_clear()
    with (
        mock.patch("nanocoder.shell.os.name", "nt"),
        mock.patch.dict(os.environ, {"ProgramFiles": str(tmp_path / "nope")}),
        mock.patch("nanocoder.shell.shutil.which", return_value=None),
    ):
        assert _git_bash() is None
    _git_bash.cache_clear()


def test_run_shell_uses_bash_list_form(tmp_path):
    bash = _nt_env(tmp_path)
    with (
        mock.patch("nanocoder.shell._git_bash", return_value=str(bash)),
        mock.patch("nanocoder.shell.subprocess.run") as run,
    ):
        run_shell("pwd", check=False)
    run.assert_called_once_with([str(bash), "-c", "pwd"], shell=False, check=False)


def test_run_shell_falls_back_to_platform_shell():
    with (
        mock.patch("nanocoder.shell._git_bash", return_value=None),
        mock.patch("nanocoder.shell.subprocess.run") as run,
    ):
        run_shell("pwd", check=False)
    run.assert_called_once_with("pwd", shell=True, check=False)


def test_posixify_rewrites_drive_paths():
    from nanocoder.shell import _posixify_drive_paths

    assert _posixify_drive_paths(r'cd C:\Users\runner\proj && pwd') == "cd C:/Users/runner/proj && pwd"
    assert _posixify_drive_paths(r'"C:\Users\runner\hook.sh"') == '"C:/Users/runner/hook.sh"'
    # shell escapes are not drive letters and stay untouched
    assert _posixify_drive_paths(r'echo "a\nb"') == r'echo "a\nb"'
    assert _posixify_drive_paths("ls /tmp && echo done") == "ls /tmp && echo done"
