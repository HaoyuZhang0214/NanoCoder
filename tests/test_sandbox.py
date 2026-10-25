"""Docker sandbox: path mapping, harden flags, lifecycle (mocked docker CLI)."""

import subprocess
from unittest import mock

import pytest

from nanocoder.sandbox import (
    CONTAINER_WORKSPACE,
    DockerSandbox,
    HostRunner,
    SandboxConfig,
    SandboxError,
    SandboxPathError,
    _host_to_container,
    check_docker,
)
from nanocoder.tools.bash import BashTool


def test_host_to_container_root_and_subdir(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    sub = root / "src"
    sub.mkdir()
    assert _host_to_container(str(root), str(root)) == CONTAINER_WORKSPACE
    assert _host_to_container(str(root), str(sub)) == f"{CONTAINER_WORKSPACE}/src"


def test_host_to_container_outside_raises(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()
    with pytest.raises(SandboxPathError):
        _host_to_container(str(root), str(outside))


def test_check_docker_ok():
    ok = subprocess.CompletedProcess(
        args=["docker"], returncode=0, stdout="27.0.0\n", stderr=""
    )
    with mock.patch("nanocoder.sandbox.subprocess.run", return_value=ok) as run:
        check_docker()
    run.assert_called_once()
    assert run.call_args.args[0][:2] == ["docker", "version"]


def test_check_docker_missing_cli():
    with mock.patch(
        "nanocoder.sandbox.subprocess.run", side_effect=FileNotFoundError("docker")
    ):
        with pytest.raises(SandboxError, match="Docker CLI not found"):
            check_docker()


def test_check_docker_daemon_down():
    bad = subprocess.CompletedProcess(
        args=["docker"], returncode=1, stdout="", stderr="Cannot connect to the Docker daemon"
    )
    with mock.patch("nanocoder.sandbox.subprocess.run", return_value=bad):
        with pytest.raises(SandboxError, match="not available"):
            check_docker()


def test_docker_sandbox_run_hardens_and_execs(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    sbx = DockerSandbox(
        SandboxConfig(workspace_root=str(root), image="python:3.12-slim", network="bridge")
    )

    run_calls: list = []

    def fake_run(cmd, **kwargs):
        run_calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="ciddeadbeef\n", stderr="")
        if cmd[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")
        if cmd[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected cmd: {cmd}")

    with mock.patch("nanocoder.sandbox.subprocess.run", side_effect=fake_run):
        proc = sbx.run("echo hi", check=False, capture_output=True, text=True, cwd=str(root))
        assert proc.returncode == 0
        assert proc.stdout == "ok\n"
        sbx.close()

    run_cmd = run_calls[0]
    assert run_cmd[0:2] == ["docker", "run"]
    assert "--read-only" in run_cmd
    assert "--cap-drop" in run_cmd and "ALL" in run_cmd
    assert "--security-opt" in run_cmd and "no-new-privileges" in run_cmd
    assert "--network" in run_cmd and "bridge" in run_cmd
    assert f"{root.resolve()}:{CONTAINER_WORKSPACE}:rw" in run_cmd
    assert "python:3.12-slim" in run_cmd

    exec_cmd = run_calls[1]
    assert exec_cmd[0:2] == ["docker", "exec"]
    assert "-w" in exec_cmd
    assert exec_cmd[exec_cmd.index("-w") + 1] == CONTAINER_WORKSPACE
    assert "ciddeadbeef" in exec_cmd
    assert exec_cmd[-3:] == ["bash", "-lc", "echo hi"]

    assert run_calls[2][:3] == ["docker", "rm", "-f"]


def test_docker_sandbox_rejects_cwd_outside_workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    sbx = DockerSandbox(SandboxConfig(workspace_root=str(root)))

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc\n", stderr="")
        raise AssertionError("exec should not be reached")

    with mock.patch("nanocoder.sandbox.subprocess.run", side_effect=fake_run):
        with pytest.raises(SandboxPathError):
            sbx.run("pwd", cwd=str(outside))


def test_docker_sandbox_maps_subdir_cwd(tmp_path):
    root = tmp_path / "ws"
    sub = root / "pkg"
    sub.mkdir(parents=True)
    sbx = DockerSandbox(SandboxConfig(workspace_root=str(root)))
    seen = {}

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="xyz\n", stderr="")
        if cmd[:2] == ["docker", "exec"]:
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with mock.patch("nanocoder.sandbox.subprocess.run", side_effect=fake_run):
        sbx.run("ls", cwd=str(sub))
    assert seen["cmd"][seen["cmd"].index("-w") + 1] == f"{CONTAINER_WORKSPACE}/pkg"


def test_bash_tool_uses_injected_runner(tmp_path):
    class FakeRunner:
        def __init__(self):
            self.commands = []

        def run(self, command, check=False, **kwargs):
            self.commands.append(command)
            return subprocess.CompletedProcess(
                args=["fake"], returncode=0, stdout="from-runner\n", stderr=""
            )

    runner = FakeRunner()
    tool = BashTool(runner=runner)
    out = tool.execute("echo hello")
    assert "from-runner" in out
    assert runner.commands == ["echo hello"]


def test_host_runner_delegates_to_run_shell():
    with mock.patch("nanocoder.sandbox.run_shell") as rs:
        rs.return_value = subprocess.CompletedProcess(["x"], 0, stdout="", stderr="")
        HostRunner().run("pwd", check=False, cwd="/tmp")
    rs.assert_called_once_with("pwd", check=False, cwd="/tmp")
