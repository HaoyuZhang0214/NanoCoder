"""Docker container sandbox for bash command execution.

Only the bash tool uses this path. Hooks and file tools stay on the host.
Requires the local ``docker`` CLI; no Python Docker SDK dependency.
"""

from __future__ import annotations

import os
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from .shell import run_shell

CONTAINER_WORKSPACE = "/workspace"
DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_MEMORY = "2g"
DEFAULT_CPUS = "2"
DEFAULT_PIDS = 256


class SandboxError(Exception):
    """Sandbox setup or Docker CLI failure."""


class SandboxPathError(SandboxError):
    """Host cwd is outside the frozen workspace root."""


@dataclass
class SandboxConfig:
    """Settings for a session-scoped Docker sandbox."""

    workspace_root: str
    image: str = DEFAULT_IMAGE
    network: str = "bridge"  # bridge | none
    memory: str = DEFAULT_MEMORY
    cpus: str = DEFAULT_CPUS
    pids_limit: int = DEFAULT_PIDS


class HostRunner:
    """Run commands on the host via ``run_shell`` (default / hooks path)."""

    def run(self, command: str, check: bool = False, **kwargs) -> subprocess.CompletedProcess:
        return run_shell(command, check=check, **kwargs)


def check_docker() -> None:
    """Raise SandboxError if the Docker CLI is missing or the daemon is down."""
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except FileNotFoundError as e:
        raise SandboxError(
            "Docker CLI not found. Install Docker and ensure `docker` is on PATH."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise SandboxError("Timed out talking to Docker. Is the daemon running?") from e
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "unknown error").strip()
        raise SandboxError(
            f"Docker is not available ({err}). Start Docker Desktop / the daemon and retry."
        )


def _sandbox_user() -> str:
    """Non-root user for the container; match host uid on POSIX so mounts are writable."""
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        return f"{os.getuid()}:{os.getgid()}"
    return "65534:65534"


def _host_to_container(workspace_root: str, host_cwd: str) -> str:
    """Map a host absolute path under the workspace to ``/workspace/...``."""
    root = Path(workspace_root).resolve()
    cwd = Path(host_cwd).resolve()
    try:
        rel = cwd.relative_to(root)
    except ValueError as e:
        raise SandboxPathError(
            f"cwd {cwd} is outside sandbox workspace {root}; "
            "bash in Docker can only run under the session workspace."
        ) from e
    if str(rel) in (".", ""):
        return CONTAINER_WORKSPACE
    return f"{CONTAINER_WORKSPACE}/{rel.as_posix()}"


class DockerSandbox:
    """Session-scoped container: one ``docker run -d``, then ``docker exec`` per command."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.workspace_root = str(Path(config.workspace_root).resolve())
        self._container_id: str | None = None
        self._lock = threading.Lock()

    def ensure_started(self) -> str:
        """Start the long-lived container once; return its id."""
        with self._lock:
            if self._container_id is not None:
                return self._container_id
            name = f"nanocoder-sbx-{uuid.uuid4().hex[:12]}"
            cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--read-only",
                "--tmpfs",
                "/tmp:rw,exec,nosuid,size=256m",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                _sandbox_user(),
                "--memory",
                self.config.memory,
                "--cpus",
                self.config.cpus,
                "--pids-limit",
                str(self.config.pids_limit),
                "--network",
                self.config.network,
                "-v",
                f"{self.workspace_root}:{CONTAINER_WORKSPACE}:rw",
                "-w",
                CONTAINER_WORKSPACE,
                "-e",
                "HOME=/tmp",
                "-e",
                "LANG=C.UTF-8",
                "-e",
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                self.config.image,
                "sleep",
                "infinity",
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=120,
                )
            except FileNotFoundError as e:
                raise SandboxError(
                    "Docker CLI not found. Install Docker and ensure `docker` is on PATH."
                ) from e
            except subprocess.TimeoutExpired as e:
                raise SandboxError("Timed out creating the sandbox container.") from e
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "unknown error").strip()
                raise SandboxError(f"Failed to start sandbox container: {err}")
            cid = (proc.stdout or "").strip()
            if not cid:
                raise SandboxError("docker run succeeded but returned no container id")
            self._container_id = cid
            return cid

    def run(self, command: str, check: bool = False, **kwargs) -> subprocess.CompletedProcess:
        """Execute ``command`` inside the sandbox via ``docker exec``."""
        cid = self.ensure_started()
        host_cwd = kwargs.pop("cwd", None) or self.workspace_root
        container_cwd = _host_to_container(self.workspace_root, str(host_cwd))
        exec_cmd = [
            "docker",
            "exec",
            "-w",
            container_cwd,
            "-e",
            "HOME=/tmp",
            "-e",
            "LANG=C.UTF-8",
            "-e",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            cid,
            "bash",
            "-lc",
            command,
        ]
        return subprocess.run(exec_cmd, check=check, **kwargs)

    def close(self) -> None:
        """Force-remove the session container if it was started."""
        with self._lock:
            cid = self._container_id
            self._container_id = None
        if not cid:
            return
        try:
            subprocess.run(
                ["docker", "rm", "-f", cid],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass  # best-effort cleanup on exit
