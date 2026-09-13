"""Read Linux container limits and detect a lost ComfyUI process without CUDA."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

LOGGER = logging.getLogger("ltx25-health")
PROC_ROOT = Path("/proc")
CGROUP_ROOT = Path("/sys/fs/cgroup")
COMFY_PID_FILE = Path(os.getenv("COMFY_PID_FILE", "/tmp/comfyui.pid"))


class ComfyUnavailableError(RuntimeError):
    """The current worker cannot safely finish the submitted ComfyUI job."""


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _integer(path: Path) -> int | None:
    try:
        value = int(_read(path))
        return value if value >= 0 else None
    except ValueError:
        return None


def _memory_group() -> tuple[Path, Path, int] | None:
    # A cgroup namespace usually exposes this container at the mount root.
    # Older hosts expose the membership path from /proc/self/cgroup instead.
    candidates = []
    for line in _read(PROC_ROOT / "self/cgroup").splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        if not controllers:
            candidates.append((CGROUP_ROOT, relative, 2))
        elif "memory" in controllers.split(","):
            candidates.append((CGROUP_ROOT / "memory", relative, 1))
    candidates.extend([(CGROUP_ROOT, "/", 2), (CGROUP_ROOT / "memory", "/", 1)])
    for root, relative, version in candidates:
        path = root / relative.lstrip("/")
        # Membership containing '..' can occur in namespaced /proc mounts.
        # In that case only inspect the visible cgroup root, never outside it.
        if ".." in Path(relative).parts:
            path = root
        filename = "memory.current" if version == 2 else "memory.usage_in_bytes"
        if (path / filename).is_file():
            return path, root, version
    return None


def memory_snapshot() -> dict:
    info = {}
    for line in _read(PROC_ROOT / "meminfo").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in {"MemTotal:", "MemAvailable:"}:
            try:
                info[fields[0][:-1]] = int(fields[1]) * 1024
            except ValueError:
                pass
    result = {"host_total_bytes": info.get("MemTotal"),
              "host_available_bytes": info.get("MemAvailable"),
              "container_limit_bytes": None, "container_used_bytes": None,
              "container_peak_bytes": None, "oom_kills": None}
    group = _memory_group()
    if group:
        leaf, root, version = group
        limit_name = "memory.max" if version == 2 else "memory.limit_in_bytes"
        usage_name = "memory.current" if version == 2 else "memory.usage_in_bytes"
        peak_name = "memory.peak" if version == 2 else "memory.max_usage_in_bytes"
        tightest = leaf
        path = leaf
        while True:
            limit = _integer(path / limit_name)
            # v1 represents an unlimited allowance with a very large integer.
            if limit is not None and limit < 2**60:
                if result["container_limit_bytes"] is None or limit < result["container_limit_bytes"]:
                    result["container_limit_bytes"] = limit
                    tightest = path
            if path == root or root not in path.parents:
                break
            path = path.parent
        result["cgroup_version"] = version
        result["container_used_bytes"] = _integer(tightest / usage_name)
        result["container_peak_bytes"] = _integer(tightest / peak_name)
        if version == 2:
            for line in _read(leaf / "memory.events").splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[0] == "oom_kill":
                    try:
                        result["oom_kills"] = int(fields[1])
                    except ValueError:
                        pass
    totals = [x for x in (result["host_total_bytes"], result["container_limit_bytes"]) if x is not None]
    result["effective_total_bytes"] = min(totals) if totals else None
    return result


def process_status() -> dict:
    pid = _integer(COMFY_PID_FILE)
    if not pid:
        return {"state": "unknown", "pid": None}
    try:
        os.kill(pid, 0)  # Existence check only; does not send a signal.
    except ProcessLookupError:
        return {"state": "dead", "pid": pid}
    except PermissionError:
        pass
    try:
        stat = (PROC_ROOT / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return {"state": "dead", "pid": pid}
    except OSError:
        return {"state": "unknown", "pid": pid}
    # The command name may contain spaces or parentheses. Fields after the
    # final ')' start at field 3; starttime is field 22.
    fields = stat.rpartition(")")[2].split()
    if len(fields) < 20:
        return {"state": "unknown", "pid": pid}
    return {"state": "dead" if fields[0] in {"Z", "X", "x"} else "alive",
            "pid": pid, "starttime": fields[19]}


def diagnostics() -> dict:
    return {"memory": memory_snapshot(), "comfyui_process": process_status(),
            "memory_profile": os.getenv("COMFY_MEMORY_PROFILE", "conservative")}


class ComfyMonitor:
    def __init__(self):
        self.initial_process = process_status()
        self.initial_oom_kills = memory_snapshot()["oom_kills"]

    def check(self) -> None:
        current = process_status()
        if current["state"] == "dead":
            self.fail("ComfyUI exited while this worker was running")
        if self.initial_process["state"] == "alive" and current["state"] == "alive":
            if (current["pid"], current.get("starttime")) != (
                    self.initial_process["pid"], self.initial_process.get("starttime")):
                self.fail("ComfyUI restarted and lost the submitted job")
        elif self.initial_process["state"] == "unknown" and current["state"] == "alive":
            self.initial_process = current

    def fail(self, reason: str) -> None:
        memory = memory_snapshot()
        LOGGER.error("%s. Container diagnostics: %s", reason, json.dumps(memory))
        kills = memory["oom_kills"]
        if kills is not None and self.initial_oom_kills is not None and kills > self.initial_oom_kills:
            reason += "; the container recorded a new out-of-memory (OOM) kill"
        else:
            reason += "; check the worker logs for an OOM kill or startup error"
        limit = memory["container_limit_bytes"]
        if limit is not None:
            reason += f". Container RAM limit: {limit / 1024**3:.1f} GiB"
        reason += ". Container RAM and GPU VRAM are separate from disk/network-volume capacity. This worker will be replaced; the handler will not resubmit this job."
        raise ComfyUnavailableError(reason)
