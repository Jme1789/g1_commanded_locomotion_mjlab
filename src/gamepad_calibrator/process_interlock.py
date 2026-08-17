"""Read-only detection of simulator processes that must not overlap calibration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_BLOCKING_EXECUTABLES = frozenset({"unitree_mujoco", "g1_ctrl"})


@dataclass(frozen=True, slots=True)
class BlockingProcess:
    pid: int
    name: str
    argv: tuple[str, ...]


def _read_process(path: Path) -> BlockingProcess | None:
    """Read a process snapshot, returning None when it races or is protected."""
    try:
        name = (path / "comm").read_text().strip()
        argv = tuple(
            part.decode(errors="replace")
            for part in (path / "cmdline").read_bytes().split(b"\0")
            if part
        )
        try:
            executable = Path(os.readlink(path / "exe")).name
        except OSError:
            return None
    except OSError:
        return None

    argv0 = Path(argv[0]).name if argv else ""
    if executable not in _BLOCKING_EXECUTABLES and argv0 not in _BLOCKING_EXECUTABLES:
        return None
    return BlockingProcess(pid=int(path.name), name=name, argv=argv)


def find_blocking_processes(
    proc_root: Path = Path("/proc"),
) -> tuple[BlockingProcess, ...]:
    """Return exact simulator/controller conflicts without modifying any process."""
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return ()
    matches = [
        process
        for entry in entries
        if entry.name.isdigit() and (process := _read_process(entry)) is not None
    ]
    return tuple(sorted(matches, key=lambda process: process.pid))
