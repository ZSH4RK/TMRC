"""
Launches TM_NUM_INSTANCES parallel TmForever/TMInterface processes.

Each instance gets its own WINEPREFIX (hardlink-cloned from the master prefix,
so cloning is near-instant and costs ~0 extra disk) because TMInterface's port
is read once at startup from a single file inside the prefix
(Documents/TMInterface/builtin.txt), and Wine scopes named objects (mutexes,
pipes) to the wineserver instance tied to a WINEPREFIX. Sharing one prefix
across instances would collide on both the port and any single-instance lock.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from instance_launch import clone_prefix, instance_tmnf_dir, patch_port, render_env, start_display

MASTER_PREFIX = Path(os.environ["WINEPREFIX"])
INSTANCES_DIR = Path("/home/wineuser/.wine-instances")
LOGS_DIR = Path("/home/wineuser/logs")
AGENT_DIR = Path(__file__).parent

NUM_INSTANCES = int(os.environ.get("TM_NUM_INSTANCES", "1"))
BASE_PORT = int(os.environ.get("TM_BASE_PORT", "8775"))
GAME_SPEED = os.environ.get("TM_GAME_SPEED", "5.0")

BOOT_STAGGER_S = 5


def build_env(instance_dir: Path, port: int, index: int) -> dict:
    env = os.environ.copy()
    env["WINEPREFIX"] = str(instance_dir)
    env["TMNF_DIR"] = str(instance_tmnf_dir(instance_dir))
    env["TMI_PORT"] = str(port)
    env["TM_GAME_SPEED"] = GAME_SPEED
    env["TM_INSTANCE_INDEX"] = str(index)
    env["DISPLAY"] = start_display(index)
    env.update(render_env(index))
    return env


def main():
    children = []

    def shutdown(signum, _frame):
        print(f"[launcher] Received signal {signum}, shutting down {len(children)} instance(s)...")
        for proc in children:
            if proc.poll() is None:
                proc.send_signal(signum)
        for proc in children:
            proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    for i in range(NUM_INSTANCES):
        port = BASE_PORT + i
        instance_dir = INSTANCES_DIR / str(i)
        log_dir = LOGS_DIR / f"instance-{i}"
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"[launcher] Preparing instance {i} (port {port})...")
        clone_prefix(MASTER_PREFIX, instance_dir)
        patch_port(instance_dir, port)

        env = build_env(instance_dir, port, i)
        log_file = open(log_dir / "agent.log", "w")

        print(f"[launcher] Starting instance {i} (port {port}, speed {GAME_SPEED}x), logs: {log_dir / 'agent.log'}")
        proc = subprocess.Popen(
            ["uv", "run", "agent.py"],
            cwd=str(AGENT_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        children.append(proc)

        if i < NUM_INSTANCES - 1:
            time.sleep(BOOT_STAGGER_S)

    exit_code = 0
    for i, proc in enumerate(children):
        ret = proc.wait()
        print(f"[launcher] Instance {i} exited with code {ret}")
        if ret != 0:
            exit_code = ret

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
