"""Shared helper to launch TMLoader/TmForever in a background thread.

Used by `agent.py` and `train_sac.py` (the single default-Wine-prefix case)
and by `train_sac_vec.py` (one call per cloned Wine prefix, for N parallel
training instances) so the process-launch logic only lives in one place.
"""

import os
import subprocess
import threading
from pathlib import Path
from typing import Optional


def start_tmnf(
    tmnf_dir: Optional[Path] = None,
    wineprefix: Optional[Path] = None,
    log_name: str = "tmloader.log",
    display: Optional[str] = None,
    extra_env: Optional[dict] = None,
) -> threading.Thread:
    """Start TMLoader (which injects mods and launches TmForever) in a background thread.

    tmnf_dir defaults to $TMNF_DIR — the single default-prefix case used by
    agent.py/train_sac.py. Pass both tmnf_dir and wineprefix explicitly to
    launch a game inside a cloned Wine prefix instead (train_sac_vec.py's
    case): the child process needs WINEPREFIX pointed at that clone, not the
    container's default prefix, since TMInterface reads its port from a file
    inside the prefix and Wine scopes named objects to the wineserver tied to
    a given WINEPREFIX.
    """
    if tmnf_dir is None:
        tmnf_dir = Path(os.environ["TMNF_DIR"])
    tm_exe = tmnf_dir / "TmForever.exe"
    tmloader_exe = tmnf_dir / "TMLoader.exe"

    env = os.environ.copy()
    if wineprefix is not None:
        env["WINEPREFIX"] = str(wineprefix)
    if display is not None:
        env["DISPLAY"] = display
    if extra_env is not None:
        env.update(extra_env)

    def run_in_thread():
        try:
            exe = tmloader_exe if tmloader_exe.exists() else tm_exe
            print(f"Starting: {exe}")
            print(f"Exists: {exe.exists()}")

            log_path = Path("/home/wineuser/logs") / log_name
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                ["vglrun", "wine", str(exe)],
                stdout=log_file,
                stderr=log_file,
                cwd=str(exe.parent),
                env=env,
            )
            print(f"TMNF process started with PID: {proc.pid}")
            proc.wait()
            print(f"TMNF process exited with code: {proc.returncode}")
        except Exception as e:
            print(f"Error starting TMNF: {e}")
            import traceback

            traceback.print_exc()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    return thread
