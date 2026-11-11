"""Wine-prefix cloning helpers shared by launcher.py (N parallel driving
instances) and train_sac_vec.py (N parallel training instances feeding one
shared SACAgent).

Each instance needs its own WINEPREFIX because TMInterface's port is read
once at startup from a single file inside the prefix
(Documents/TMInterface/builtin.txt), and Wine scopes named objects (mutexes,
pipes) to the wineserver instance tied to a WINEPREFIX. Sharing one prefix
across instances would collide on both the port and any single-instance lock.
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

HEADLESS_SCREEN = os.environ.get("TM_HEADLESS_SCREEN", "1024x768x24")
VNC_SCREEN = os.environ.get("VNC_SCREEN", "1280x720x24")
VNC_PASSWORD_FILE = "/home/wineuser/.vnc/passwd"
BASE_VNC_PORT = 5900
BASE_NOVNC_PORT = 6080
# How many instances (counting up from 0) can be watched. Instance 0 is always
# watchable; the entrypoint serves it. Each extra one costs its frames being
# copied back from the GPU, so only ask for the views you'll use.
VIEW_INSTANCES = max(1, int(os.environ.get("TM_VIEW_INSTANCES", "1")))
NOVNC_ENABLED = os.environ.get("NOVNC_ENABLE", "false") == "true"


def display_for(index: int) -> str:
    """X display for an instance. Instance 0 uses the entrypoint's :0; every
    other instance gets its own."""
    return f":{index}"


def is_viewable(index: int) -> bool:
    return index < VIEW_INSTANCES


def render_env(index: int) -> dict:
    """Extra game environment for this instance.

    Observations come from the simulation state, not pixels, so frames nobody
    is watching have no consumer: skip copying them back from the GPU.
    """
    return {} if is_viewable(index) else {"VGL_READBACK": "none"}


def hide_console(index: int):
    """Close TMInterface's console, which otherwise covers the watched game.

    Call once the game is racing. The tilde key is TMInterface's own toggle;
    only the view depends on this, so failing is harmless.
    """
    if NOVNC_ENABLED and is_viewable(index):
        env = {**os.environ, "DISPLAY": display_for(index)}
        subprocess.run(["xdotool", "key", "grave"], env=env, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_display(index: int) -> str:
    """Start the X server for instance `index` (>0) and return its display.

    A viewable instance gets a TigerVNC server, on VNC port 5900+index and noVNC
    port 6080+index. Any other gets a small Xvfb. Instance 0 shares the
    entrypoint's :0.
    """
    display = display_for(index)
    if index == 0:
        return display
    sock = Path(f"/tmp/.X11-unix/X{index}")
    if not sock.exists():
        viewable = NOVNC_ENABLED and is_viewable(index)
        if viewable:
            width_height, depth = VNC_SCREEN.rsplit("x", 1)
            command = ["Xtigervnc", display, "-geometry", width_height, "-depth", depth,
                       "-rfbport", str(BASE_VNC_PORT + index), "-rfbauth", VNC_PASSWORD_FILE,
                       "-SecurityTypes", "VncAuth", "-AlwaysShared"]
        else:
            command = ["Xvfb", display, "-screen", "0", HEADLESS_SCREEN, "+extension", "MIT-SHM"]
        subprocess.Popen(
            command + ["-ac", "+extension", "GLX", "+iglx", "-nolisten", "tcp", "-noreset"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            if sock.exists():
                break
            time.sleep(0.1)
        if viewable:
            subprocess.Popen(
                ["websockify", "--web", "/usr/share/novnc/", str(BASE_NOVNC_PORT + index),
                 f"localhost:{BASE_VNC_PORT + index}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    return display


def clone_prefix(master_prefix: Path, instance_dir: Path):
    """Hard-link clone master_prefix into instance_dir.

    `cp -al` makes cloning near-instant and ~free on disk: files Wine never
    writes keep sharing inodes with the master prefix (and every other
    clone); only files an instance actually writes get their own copy.
    """
    if instance_dir.exists():
        shutil.rmtree(instance_dir)
    instance_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-al", str(master_prefix), str(instance_dir)], check=True)


def patch_port(instance_dir: Path, port: int):
    """Give this clone its own TMInterface port.

    The file must be unlinked (not truncated in place) before rewriting —
    truncating a hardlinked file mutates the inode shared with the master
    prefix (and every other clone), which would silently reset everyone's
    port back to the same value.
    """
    builtin_txt = instance_dir / "drive_c" / "users" / "wineuser" / "Documents" / "TMInterface" / "builtin.txt"
    content = builtin_txt.read_text()
    new_content = re.sub(r'set "custom_port" \d+', f'set "custom_port" {port}', content)
    builtin_txt.unlink()
    builtin_txt.write_text(new_content)


def instance_tmnf_dir(instance_dir: Path) -> Path:
    """Where TmForever.exe/TMLoader.exe live inside a cloned prefix."""
    return instance_dir / "drive_c" / "Program_Files_x86" / "TmNationsForever"
