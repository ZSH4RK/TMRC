"""Captures *what* produced a result: code, game files, config and hardware.

Git isn't available inside the container (the image gets a copy of the code,
not the repository), so the code is identified by content hash instead, and a
copy of it is kept with every run. TMRC_GIT_COMMIT is recorded too when the
image was built with it (see docker-compose.yml), or read from git when
running on the host.
"""

import hashlib
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

AGENT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = AGENT_DIR.parent

# Environment variables recorded in every manifest. TM_* covers all of the
# project's own settings; the rest change how the game renders or runs.
ENV_PREFIXES = ("TM_", "TMI_")
ENV_EXTRA = ("NOVNC_ENABLE", "NVIDIA_VISIBLE_DEVICES", "VNC_SCREEN", "TRACKS_DIR", "TMRC_GIT_COMMIT")

_SKIP_DIRS = {"__pycache__", "tmp", ".venv", ".pytest_cache"}


def sha256_file(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(files: dict) -> str:
    """One hash over {relative name: file hash}, independent of dict order."""
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(f"{name}\0{files[name] or ''}\n".encode())
    return digest.hexdigest()


def _python_sources(root: Path) -> list:
    return sorted(
        p for p in root.rglob("*.py")
        if not any(part in _SKIP_DIRS for part in p.relative_to(root).parts)
    )


def wine_paths() -> dict:
    """Files inside the Wine prefix that change what the game does. In the
    container these are the copies TMLoader/TMInterface actually load, which
    can differ from the repo's plugins/ if the image is stale."""
    prefix = Path(os.environ.get("WINEPREFIX", "/home/wineuser/.wine"))
    tmnf_dir = Path(os.environ.get("TMNF_DIR", prefix / "drive_c" / "Program_Files_x86" / "TmNationsForever"))
    documents = prefix / "drive_c" / "users" / "wineuser" / "Documents"
    return {
        "Python_Link.as": documents / "TMInterface" / "Plugins" / "Python_Link.as",
        "TMInterface_builtin.txt": documents / "TMInterface" / "builtin.txt",
        "TMLoader_default.yaml": tmnf_dir / "database" / "TmForever" / "profiles" / "default.yaml",
        "TMLoader_index.jsx": tmnf_dir / "ui" / "layouts" / "default" / "index.jsx",
    }


def code_files(tracks_dir: Optional[Path] = None) -> dict:
    """{name in the run's code/ snapshot: source path} for everything that
    decides how an agent drives: agent code, track code, the game-side plugin
    and config, and the dependency lock."""
    files = {f"agent/{p.relative_to(AGENT_DIR)}": p for p in _python_sources(AGENT_DIR)}
    if tracks_dir is not None and Path(tracks_dir).is_dir():
        files.update({f"Tracks/{p.name}": p for p in sorted(Path(tracks_dir).glob("*.py"))})
    for name in ("pyproject.toml", "uv.lock"):
        files[name] = PROJECT_DIR / name
    for name, path in wine_paths().items():
        files[f"game/{name}"] = path
    # Fall back to the repo's plugins when not in the container.
    for path in sorted((PROJECT_DIR / "plugins").glob("*")):
        files.setdefault(f"plugins/{path.name}", path)
    return {name: path for name, path in files.items() if Path(path).is_file()}


def snapshot_code(dest: Path, tracks_dir: Optional[Path] = None) -> dict:
    """Copy the code into dest and return {name: sha256}."""
    hashes = {}
    for name, path in code_files(tracks_dir).items():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[name] = sha256_file(path)
    return hashes


def git_info() -> dict:
    """Commit + dirty flag from git when available, else from the image's build arg."""
    info = {"commit": os.environ.get("TMRC_GIT_COMMIT") or None, "dirty": None, "source": "build-arg"}
    try:
        commit = subprocess.run(["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(PROJECT_DIR), "status", "--porcelain", "--", "agent", "plugins"],
                                capture_output=True, text=True, timeout=5, check=True).stdout
        info = {"commit": commit, "dirty": bool(status.strip()), "source": "git"}
    except (OSError, subprocess.SubprocessError):
        pass
    if info["commit"] in (None, "", "unknown"):
        info["commit"] = None
    return info


def find_map_file(map_file: str) -> Optional[Path]:
    """Locate a .Challenge.Gbx the way the game's `map` command would: the
    user's Challenges folder first, then the install's GameData."""
    prefix = Path(os.environ.get("WINEPREFIX", "/home/wineuser/.wine"))
    roots = [
        prefix / "drive_c" / "users" / "wineuser" / "Documents" / "TmForever" / "Tracks",
        prefix / "drive_c" / "Program_Files_x86" / "TmNationsForever" / "GameData" / "Tracks",
    ]
    for root in roots:
        if root.is_dir():
            for path in root.rglob(map_file):
                return path
    return None


def game_info(map_name: str, map_file: str, tracks_dir: Optional[Path]) -> dict:
    """Hashes of the map and the track geometry the observations are built from.
    A changed CSV changes every observation and reward, so it matters as much
    as a code change."""
    map_path = find_map_file(map_file)
    track_csv = Path(tracks_dir) / map_name / "points-offset.csv" if tracks_dir else None
    return {
        "map_name": map_name,
        "map_file": map_file,
        "map_sha256": sha256_file(map_path) if map_path else None,
        "track_csv_sha256": sha256_file(track_csv) if track_csv else None,
    }


def captured_env() -> dict:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES) or k in ENV_EXTRA}


def hardware_info() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "in_container": Path("/.dockerenv").exists(),
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        info["nvidia_smi"] = out.splitlines()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def checkpoint_hashes(directory) -> Optional[dict]:
    """{file: sha256} of a checkpoint directory, so a manifest pins exactly
    which weights a run started from or evaluated."""
    if directory is None:
        return None
    directory = Path(directory)
    if not directory.is_dir():
        return None
    return {p.name: sha256_file(p) for p in sorted(directory.iterdir()) if p.is_file()}
