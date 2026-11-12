"""A run: one directory under results/ holding everything one execution produced.

The process that starts the work creates the run (Run.create). Child
processes it launches, like launcher.py's per-instance agent.py processes,
join it with Run.attach() using TM_RUN_DIR, which create() exports. Each
process writes its own episodes*.csv, so no two processes write the same
file. Threads within a process share one Run, which locks around its writes.
"""

import csv
import json
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import provenance, stats
from .replay import EpisodeRecorder, Replay
from .seeding import seed_everything

RESULTS_DIR = Path(os.environ.get("TM_RESULTS_DIR", provenance.PROJECT_DIR / "results"))

# Which episodes get a replay file:
#   best      every episode that beats this process's best so far (default):
#             a history of how the run improved, at a few files per run
#   finished  best, plus every finish
#   all       every episode (large: roughly 50-100 KB each)
#   none      none
SAVE_REPLAYS_MODES = ("best", "finished", "all", "none")

MANIFEST = "manifest.json"
SUMMARY = "summary.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")[:60]


def _write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


class Run:
    def __init__(self, path: Path, writer: str, manifest: dict, owner: bool, save_replays: str):
        if save_replays not in SAVE_REPLAYS_MODES:
            raise ValueError(f"TM_SAVE_REPLAYS must be one of {SAVE_REPLAYS_MODES}, not {save_replays!r}")
        self.path = Path(path)
        self.writer = writer
        self.manifest = manifest
        self.owner = owner
        self.save_replays = save_replays
        self.lock = threading.RLock()
        self._episode_count = 0
        self._best = None
        self._episode_file = None
        self._episode_writer = None
        self._metric_files = {}
        self._started = time.monotonic()
        self._closed = False

    # ---- creating / joining -------------------------------------------------

    @classmethod
    def create(
        cls,
        kind: str,
        config: Optional[dict] = None,
        *,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        map_name: Optional[str] = None,
        map_file: Optional[str] = None,
        tracks_dir: Optional[Path] = None,
        loaded_checkpoint: Optional[str] = None,
        root: Optional[Path] = None,
        save_replays: Optional[str] = None,
    ) -> "Run":
        """Start a new run: make its directory, seed every RNG, snapshot the
        code and write the manifest. `kind` is what's running (train, eval,
        drive, verify); `name` (default TM_RUN_NAME) labels the experiment."""
        name = name if name is not None else os.environ.get("TM_RUN_NAME") or None
        root = Path(root or RESULTS_DIR)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = "-".join(p for p in (stamp, kind, _slug(name) if name else None) if p)
        path = root / run_id
        suffix = 1
        while path.exists():
            suffix += 1
            path = root / f"{run_id}-{suffix}"
        path.mkdir(parents=True)

        seed = seed_everything(seed)
        code_hashes = provenance.snapshot_code(path / "code", tracks_dir)
        manifest = {
            "run_id": path.name,
            "kind": kind,
            "name": name,
            "status": "running",
            "created_at": _now(),
            "seed": seed,
            "config": config or {},
            "game": provenance.game_info(map_name, map_file, tracks_dir) if map_name else None,
            "loaded_checkpoint": {
                "path": str(loaded_checkpoint),
                "files": provenance.checkpoint_hashes(loaded_checkpoint),
            } if loaded_checkpoint else None,
            "code": {
                "sha256": provenance.hash_files(code_hashes),
                "git": provenance.git_info(),
                "files": code_hashes,
            },
            "env": provenance.captured_env(),
            "hardware": provenance.hardware_info(),
            "checkpoints": {},
        }
        _write_json(path / MANIFEST, manifest)
        # Children inherit this and attach instead of starting their own run.
        os.environ["TM_RUN_DIR"] = str(path)
        mode = save_replays or os.environ.get("TM_SAVE_REPLAYS", "best")
        run = cls(path, "main", manifest, owner=True, save_replays=mode)
        print(f"[run] Recording to {path} (seed {seed}, replays: {mode})", flush=True)
        return run

    @classmethod
    def attach(cls, path=None, writer: Optional[str] = None, save_replays: Optional[str] = None) -> "Run":
        """Join a run another process created (TM_RUN_DIR by default). Seeds
        this process from the run's seed, offset by writer so processes
        don't all draw the same random numbers."""
        path = Path(path or os.environ["TM_RUN_DIR"])
        manifest = json.loads((path / MANIFEST).read_text())
        writer = writer or f"instance-{os.environ.get('TM_INSTANCE_INDEX', os.getpid())}"
        offset = int(os.environ.get("TM_INSTANCE_INDEX", "0")) + 1
        seed_everything(manifest["seed"] + offset)
        mode = save_replays or os.environ.get("TM_SAVE_REPLAYS", "best")
        return cls(path, writer, manifest, owner=False, save_replays=mode)

    @classmethod
    def create_or_attach(cls, kind: str, config: Optional[dict] = None, **kwargs) -> "Run":
        if os.environ.get("TM_RUN_DIR") and (Path(os.environ["TM_RUN_DIR"]) / MANIFEST).exists():
            return cls.attach()
        return cls.create(kind, config, **kwargs)

    # ---- recording ----------------------------------------------------------

    @property
    def id(self) -> str:
        return self.manifest["run_id"]

    def recorder(self, **meta) -> EpisodeRecorder:
        return EpisodeRecorder(meta={"run_id": self.id, **meta})

    def log_episode(self, row: dict, replay: Optional[Replay] = None) -> dict:
        """Append an episode row, saving its replay if TM_SAVE_REPLAYS says
        to. Returns the row as written (with episode number and replay path)."""
        with self.lock:
            self._episode_count += 1
            row = {"episode": self._episode_count, "wall_time": _now(), **row}
            row["finished"] = row.get("done_reason") == "finished"
            new_best = stats.is_better(row, self._best)
            if new_best:
                self._best = dict(row)
            if replay is not None and self._should_save(row, new_best):
                stem = f"ep{row['episode']:06d}-{self.writer}-{row.get('done_reason') or 'end'}"
                if row["finished"] and row.get("race_time_ms") is not None:
                    stem += f"-{int(row['race_time_ms'])}ms"
                replay.meta.update({k: row.get(k) for k in ("episode", "done_reason", "race_time_ms", "distance_m", "policy")})
                replay.meta["new_best"] = new_best
                saved = replay.save(self.path / "replays" / stem)
                row["replay"] = str(saved.relative_to(self.path))
            self._write_episode(row)
            return row

    def _should_save(self, row: dict, new_best: bool) -> bool:
        return {
            "all": True,
            "none": False,
            "best": new_best,
            "finished": new_best or row["finished"],
        }[self.save_replays]

    def _write_episode(self, row: dict):
        if self._episode_writer is None:
            name = "episodes.csv" if self.writer == "main" else f"episodes-{self.writer}.csv"
            path = self.path / name
            new_file = not path.exists()
            self._episode_file = open(path, "a", newline="")
            self._episode_writer = csv.DictWriter(self._episode_file, fieldnames=stats.EPISODE_FIELDS, extrasaction="ignore")
            if new_file:
                self._episode_writer.writeheader()
        self._episode_writer.writerow(row)
        self._episode_file.flush()

    def log_metrics(self, name: str, row: dict):
        """Append a row to metrics/<name>.csv. Columns come from the first row."""
        with self.lock:
            entry = self._metric_files.get(name)
            if entry is None:
                path = self.path / "metrics" / f"{name}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                new_file = not path.exists()
                handle = open(path, "a", newline="")
                fields = ["wall_time", "elapsed_s", *[k for k in row if k not in ("wall_time", "elapsed_s")]]
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                entry = self._metric_files[name] = (handle, writer)
            handle, writer = entry
            writer.writerow({"wall_time": _now(), "elapsed_s": round(time.monotonic() - self._started, 1), **row})
            handle.flush()

    def archive_checkpoint(self, source_dir, label: str):
        """Copy a checkpoint directory into checkpoints/<label>/ and record its
        hashes, so the weights stay with the results they produced even after
        agent/tmp/ is overwritten by a later run."""
        source_dir = Path(source_dir)
        if not source_dir.is_dir() or not any(source_dir.iterdir()):
            return
        dest = self.path / "checkpoints" / label
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source_dir, dest)
        self.update_manifest(checkpoints={**self.manifest.get("checkpoints", {}),
                                          label: provenance.checkpoint_hashes(dest)})

    def update_manifest(self, **fields):
        if not self.owner:
            return
        with self.lock:
            self.manifest.update(fields)
            _write_json(self.path / MANIFEST, self.manifest)

    # ---- finishing ----------------------------------------------------------

    def finalize(self, status: str = "completed", **extra) -> Optional[dict]:
        """Close files; as the owner, also write summary.json and the final status."""
        with self.lock:
            if self._closed:
                return None
            self._closed = True
            if self._episode_file is not None:
                self._episode_file.close()
            for handle, _writer in self._metric_files.values():
                handle.close()
            if not self.owner:
                return None
            summary = {
                "run_id": self.id,
                "kind": self.manifest["kind"],
                "name": self.manifest.get("name"),
                "status": status,
                "ended_at": _now(),
                "duration_s": round(time.monotonic() - self._started, 1),
                "map_name": (self.manifest.get("game") or {}).get("map_name"),
                "policy": self.manifest.get("config", {}).get("policy"),
                **stats.summarize(stats.read_episodes(self.path)),
                **extra,
            }
            _write_json(self.path / SUMMARY, summary)
            self.manifest.update(status=status, ended_at=summary["ended_at"])
            _write_json(self.path / MANIFEST, self.manifest)
            print(f"[run] {status}: {summary.get('episodes', 0)} episodes, "
                  f"{summary.get('finishes', 0)} finishes, best time "
                  f"{summary.get('best_time_ms')}ms -> {self.path / SUMMARY}", flush=True)
            return summary

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, _tb):
        if exc_type is None:
            self.finalize("completed")
        elif issubclass(exc_type, KeyboardInterrupt):
            self.finalize("interrupted")
        else:
            self.finalize("crashed", error=f"{exc_type.__name__}: {exc}")
        return False
