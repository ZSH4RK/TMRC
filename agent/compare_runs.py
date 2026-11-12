"""Compare recorded runs: a leaderboard, or what differs between two runs.

Works on the host (no game needed) as well as in the container:

    uv run agent/compare_runs.py                       # every run, best first
    uv run agent/compare_runs.py --kind eval --map A03-Race
    uv run agent/compare_runs.py --diff <run A> <run B>
    uv run agent/compare_runs.py --show <run>

Runs are ranked like a race (see experiments/stats.py): finishes by time, then
non-finishes by distance. Run ids can be given as any unique prefix or suffix.
"""

import argparse
import json
import sys
from pathlib import Path

from experiments.run import MANIFEST, RESULTS_DIR, SUMMARY
from experiments.stats import read_episodes, summarize


def load_run(path: Path) -> dict:
    manifest = json.loads((path / MANIFEST).read_text())
    summary_path = path / SUMMARY
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        # Still running, or killed before it could finalize: summarize what's there.
        summary = summarize(read_episodes(path))
        summary["status"] = manifest.get("status", "running")
    return {"path": path, "manifest": manifest, "summary": summary}


def all_runs(root: Path) -> list:
    return [load_run(p) for p in sorted(root.iterdir()) if (p / MANIFEST).exists()] if root.is_dir() else []


def find_run(root: Path, key: str) -> dict:
    direct = Path(key)
    if (direct / MANIFEST).exists():
        return load_run(direct)
    if (root / key / MANIFEST).exists():
        return load_run(root / key)
    matches =[p for p in root.iterdir() if (p / MANIFEST).exists() and (p.name.startswith(key) or p.name.endswith(key))]
    if len(matches) != 1:
        raise SystemExit(f"{'No' if not matches else 'Ambiguous'} run matching {key!r} in {root}"
                         + (f": {[m.name for m in matches]}" if matches else ""))
    return load_run(matches[0])


def _rank(run) -> tuple:
    s = run["summary"]
    if s.get("best_time_ms") is not None:
        return (0, s["best_time_ms"])
    best = s.get("best_episode") or {}
    return (1, -(best.get("distance_m") or 0.0))


def _fmt_ms(ms):
    return "-" if ms is None else f"{ms / 1000:.2f}s"


def _fmt(value, spec=".1f"):
    return "-" if value is None else format(value, spec)


def leaderboard(runs: list):
    header = ["#", "run", "status", "map", "policy", "eps", "finish%", "best", "median", "last-N track%", "gaps", "code", "seed"]
    rows = []
    for i, run in enumerate(sorted(runs, key=_rank), 1):
        m, s = run["manifest"], run["summary"]
        last = s.get("last_window") or {}
        rows.append([
            str(i), m["run_id"], s.get("status", m.get("status", "?")),
            (m.get("game") or {}).get("map_name") or "-",
            str(m.get("config", {}).get("policy") or "-")[-28:],
            str(s.get("episodes", 0)),
            _fmt(100 * s["finish_rate"]) if s.get("finish_rate") is not None else "-",
            _fmt_ms(s.get("best_time_ms")), _fmt_ms(s.get("median_time_ms")),
            _fmt((last.get("track_pct") or {}).get("mean")),
            str(s.get("step_gaps", "-")),
            m.get("code", {}).get("sha256", "")[:8], str(m.get("seed")),
        ])
    widths = [max(len(r[c]) for r in [header, *rows]) for c in range(len(header))]
    for r in [header, *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)))


def _flatten(data, prefix=""):
    out = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = data
    return out


def diff(a: dict, b: dict):
    """Everything that could explain a difference in results, side by side."""
    ma, mb = a["manifest"], b["manifest"]
    print(f"A: {ma['run_id']}\nB: {mb['run_id']}\n")
    sections = {
        "seed": ({"seed": ma.get("seed")}, {"seed": mb.get("seed")}),
        "config": (ma.get("config", {}), mb.get("config", {})),
        "env vars": (ma.get("env", {}), mb.get("env", {})),
        "game files": (ma.get("game") or {}, mb.get("game") or {}),
        "loaded checkpoint": (ma.get("loaded_checkpoint") or {}, mb.get("loaded_checkpoint") or {}),
        "code files": (ma.get("code", {}).get("files", {}), mb.get("code", {}).get("files", {})),
        "hardware": ({k: v for k, v in ma.get("hardware", {}).items() if k != "hostname"},
                     {k: v for k, v in mb.get("hardware", {}).items() if k != "hostname"}),
    }
    for title, (da, db) in sections.items():
        fa, fb = _flatten(da), _flatten(db)
        changed = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
        if title == "code files":
            print(f"[{title}] " + ("identical" if not changed else f"{len(changed)} differ:"))
            for k in changed:
                state = "only in A" if k not in fb else "only in B" if k not in fa else "changed"
                print(f"    {k}: {state}")
            continue
        print(f"[{title}] " + ("identical" if not changed else ""))
        for k in changed:
            print(f"    {k}: {fa.get(k)!r} -> {fb.get(k)!r}")
    print()
    leaderboard([a, b])


def show(run: dict):
    print(json.dumps({"manifest": {k: v for k, v in run["manifest"].items() if k != "code"},
                      "code_sha256": run["manifest"].get("code", {}).get("sha256"),
                      "summary": run["summary"]}, indent=2, default=str))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=RESULTS_DIR, help=f"results directory (default {RESULTS_DIR})")
    parser.add_argument("--kind", help="only runs of this kind (train, eval, drive, verify)")
    parser.add_argument("--map", help="only runs on this map")
    parser.add_argument("--name", help="only runs whose id contains this")
    parser.add_argument("--diff", nargs=2, metavar=("A", "B"), help="show what differs between two runs")
    parser.add_argument("--show", metavar="RUN", help="print one run's manifest and summary")
    args = parser.parse_args(argv)

    if args.diff:
        diff(find_run(args.root, args.diff[0]), find_run(args.root, args.diff[1]))
        return
    if args.show:
        show(find_run(args.root, args.show))
        return

    runs = all_runs(args.root)
    if args.kind:
        runs = [r for r in runs if r["manifest"].get("kind") == args.kind]
    if args.map:
        runs = [r for r in runs if (r["manifest"].get("game") or {}).get("map_name") == args.map]
    if args.name:
        runs = [r for r in runs if args.name in r["manifest"]["run_id"]]
    if not runs:
        print(f"No runs found in {args.root}", file=sys.stderr)
        return
    leaderboard(runs)


if __name__ == "__main__":
    main()
