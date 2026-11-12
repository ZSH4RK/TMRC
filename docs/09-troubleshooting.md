# 9. Troubleshooting

## My code changes aren't doing anything

Add `--build`: `docker compose up --build`. Code is copied into the image when
it's built, so without `--build` you're running the old version.

## The build fails

- **`failed to fetch anonymous token` / network errors:** a download failed.
  Check your internet connection and try again.
- **`tmnf-test:latest` not found:** you need the base TrackMania image first (see
  [page 2](02-setup-and-first-run.md#what-you-need)).
- **`nvcr.io` permission errors:** only when building the base image; it needs
  `docker login nvcr.io`.

## It can't find the track data (`points-offset.csv`)

The `Tracks` repository must be **next to** `TMNF-Docker`, not inside it (see
[page 2](02-setup-and-first-run.md#get-the-code)). If you changed `TM_MAP_NAME`,
that map needs its own folder in `Tracks/`.

## The browser view is black or won't connect

- Wait: the game takes 20–30 seconds to start.
- Use `http://localhost:6080` on the same computer (password `mypasswd`).
  From another computer, set `VNC_BIND=0.0.0.0`.
- Only instance 0 can be watched unless you raise `TM_VIEW_INSTANCES`.

## "Connection refused" or it hangs while connecting

The game took longer than expected to start. This is more likely with many
instances. Try fewer (`TM_NUM_INSTANCES=1`), and check the game's logs inside
the container in `/home/wineuser/logs/`:

```bash
docker compose exec tmnf ls /home/wineuser/logs
docker compose exec tmnf tail -50 /home/wineuser/logs/tmloader-eval-0.log
```

## My agent file "has no decide function" / isn't found

- The path after `agent:` is relative to the `agent/` folder:
  `agent:my_driver.py` means `TMRC/agent/my_driver.py`.
- The function must be called exactly `decide` and take two arguments.
- Did you rebuild (`--build`)?

## `distinct_outcomes` is more than 1 for a deterministic driver

Every episode should be identical, but they weren't. Possible causes:
- your driver uses randomness or remembers things between episodes (variables
  at the top of your file keep their values)
- `step_gaps` is above 0 (see below)
- the computer was heavily overloaded

## `step_gaps` is above 0

Game time jumped by more than one step between decisions. The game normally
waits for your code, so this means the connection timed out or something got
out of sync. The results of that run aren't fully trustworthy. Try fewer
instances, check your `decide` isn't extremely slow (`agent_ms_max`), and
rerun.

## A replay `DIVERGED` when verified

The same keys led to a different route. Check `compare_runs.py --diff` between
the original run and the verify run, especially the **code files** and **game
files** sections. If nothing differs and `step_gaps` was 0 in both, that's
unexpected, so please report it with both run folders.

## Training is slow

- Check `metrics/perf.csv`: `realtime_per_game` says how many times faster than
  real time each game runs. If it's well below `TM_GAME_SPEED`, the computer is
  the bottleneck. Try fewer instances.
- RL genuinely needs a lot of experience. Hours is normal.

## Results files are owned by another user / permission denied

Files in `results/` are written by the container's user (ID 1000). If your user
on the host has a different ID, fix ownership with
`sudo chown -R $(id -u):$(id -g) TMRC/results`.

## Still stuck?

Collect: the command you ran, the last ~50 lines of output, and the run folder
(it has the manifest with all settings). That's usually enough for someone to
help.
