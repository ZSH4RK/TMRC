"""Trace a map's walls by driving along them, to make its Tracks/<map>/ data.

The car "sees" walls by measuring against a map of the track (Tracks/<map>/
points-offset.csv). For A03-Race that map was made from two hand-driven
replays, one hugging each wall. This script drives those laps itself.

It follows a route from start to finish (the map's author replay, which the
game ships for every campaign map), but aims at a point well to one side of
it, so the car runs into that side's wall and rubs along it. Whenever the
game reports wall contact, the car's position is a point on that wall.
Aiming along the route keeps it heading for the finish: plain wall-following
would circle any pocket in the track, such as A01's start area.

Run it (one map per run, both walls; game speed doesn't change the physics):
    TM_MODE=script TM_SCRIPT=agent/tools/trace_walls.py TM_MAP_FILE=A01-Race.Challenge.Gbx \\
        TM_NUM_INSTANCES=1 docker compose up --build

Needs results/traces/<map>-route.csv (x,y,z per row, start to finish), which
Tracks/route_from_replay.py makes from the author replay. Writes
results/traces/<map>-left.csv and <map>-right.csv, one row per 10 ms step.
Then, on the host, Tracks/traces_to_points.py turns those into
Tracks/<map>/points-offset.csv. Check the plot it makes before trusting it.
"""

import math
import os
import shutil
import socket
import struct
import sys
from pathlib import Path

import numpy as np

# The agent/ folder: env, track_features and game/ live there.
AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

from game.instance_launch import hide_console, launch_games  # noqa: E402
from game.tminterface2 import MessageType, TMInterface  # noqa: E402

MAP_FILE = os.environ.get("TM_MAP_FILE", "A01-Race.Challenge.Gbx")
MAP_NAME = MAP_FILE.removesuffix(".Challenge.Gbx")
SIDES = os.environ.get("TM_TRACE_SIDES", "left,right").split(",")
BASE_PORT = int(os.environ.get("TM_BASE_PORT", "8775"))
GAME_SPEED = float(os.environ.get("TM_GAME_SPEED", "3.0"))
OUT_DIR = Path(os.environ.get("TM_TRACE_DIR", AGENT_DIR.parent / "results" / "traces"))
# Tracks/<map>/route.csv (route_from_replay.py), else results/traces/<map>-route.csv.
_TRACKS_ROUTE = AGENT_DIR.parent / "Tracks" / MAP_NAME / "route.csv"
ROUTE_CSV = _TRACKS_ROUTE if _TRACKS_ROUTE.exists() else OUT_DIR / f"{MAP_NAME}-route.csv"

STEP_MS = 10                 # finest control the physics allows
TARGET_SPEED_MS = float(os.environ.get("TM_TRACE_SPEED", "16"))  # ~58 km/h
LOOKAHEAD_M = float(os.environ.get("TM_TRACE_LOOKAHEAD", "30"))  # how far along the route to aim
SIDE_OFFSET_M = float(os.environ.get("TM_TRACE_OFFSET", "40"))   # how far to the side of it
HEADING_DEADBAND = math.radians(3)
AWAY_MS = 40                 # steer away this long after touching the wall
MAX_RACE_MS = 6 * 60 * 1000
STUCK_MS = 3000              # less than STUCK_PROGRESS_M along the route in this long: stuck
STUCK_PROGRESS_M = 3.0
REVERSE_MS = 1200            # then reverse away from whatever it's against...
DETOUR_M = 40.0              # ...and follow the route itself this far before finding the wall again
SNAPSHOT_MS = 500            # keep a state to rewind to this often
REWIND_SNAPSHOTS = 4         # need at least this many snapshots to rewind
REWIND_BACK_M = 30.0         # rewind this far back along the route; 3x, 5x... on repeats
DETOUR_LOOKAHEAD_M = 12.0    # aim this close ahead while following the route itself
REWIND_DETOUR_M = 150.0      # and follow the route this far past where it fell
FALL_MS = 1500               # below the road this long...
FALL_PROGRESS_M = 20.0       # ...with less progress than this = fallen off
FALL_DROP_M = 6.0            # this far below the route's height = fallen off
AIRBORNE_DETOUR_M = 60.0     # follow the route until this far past the last time in the air
CALIBRATE_MS = 600           # hold left this long at the start to learn which way it turns

GAME_DIR = Path(os.environ.get("WINEPREFIX", "/home/wineuser/.wine")) / "drive_c"
CAMPAIGN_DIR = GAME_DIR / "Program_Files_x86/TmNationsForever/GameData/Tracks/Campaigns/Nations"
CHALLENGES_DIR = GAME_DIR / "users/wineuser/Documents/TmForever/Tracks/Challenges"


def stage_map():
    """`map <file>` loads from Tracks/Challenges. Campaign maps live in the
    game folder, so copy this one over if it isn't there yet."""
    dst = CHALLENGES_DIR / MAP_FILE
    if dst.exists():
        return
    matches = list(CAMPAIGN_DIR.rglob(MAP_FILE))
    if not matches:
        raise FileNotFoundError(f"{MAP_FILE} is neither in {CHALLENGES_DIR} nor in the campaign folders")
    CHALLENGES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(matches[0], dst)
    print(f"[trace] staged {matches[0]} -> {dst}", flush=True)


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Route:
    """The start-to-finish route as an arc-length-parameterised XZ polyline."""

    def __init__(self, csv_path: Path):
        pts = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1) > 0.05])
        self.xz = pts[keep][:, [0, 2]]
        self.y = pts[keep][:, 1]
        seg = np.diff(self.xz, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])
        self.index = 0

    def reset(self):
        self.index = 0

    def project(self, x, z) -> float:
        """Arc length of the car's projection onto the route, searching only
        near where it was last, so it can't jump to another part of the track
        that passes close by."""
        lo = max(0, self.index - 10)
        hi = min(len(self.xz) - 1, self.index + 60)
        a, b = self.xz[lo:hi], self.xz[lo + 1:hi + 1]
        ab = b - a
        t = np.clip(np.sum((np.array([x, z]) - a) * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-9), 0, 1)
        d = np.sum((a + t[:, None] * ab - np.array([x, z])) ** 2, axis=1)
        i = int(np.argmin(d))
        self.index = lo + i
        return float(self.cum[lo + i] + t[i] * self.seg_len[lo + i])

    def height(self, s, window_m=150.0):
        """Lowest route height within window_m of s. The route is a driven
        line, so it's in the air over jumps: the lowest nearby point is the
        road's height, near enough."""
        lo, hi = np.searchsorted(self.cum, [s - window_m, s + window_m])
        nearby = self.y[max(lo - 1, 0):hi + 1]
        return float(nearby.min()) if len(nearby) else float(np.interp(s, self.cum, self.y))

    def point(self, s):
        s = min(max(s, 0.0), self.length)
        i = min(int(np.searchsorted(self.cum, s, side="right")) - 1, len(self.seg_len) - 1)
        t = (s - self.cum[i]) / max(self.seg_len[i], 1e-9)
        return self.xz[i] + t * (self.xz[i + 1] - self.xz[i])


class Game:
    """The bare socket loop: wait for a step, answer it."""

    def __init__(self, port):
        self.tm = TMInterface(port)
        self.tm.register(timeout=5)
        self.sock = self.tm.sock
        self.sock.sendall(struct.pack("i", MessageType.SC_ON_CONNECT_SYNC))
        self.tm.set_speed(GAME_SPEED)
        self.tm.set_on_step_period(STEP_MS)
        self.tm.set_timeout(60_000)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, struct.pack("ll", 120, 0))
        self.finished = False

    def wait_step(self):
        while True:
            header = self.sock.recv(4, socket.MSG_WAITALL)
            if len(header) < 4:
                raise ConnectionError("TMInterface closed the connection")
            msg = struct.unpack("i", header)[0]
            if msg == MessageType.SC_RUN_STEP_SYNC:
                race_time = struct.unpack("i", self.sock.recv(4, socket.MSG_WAITALL))[0]
                return race_time, self.tm.get_simulation_state()
            if msg in (MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC, MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                current, target = struct.unpack("ii", self.sock.recv(8, socket.MSG_WAITALL))
                if msg == MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
                    print(f"[trace]   checkpoint {current}/{target}", flush=True)
                    if current == target and target > 0:
                        self.tm.prevent_simulation_finish()
                        self.finished = True
            self.sock.sendall(struct.pack("i", msg))

    def ack(self):
        self.sock.sendall(struct.pack("i", MessageType.SC_RUN_STEP_SYNC))

    def restart(self, first: bool):
        """Start a fresh run and return at race time 0 (after the countdown)."""
        if first:
            self.tm.execute_command(f"map {MAP_FILE}")
        else:
            self.tm.give_up()
            self.ack()
        self.finished = False
        race_time, state = self.wait_step()
        while race_time > 0:
            self.ack()
            race_time, state = self.wait_step()
        while race_time < 0:
            self.ack()
            race_time, state = self.wait_step()
        return race_time, state


def trace(game: Game, route: Route, side: str, first: bool) -> Path:
    toward = side
    away = "right" if side == "left" else "left"
    out = OUT_DIR / f"{MAP_NAME}-{side}.csv"
    header = "race_time_ms,x,y,z,dx,dz,speed,contact,route_m\n"
    rows = []          # (race_time, csv line)
    route.reset()
    race_time, state = game.restart(first)
    # Which way the heading angle (atan2(dz, dx)) turns when steering left.
    start_heading = None
    left_sign = None
    last_contact = -10_000
    stuck_ref = (0, 0.0)         # (race time, route metres) progress is measured from
    stuck_count = {}             # route metre (rounded) -> times stuck there
    reversing_until = -1
    detour_until_m = -1.0
    flat_out_until_m = -1.0      # no speed cap until here: for jumps the tracer fell short of
    falls = {}                   # route metre (rounded to 20 m) -> falls there
    snapshots = []               # (race_time, route index, state) every SNAPSHOT_MS
    below_ref = None             # (race time, route metres) when it went below the road
    best_s = 0.0
    step = 0
    if first:
        hide_console(0)
    print(f"[trace] {MAP_NAME}: following the {side} wall", flush=True)
    while True:
        x, y, z = state.position
        vx, vy, vz = state.velocity
        rot = state.rotation_matrix
        dx, dz = rot[0][2], rot[2][2]
        heading = math.atan2(dz, dx)
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        contact = bool(state.scene_mobil.has_any_lateral_contact)
        airborne = not any(w.real_time_state.has_ground_contact for w in state.simulation_wheels)
        s = route.project(x, z)
        best_s = max(best_s, s)
        if airborne and left_sign is not None:
            # Take jumps and land along the route, not against a wall.
            detour_until_m = max(detour_until_m, s + AIRBORNE_DETOUR_M)

        # The end of the route is enough: the walls don't need a valid finish
        # (hugging a wall can miss a checkpoint the route flies through).
        at_end = s >= route.length - 5
        if game.finished or at_end or race_time >= MAX_RACE_MS:
            rows.append((race_time, f"{race_time},{x:.3f},{y:.3f},{z:.3f},{dx:.4f},{dz:.4f},{speed:.2f},{int(contact)},{s:.2f}"))
            break

        # Off the edge (a section with no wall on this side): go back in time
        # and drive past it along the route instead.
        # Below the road and not getting anywhere: fallen off. (Below the road
        # but moving along the route is a drop that's part of the track, like
        # A01's jump down to the finish.)
        below = y < route.height(s) - FALL_DROP_M
        if not below:
            below_ref = None
        elif below_ref is None:
            below_ref = (race_time, s)
        fell = below and race_time - below_ref[0] > FALL_MS and s - below_ref[1] < FALL_PROGRESS_M
        stuck_here = stuck_count.get(round(s / 10), 0) >= 3
        if (fell or stuck_here) and len(snapshots) > REWIND_SNAPSHOTS:
            # Each repeat at the same place rewinds further back and, from the
            # second, drives that stretch flat out (a jump needs the speed).
            n = falls[round(s / 20)] = falls.get(round(s / 20), 0) + 1
            # Restart from REWIND_BACK_M further back along the route each time.
            back_to = s - REWIND_BACK_M * (2 * n - 1)
            k = len(snapshots) - 1
            while k > 0 and route.cum[snapshots[k][1]] > back_to:
                k -= 1
            t0, index0, state0 = snapshots[k]
            del snapshots[k + 1:]
            if n >= 2:
                flat_out_until_m = max(flat_out_until_m, s + REWIND_DETOUR_M)
            rows = [r for r in rows if r[0] <= t0]
            route.index = index0
            s0 = float(route.cum[index0])
            detour_until_m = max(detour_until_m, s + REWIND_DETOUR_M)
            stuck_count.pop(round(s / 10), None)
            print(f"[trace]   {'fell off' if fell else 'stuck again'} at {s:.0f} m: rewinding to {t0 / 1000:.1f}s, "
                  f"following the route to {detour_until_m:.0f} m", flush=True)
            game.tm.rewind_to_state(state0)
            stuck_ref = (t0, s0)
            below_ref = None
            reversing_until = -1
            game.tm.set_input_state(left=False, right=False, accelerate=True, brake=False)
            game.ack()
            race_time, state = game.wait_step()
            continue

        rows.append((race_time, f"{race_time},{x:.3f},{y:.3f},{z:.3f},{dx:.4f},{dz:.4f},{speed:.2f},{int(contact)},{s:.2f}"))
        if not snapshots or race_time - snapshots[-1][0] >= SNAPSHOT_MS:
            if not below:
                snapshots.append((race_time, route.index, state))
                del snapshots[:-600]

        inputs = {"left": False, "right": False, "accelerate": True, "brake": False}
        if left_sign is None:
            # Calibrate: hold left briefly and see which way the heading moves.
            if start_heading is None:
                start_heading = heading
            if race_time < CALIBRATE_MS:
                inputs["left"] = True
            else:
                turned = wrap(heading - start_heading)
                left_sign = 1.0 if turned > 0 else -1.0
                print(f"[trace]   steering left turns the heading by {math.degrees(turned):+.1f} deg", flush=True)
        else:
            if contact:
                last_contact = race_time
            if s >= stuck_ref[1] + STUCK_PROGRESS_M or race_time < reversing_until:
                stuck_ref = (race_time, s)
            elif race_time - stuck_ref[0] > STUCK_MS:
                stuck_count[round(s / 10)] = stuck_count.get(round(s / 10), 0) + 1
                print(f"[trace]   stuck at {s:.0f} m: reversing, then following the route for {DETOUR_M:.0f} m", flush=True)
                reversing_until = race_time + REVERSE_MS
                detour_until_m = max(detour_until_m, s + DETOUR_M)
                stuck_ref = (race_time, s)

            if race_time < reversing_until:
                # Pinned: reverse, steering so the nose swings away from the wall.
                inputs = {"left": False, "right": False, "accelerate": False, "brake": True}
                inputs[toward] = True
            elif race_time - last_contact < AWAY_MS and s >= detour_until_m:
                inputs[away] = True
            else:
                # Aim ahead on the route, pushed sideways towards this wall.
                # Following the route: aim close ahead, but not so close that it
                # weaves at speed (a ramp needs all the speed it can get).
                look = max(DETOUR_LOOKAHEAD_M, 0.8 * speed) if s < detour_until_m else LOOKAHEAD_M
                ahead = route.point(s + look)
                behind = route.point(s + look - 5)
                tangent = math.atan2(ahead[1] - behind[1], ahead[0] - behind[0])
                normal = tangent + left_sign * (math.pi / 2) * (1 if side == "left" else -1)
                offset = 0.0 if s < detour_until_m else SIDE_OFFSET_M
                tx = ahead[0] + offset * math.cos(normal)
                tz = ahead[1] + offset * math.sin(normal)
                err = wrap(math.atan2(tz - z, tx - x) - heading)
                if abs(err) > HEADING_DEADBAND:
                    inputs["left" if err * left_sign > 0 else "right"] = True
            climbing = vy > 1.0
            flat_out = s < flat_out_until_m
            if speed > TARGET_SPEED_MS and not climbing and not flat_out and race_time >= reversing_until:
                inputs["accelerate"] = False
                if speed > TARGET_SPEED_MS + 8:
                    inputs["brake"] = True

        game.tm.set_input_state(**inputs)
        game.ack()
        race_time, state = game.wait_step()
        step += 1
        if step % 1000 == 0:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out.write_text(header + "\n".join(r[1] for r in rows) + "\n")
            print(f"[trace]   t={race_time / 1000:.1f}s pos=({x:.0f},{z:.0f}) speed={speed:.1f} "
                  f"route {s:.0f}/{route.length:.0f} m", flush=True)

    out.write_text(header + "\n".join(r[1] for r in rows) + "\n")
    contacts = sum(r[1].split(",")[7] == "1" for r in rows)
    result = "finished" if game.finished else ("reached the end of the route" if best_s >= route.length - 5 else "did NOT reach the end")
    print(f"[trace] {side} wall: {result} at {race_time / 1000:.1f}s, {len(rows)} rows "
          f"({contacts} touching the wall), route {best_s:.0f}/{route.length:.0f} m -> {out}", flush=True)
    return out


def main():
    route = Route(ROUTE_CSV)
    stage_map()
    (port,) = launch_games(1, "/home/wineuser/.wine-instances-trace", BASE_PORT, "tmloader-trace")
    game = Game(port)
    for i, side in enumerate(SIDES):
        trace(game, route, side, first=(i == 0))
    game.tm.close()


if __name__ == "__main__":
    main()
