# 1. The big picture

This page explains what each piece is and how they talk to each other. There's
no code here.

## The pieces

```
 ┌──────────────────────── Docker container ────────────────────────┐
 │                                                                  │
 │   TrackMania (the game)                                          │
 │      └─ TMInterface (a mod that lets programs control the game)  │
 │            └─ Python_Link plugin ──── network socket ───┐        │
 │                                                         │        │
 │   Your Python code  ◄───────────────────────────────────┘        │
 │      └─ the framework records everything into results/ ──────────┼──► TMRC/results/ on your computer
 │                                                                  │
 │   noVNC (lets you watch the game in a web browser) ──────────────┼──► http://localhost:6080
 └──────────────────────────────────────────────────────────────────┘
```

- **TrackMania Nations Forever (TMNF)** is the racing game. It's a Windows game,
  so it runs through **Wine** (software that runs Windows programs on Linux).
- **TMInterface** is a mod for the game. It can pause the game, read the car's
  exact position and speed, and press keys.
- **The Python_Link plugin** connects TMInterface to Python over a network socket.
  You never need to touch it.
- **Docker** packages all of this so it runs the same way on every computer.
  The setup is fiddly, and Docker means you don't have to repeat it.

## What happens while the car drives

The game runs in small time slices. Every **50 milliseconds of game time** (a
**step**), this happens:

1. The game **pauses** and sends the car's state (position, speed, direction) to Python.
2. Python works out the **observation**: what the car can "see". Here that's
   distances to the walls in 7 directions, plus speed and direction.
3. **Your code decides** which keys to press: left, right, accelerate, brake.
4. Python sends the keys back and the game runs the next 50ms.

The game waits for your decision every step. A slow decision makes the run
take longer in real time, but the driving itself comes out exactly the same.

An **episode** is one attempt at the track, from the start line until one of these happens:

- **finished**: crossed the finish line
- **timeout**: 120 seconds of game time passed
- **stuck**: the car made less than 1 metre of progress in 3 seconds

Then the car restarts at the start line and the next episode begins.

## The game is deterministic, and that matters a lot

If you press **exactly the same keys at exactly the same times**, the car does
**exactly the same thing**, every time. There's no randomness in TrackMania's physics.

That means:
- A recording of the keys (a **replay**) is enough to reproduce a run perfectly.
- A driver with no randomness in its decisions gets the same result every
  episode. If it doesn't, something is wrong with the setup, which is useful to know.
- Comparisons between drivers can be fair and exact.

## Speed

The game normally runs at real-time speed, but the framework speeds it up (5×
by default) so you can test faster. It can also run several copies of the game
at once ("**instances**"), each on its own track, to collect more experience
in the same time.

## What gets recorded

Every time you run anything, the framework creates a folder in `TMRC/results/`
called a **run**. It holds:

- what code and settings were used, so the run can be repeated later
- one line per episode: did it finish, how fast, how far
- replays of the best attempts
- for learning methods, how learning went over time

[Page 6](06-results-and-replays.md) goes through it in detail.

## Where to go next

[2. Setup and your first run](02-setup-and-first-run.md)
