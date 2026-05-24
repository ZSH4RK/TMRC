# Task:
Read through all files to understand the codebase.

Figure out how to get Trackmania and TMInterface to run on linux inside Docker

Complete the goal in `CLAUDE.md`

Write observations in this file

# Observations

## Project Overview
TMRC is a competition for AI bots competing in Trackmania Nations Forever (TMNF). The system:
- Uses TMInterface 2.1.4 to communicate with TMNF via TCP sockets
- Runs agents as Python scripts that receive game state and send inputs
- Targets Docker containerization for distributed execution

## Current Architecture

### Docker Setup (Dockerfile)
- Base image: `zsh4rk/trackmania_rl_framework:1.0.0` (non-rendering variant)
- Sets up a Python 3 virtual environment at `/opt/venv`
- Installs: `numpy`, `tminterface`
- Patches three plugin/config files (see Plugin Patches below)
- Runs `/entrypoint.sh` as entry point

### Entrypoint Script (entrypoint.sh)
- Starts Xvfb (headless X11 server) on display :99 (1024x768x16)
- Tracks Xvfb PID and kills it on exit
- Runs `/agent/agent.py` with `PYTHONUNBUFFERED=1`

### Agent Code (agent/agent.py)
- Spawns TMLoader.exe via Wine in a background thread
- Polls TCP port 8775 (up to 60s) for TMInterface readiness
- After port is up, waits 8 seconds for the menu to render, then auto-navigates menus
- Connects to TMInterface and enters the simulation loop
- Implements "always accelerate" controller (placeholder for real agent)
- Uses custom `tminterface2.py` wrapper for socket-based communication

### Plugin Patches (plugins/)
Three files are copied into the container at build time:

**plugins/Python_Link.as** — Patched AngelScript plugin
- Original upstream version defers `Init_Socket()` to a CommandList callback that never fires while in menus
- Patched version calls `Init_Socket()` directly from `Main()` so the TCP server starts immediately when the plugin loads
- Port is read from `custom_port` variable (default 8775)

**plugins/index.jsx** — Patched TMLoader UI
- TMLoader normally shows a GUI requiring a button click to launch the game
- Patched version adds `autoLaunch()` which runs on document load, automatically starting the default profile without any user interaction

**plugins/default.yaml** — TMLoader profile config
- Configures TmForever to launch with mods: TMInterface 2.1.1, TMUnlimiter, CoreMod 1.0.3
- Passes `GameData\Tracks\Campaigns\Nations\White\A01-Race.Challenge.Gbx` as a command-line arg to TmForever, which causes the game to navigate to the White campaign on startup

### Communication Layer (agent/tminterface2.py)
- Socket-based implementation of TMInterface 2.1.4 protocol
- Handles message serialization/deserialization via struct packing
- Provides methods for get simulation state, set input state, execute commands

## How It All Works Together (Full Stack)

1. **Container starts** → `entrypoint.sh` launches Xvfb on :99
2. **agent.py starts** → spawns `wine TMLoader.exe` in a background thread
3. **TMLoader** reads `default.yaml`, calls the patched `autoLaunch()` from `index.jsx`, and launches TmForever with mods injected
4. **TmForever** loads with TMInterface.dll, TMUnlimiter.dll, CoreMod.dll injected
5. **Python_Link.as** runs its patched `Main()`, calls `Init_Socket()` → TCP server on port 8775
6. **agent.py** polls port 8775, detects it listening, waits 8 more seconds for menu to fully render
7. **navigate_to_race()** runs: finds TmForever window, clicks Play Solo → closes TMInterface console → clicks White → double-clicks A01-Race
8. **Race starts** → TmForever enters LocalRace state, OnRunStep callbacks fire
9. **agent.py main()** connects to TMInterface, receives SC_RUN_STEP_SYNC messages, sends input state

## Key Findings & Debugging Notes

### Why TMLoader instead of TmForever.exe directly?
TMLoader is a Sciter-based launcher that performs DLL injection of TMInterface.dll, TMUnlimiter.dll, and CoreMod.dll into TmForever.exe. Running TmForever.exe directly would bypass all mods.

### Why is Python_Link.as patched?
The upstream Python_Link.as registers a CommandList command that calls `Init_Socket()`. CommandList commands only fire during race simulation — not in menu state. Since the agent connects before navigating to a race, the socket would never start. The patch calls `Init_Socket()` directly in `Main()` instead.

### Why is index.jsx patched?
TMLoader's UI shows a "Play" button that must be clicked to launch the game. In a headless container there is no human to click it. The patch adds an `autoLaunch()` async function that fires on page load and calls `profile.prepare()` + `proc.async_start()`.

### Menu navigation: why xdotool instead of TMInterface commands?
TMInterface commands (like `map`, `replay`) only work when already in a race state. From the menu, there is no API to load a challenge — the game must be navigated via UI. `map` is not a valid command at all (not in TMInterface DLL string table). The xdotool approach sends synthetic X11 events to the TmForever window to click through the menus.

### xdotool coordinate system
- TmForever window is at absolute screen position (4, 30), size 640x480, no WM decorations (Xvfb has no window manager)
- Play Solo: window coords (66, 70) = screen (70, 100)
- White campaign: absolute screen (67, 101) — used without `--window` so event goes to top-level window at that position
- A01-Race thumbnail: absolute screen (213, 190) — same reason

### TMInterface console overlay
The TMInterface console appears as an overlay window on top of the game. Using `xdotool click --window <tm_win_id>` sends events directly to TmForever bypassing the overlay. However, for the White/A01-Race clicks, absolute screen coordinates are used instead, which requires the console to not be covering those positions (it starts at screen y≈232, and the tracks are at y≈155-230).

### "Not connected to online account" dialog
TMNF sometimes shows this dialog on startup when it can't reach the master server. Pressing Enter from the TmForever window dismisses it. The navigate_to_race() function handles this by focusing the window first and waiting before clicking menus.

### builtin.txt settings
Key TMInterface settings in effect:
- `skip_map_load_screens true` — skips "Press any key" and choose-opponent screens
- `custom_port 8775` — Python_Link TCP server port
- `plugin_python_link_enabled true` — enables the Python_Link plugin
- `execute_commands true` — allows console commands
- `sim_speed 5` — simulation runs at 5x real-time speed

## Completed Checklist

- [x] Docker image builds successfully
- [x] Container starts without crashing
- [x] TMInterface TCP socket listens on port 8775
- [x] Agent connects to TMInterface
- [x] A01-Race map loads automatically (via menu navigation)
- [x] Agent receives SC_RUN_STEP_SYNC simulation steps
- [x] Agent can set input state (always-accelerate demo works)
- [x] Container runs reliably from cold start with no manual intervention

## Remaining Work

- [ ] Implement actual RL/AI agent logic instead of always-accelerate placeholder
- [ ] Handle race finish / auto-restart for training loops
- [ ] Expose game state (position, velocity, checkpoints) to agent properly
- [ ] Make map selection configurable (not hardcoded to A01-Race)
- [ ] Consider making navigation more robust (retry logic if clicks miss)
