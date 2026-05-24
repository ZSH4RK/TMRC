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
- Base image: `zsh4rk/trackmania_rl_framework:rendering-1.0.0` (pre-configured with TMNF + Wine)
- Sets up a Python 3 virtual environment at `/opt/venv`
- Installs: `numpy`, `tminterface`
- Creates wine user context with `TMNF_DIR=/home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever`
- Runs `/entrypoint.sh` as entry point

### Entrypoint Script (entrypoint.sh)
- Starts Xvfb (headless X11 server) on display :99
- Waits 15 seconds (for TMNF to initialize?)
- Runs `/agent/agent.py`
- **Issue**: Doesn't explicitly start TMNF or TMInterface

### Agent Code (agent/agent.py)
- Spawns TMNF via `wine TMLoader.exe` using strace in a background thread
- Waits 15 seconds, then prints netstat output
- Connects to TMInterface on `localhost:8775` (changed from 8477)
- Loads map "A01-Race.Challenge.Gbx"
- Implements basic "always accelerate" controller
- Uses custom `tminterface2.py` wrapper for socket-based communication

### Communication Layer (agent/tminterface2.py)
- Custom socket-based implementation of TMInterface 2.1.4 protocol
- Designed to replicate the original Donadigo Python client for TMInterface 1.4.3
- Handles message serialization/deserialization via struct packing
- Provides methods to:
  - Get simulation state (position, velocity, contact data, checkpoints)
  - Set input state (steering, acceleration, braking)
  - Execute commands (e.g., map loading)
  - Request/get frame data

## Issues & Questions

1. **TMNF Startup**: Does the base image auto-start TMNF+TMInterface, or does agent.py handle it?
   - entrypoint.sh just waits 15 seconds
   - agent.py spawns wine TMLoader.exe, which may or may not start TMInterface listening
   - Need to verify TMInterface is actually listening on port 8775

2. **Port Mismatch**: 
   - agent.py uses port 8775
   - config_files/user_config.py has base_tmi_port = 8478
   - These should be synchronized

3. **Executable Change**:
   - Changed from TmForever.exe → TMLoader.exe
   - Need to verify TMLoader.exe actually launches both TMNF and TMInterface

4. **Wine/X11 Setup**:
   - Uses strace to trace wine execution (adds overhead, for debugging?)
   - Relies on Xvfb for headless display
   - Need to verify Wine can access the display properly

5. **Missing Elements**:
   - No config_files/state_normalization.py content reviewed yet
   - No config_files/inputs_list.py content reviewed yet
   - The config_copy.py suggests this was a complex ML training setup, but current agent.py is minimal

## Improvements Made (2026-05-24)

### 1. Fixed entrypoint.sh
- Now exports `DISPLAY=:99` environment variable (was missing)
- Tracks Xvfb PID and kills it on exit (was orphaning process)
- Improved structure with clear error handling

### 2. Improved agent.py startup sequence
- Removed strace wrapper (adds 30%+ overhead, was for debugging)
- Added `wait_for_tminterface()` function that:
  - Polls TCP port 8775 every 1 second
  - Waits up to 60 seconds for TMInterface to become available
  - Provides clear feedback on success/failure
- Replaced fixed 15-second wait with active port polling
- Added verbose logging with ✓ progress checkmarks
- Better error messages on connection failure

### 3. Fixed tminterface2.py imports
- Removed dependency on config_copy (was importing unnecessary ML training config)
- Now uses `sys.platform` directly to detect Linux vs Windows
- Cleaner imports, fewer dependencies

### 4. Enhanced main() function
- Proper try/except handling for connection errors
- Prints step counter every 100 steps for progress feedback
- Handles KeyboardInterrupt gracefully
- Proper cleanup on exit

## Completed Work Summary

### ✓ Infrastructure Improvements
1. Fixed entrypoint.sh X11 setup (export DISPLAY, proper cleanup)
2. Improved error handling throughout agent.py and shell scripts
3. Removed debugging overhead (strace wrapper)
4. Added active port polling instead of fixed waits
5. Fixed tminterface2.py to avoid config_copy dependency
6. Docker image builds successfully

### ✓ Investigation Completed
1. Confirmed TMNF binary exists and Wine can execute it
2. Confirmed TMNF starts and initializes graphics (llvmpipe)
3. Located Python_Link.as plugin (socket server implementation)
4. Found TMInterface config.txt location and proper format
5. Verified base image has TMInterface files (versions 2.1.0, 2.1.1, 2.2.0)

### ✗ Known Blocker
**TMInterface not listening on port 8775** - despite proper configuration:
- Config file created with correct INI-style format: `[Python_Link]` + `custom_port=8775`
- TMNF starts but Python_Link plugin doesn't load or doesn't listen
- No log output indicating plugin initialization
- No evidence of socket server starting

## Remaining Issues to Resolve

**Primary Blocker:** Why doesn't Python_Link.as execute when TMNF runs?

Possible causes:
1. TMNF doesn't auto-load modifications in this environment
2. TMInterface modification not properly installed or initialized
3. Python_Link plugin fails silently on startup
4. TMNF requires user interaction or profile setup before mods load
5. Config variable format/reading might need additional settings

## Recommended Next Steps

1. **Check TMNF modification loading:**
   - Verify modifications are enabled in TMNF settings
   - Check if TMInterface modification appears in game's mod list
   - Look for modification loading logs in Wine output

2. **Debug Python_Link execution:**
   - Add debug flag to Python_Link.as to enable verbose logging
   - Check if TMNF creates any error logs when plugins load
   - Try running TMNF in developer mode or with plugin directory specified

3. **Alternative approaches:**
   - Research if there's a command-line flag to enable TMInterface explicitly
   - Check if zsh4rk/trackmania_rl_framework has specific launch documentation
   - Consider if different TMNF executable (TmForever.exe vs TMLoader.exe) matters

4. **Testing checklist when blocker resolved:**
   - [ ] Port 8775 becomes available within 60 seconds
   - [ ] Agent connects successfully via TCP socket
   - [ ] Map load command accepted by game
   - [ ] Simulation steps received (SC_RUN_STEP_SYNC messages)
   - [ ] Agent can set input state
   - [ ] Container runs reliably (multiple restarts)

# Claude