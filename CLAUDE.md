# TMRC Goals

## Primary Goal
Get Trackmania Nations Forever (TMNF) with TMInterface running in a Docker container on Linux, where:
1. The Docker image successfully builds
2. A Python agent can connect to TMInterface via TCP socket
3. The agent can load a map and receive game state updates
4. The agent can send input commands (steering, acceleration, braking) to the running game

## Success Criteria
- [x] Code refactored for robustness and better error handling
- [ ] Docker image builds without errors
- [ ] Container starts without crashing
- [ ] TMInterface TCP socket is listening on port 8775
- [ ] Agent connects successfully to TMInterface
- [ ] Agent loads A01-Race map without errors
- [ ] Agent receives at least one step sync message from TMNF
- [ ] Agent can set input state and advance game simulation
- [ ] Container can be run multiple times reliably

## Work Completed (2026-05-24)

### Code & Configuration Improvements
1. **entrypoint.sh**: Fixed missing DISPLAY export, proper Xvfb setup and cleanup
2. **agent.py**: 
   - Replaced fixed 15-second wait with active TCP port polling (up to 60s timeout)
   - Removed strace wrapper (reduced overhead)
   - Added detailed logging and startup verification
   - Improved error handling and graceful shutdown
3. **tminterface2.py**: Removed hardcoded config_copy import, use sys.platform directly
4. **Dockerfile**: 
   - Changed to base image 1.0.0 (non-rendering)
   - Added TMInterface config file setup with custom_port=8775
   - Proper INI-style format matching TMNF conventions

### Investigation & Findings
- ✓ Docker image builds successfully (10.9GB)
- ✓ Xvfb X server initializes and is accessible
- ✓ Wine can execute TMLoader.exe (TMNF)
- ✓ TMNF starts and initializes (uses software rendering via llvmpipe)
- ✓ Python_Link.as plugin exists and implements socket server
- ✗ **BLOCKING ISSUE**: TMInterface doesn't listen on port 8775 despite configuration
  - Plugin appears not to load when TMNF starts
  - No log evidence of Python_Link initialization
  - TMNF runs but modifications may not be enabled

### Commits Made
1. 2da1210: "Improve Docker startup robustness and error handling"
2. 6a8c9b1: "Add debugging and improve Docker configuration for TMInterface"
3. d797834: "Fix TMInterface config file format to INI-style"

## Current Status: BLOCKED ❌
Cannot proceed to testing agent connectivity until Python_Link plugin loads and listens on port 8775.

## Critical Blocker
**Why doesn't TMNF load the Python_Link plugin?**
- TMNF starts successfully in container
- Python_Link.as file exists at correct location
- Config file created with proper format
- But socket server never starts listening
- Hypothesis: TMNF might not load modifications by default in this environment

## Next Actions Required
1. **Investigate TMNF modification loading:**
   - Check if modifications are enabled in TMNF user config
   - Verify TMInterface appears in game's modification list
   - Look for plugin initialization errors in Wine/game logs

2. **Debug Python_Link execution:**
   - Enable verbose logging in Python_Link.as script
   - Check for TMNF error/debug output files
   - Try explicit plugin/modification directory specifications

3. **Determine proper TMNF initialization:**
   - Research if TMInterface needs explicit launch command
   - Check if there's a specific TMNF mode/profile for headless use
   - Review zsh4rk/trackmania_rl_framework documentation

## Architecture Notes
- Base image: `zsh4rk/trackmania_rl_framework:1.0.0`
- Protocol: TMInterface via TCP socket (port 8775)
- Game: TMNF (TrackMania Nations Forever) via Wine
- Display: Xvfb headless X server on :99
- Agent: Simple Python controller implementing "always accelerate" for testing
