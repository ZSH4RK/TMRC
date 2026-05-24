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

## Recent Work (2026-05-24)

**Improvements made:**
1. Fixed missing DISPLAY export in entrypoint.sh
2. Replaced fixed 15-second wait with active TCP port polling (up to 60s)
3. Removed strace wrapper (debugging overhead)
4. Improved error handling and logging throughout
5. Removed hardcoded config file imports, use sys.platform instead
6. Commit: 2da1210 "Improve Docker startup robustness and error handling"

**Current status:** Code improvements complete, ready for Docker testing

## Implementation Notes
- Base image `zsh4rk/trackmania_rl_framework:rendering-1.0.0` should have TMNF pre-configured
- Must use Wine to run Windows TMNF binary (TMLoader.exe) on Linux
- Must use Xvfb for headless display (no physical monitor)
- TMInterface port 8775 is used for TCP communication
- agent.py currently implements a simple "always accelerate" controller for testing

## Critical Path - Next Steps
1. Test Docker build to verify syntax and dependencies
2. Run container and monitor startup sequence for failures
3. Debug any port 8775 or TMInterface availability issues
4. Verify agent can connect and enter game loop
