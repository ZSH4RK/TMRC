#!/bin/bash
set -e

# Headless display
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x16 &
XVFB_PID=$!
sleep 2

# Start TrackMania + agent
# The agent will handle starting TMNF and connecting to TMInterface
python3 "/agent/agent.py"
AGENT_EXIT=$?

# Cleanup
kill $XVFB_PID 2>/dev/null || true

exit $AGENT_EXIT
