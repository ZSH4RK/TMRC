#!/bin/bash
set -e

# Headless display with debugging
export DISPLAY=:99
echo "Starting Xvfb on display :99"
Xvfb :99 -screen 0 1024x768x16 -nolisten tcp &
XVFB_PID=$!
echo "Xvfb PID: $XVFB_PID"
sleep 3

# Verify X server is running
if ! ps -p $XVFB_PID > /dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
echo "Xvfb is running"

# Test X11 connection
if ! xdpyinfo -display :99 > /dev/null 2>&1; then
    echo "WARNING: xdpyinfo check failed, continuing anyway"
fi

# Start TrackMania + agent
# The agent will handle starting TMNF and connecting to TMInterface
echo "Starting agent..."
python3 "/agent/agent.py"
AGENT_EXIT=$?

# Cleanup
echo "Cleaning up Xvfb (PID: $XVFB_PID)"
kill $XVFB_PID 2>/dev/null || true
wait $XVFB_PID 2>/dev/null || true

exit $AGENT_EXIT
