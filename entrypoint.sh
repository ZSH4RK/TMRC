#!/bin/bash
set -e

# Headless display
Xvfb :99 -screen 0 1024x768x16 &
sleep 2

# Start TrackMania
# Wait for TMNF + TMInterface to load and enter a map
sleep 15

# Start agent
python3 "/agent/agent.py"
