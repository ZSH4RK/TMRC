FROM zsh4rk/trackmania_rl_framework:1.0.0 AS base

USER root

RUN mkdir -p /opt/venv
RUN apt update
RUN apt install -y python3-pip python3-dev python3-venv
RUN python3 -m venv /opt/venv
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install numpy tminterface

# Fix Python_Link.as: the upstream version defers Init_Socket() to a simulation
# CommandList callback that never fires while in menus. Revert to the direct
# approach so the TCP socket starts immediately when the plugin loads.
COPY plugins/Python_Link.as /home/wineuser/.wine/drive_c/users/wineuser/Documents/TMInterface/Plugins/Python_Link.as
RUN chown wineuser:wineuser /home/wineuser/.wine/drive_c/users/wineuser/Documents/TMInterface/Plugins/Python_Link.as

# Patch TMLoader UI to auto-launch the default profile on startup (no button click required).
COPY plugins/index.jsx /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever/ui/layouts/default/index.jsx
RUN chown wineuser:wineuser /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever/ui/layouts/default/index.jsx

# Pass A01-Race challenge path as args so TmForever auto-starts that race on launch.
COPY plugins/default.yaml /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever/database/TmForever/profiles/default.yaml
RUN chown wineuser:wineuser /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever/database/TmForever/profiles/default.yaml

FROM base

COPY agent /agent
COPY entrypoint.sh /entrypoint.sh

RUN chmod 777 /entrypoint.sh

USER wineuser

ENV TMNF_DIR=/home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever

ENTRYPOINT ["/entrypoint.sh"]