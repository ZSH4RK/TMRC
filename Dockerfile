FROM zsh4rk/trackmania_rl_framework:rendering-1.0.0 AS base

USER root

RUN mkdir -p /opt/venv
RUN apt update
RUN apt install -y python3-pip python3-dev python3-venv
RUN python3 -m venv /opt/venv
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install numpy tminterface

FROM base

COPY agent /agent
COPY entrypoint.sh /entrypoint.sh

RUN chmod 777 /entrypoint.sh

USER wineuser

ENV TMNF_DIR=/home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever

#CMD ["/usr/bin/sleep", "600"]

ENTRYPOINT ["/entrypoint.sh"]