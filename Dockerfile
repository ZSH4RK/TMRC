FROM zsh4rk/trackmania_rl_framework:1.0.0

COPY agent /agent

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]