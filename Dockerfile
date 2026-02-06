FROM ubuntu:latest
LABEL authors="iamac"

ENTRYPOINT ["top", "-b"]