from python:3.11-slim

shell ["/bin/bash", "-o", "pipefail", "-c"]

# uv, curl, git
copy --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

run apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# gooseuser, goosegroup
run groupadd -g 1000 goosegroup && \
    useradd -u 1000 -g goosegroup -m -s /bin/bash gooseuser

run mkdir /workspace && chown gooseuser:goosegroup /workspace

user gooseuser
workdir /workspace

# goose
run curl -fsSL https://github.com/block/goose/releases/latest/download/download_cli.sh | CONFIGURE=false bash

env PATH="/home/gooseuser/.local/bin:${PATH}"
env PYTHONUNBUFFERED=1

entrypoint ["bash", "-c"]
