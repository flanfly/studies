#from nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
from us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu124.py310

label org.opencontainers.image.source=https://github.com/flanfly/studies

run apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    git-crypt \
    parallel \
    r-base \
    r-base-dev \
    libicu-dev \
    libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*

copy --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

env PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

workdir /app

run uv python install 3.10

copy pyproject.toml uv.lock ./

run uv sync --frozen --no-dev --no-install-project

copy . .

entrypoint ["/app/run.sh"]
