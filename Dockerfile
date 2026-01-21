from nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

label org.opencontainers.image.source=https://github.com/flanfly/studies

copy --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
env PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

workdir /app

run uv python install 3.10

copy pyproject.toml uv.lock ./

run uv sync --frozen --no-dev --no-install-project

copy . .

entrypoint ["uv", "python", "-v"]
