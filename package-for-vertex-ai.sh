#!/bin/bash

set -euo pipefail

uv export --format requirements.txt --output-file requirements.txt --no-dev

docker buildx build --platform linux/amd64 -t asia-southeast1-docker.pkg.dev/prj-vertexai-test/default/myimage:latest --push .
