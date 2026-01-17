#!/bin/bash

set -euo pipefail

uv export --format requirements.txt --output-file requirements.txt --no-dev

docker buildx build --platform linux/amd64 -t ghcr.io/flanfly/studies:latest -t studies:latest .

#trivy image --scanners secret --severity CRITICAL,HIGH --skip-dirs "/etc,/var,/usr,/lib,/bin" ghcr.io/flanfly/studies:latest

docker push ghcr.io/flanfly/studies:latest

docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/flanfly/studies
