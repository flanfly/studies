#!/bin/bash

set -e
set -o pipefail

RFC3339=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
FILE="polarity-${RFC3339}"
SAFE_FILE=$(echo "${FILE}" | tr ':' '_')

echo "Starting polarity data update at ${RFC3339}"

uv run update-polarity.py \
  --output "${SAFE_FILE}.parquet" \
  --log-level DEBUG \
  --log-file "${SAFE_FILE}.log"

rm -f latest-data.parquet
ln -sf "${SAFE_FILE}.parquet" "latest-data.parquet"

