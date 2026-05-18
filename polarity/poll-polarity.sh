#!/bin/bash

set -e
set -o pipefail

RFC3339=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
PARQUET_DIR="polarity-$RFC3339-output"
SAFE_PARQUET_DIR=$(echo "${PARQUET_DIR}" | tr ':' '_')

echo "Starting polarity data update at $RFC3339"

if [ -s "IDTOKEN" ]; then
  uv run update-polarity.py \
    --output-dir "${SAFE_PARQUET_DIR}" \
    --log-level DEBUG \
    --log-file "update-polarity-$RFC3339.log" \
    --idtoken "$(cat IDTOKEN)"
else
  uv run update-polarity.py \
    --output-dir "${SAFE_PARQUET_DIR}" \
    --log-level DEBUG \
    --log-file "update-polarity-$RFC3339.log"
fi

rm latest-data
ln -sf "${SAFE_PARQUET_DIR}" "latest-data"

