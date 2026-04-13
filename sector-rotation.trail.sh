#!/bin/bash

set -euo pipefail

for cmd in uv xsv; do
  if ! command -v "$cmd" &> /dev/null; then
    echo "$cmd could not be found. Please install $cmd to run this script."
    exit 1
  fi
done

# best single parameter set:
PARAMETER_SETS="
  --signal mom1m
  --signal mom2m
  --signal mom3m
  --signal mom6m
  --signal mom12m
  --signal mom12-1m-a *
  --signal mom12-1m-b

  --period 21 *
  --period 30
  --period 42
  --period 60

  --stop_long 0.05
  --stop_long 0.1
  --stop_long 0.2 *
  --stop_long 0.3
  --stop_long 0.4
  --stop_long 0.5

  --stop_short 0.05 *
  --stop_short 0.1
  --stop_short 0.2
  --stop_short 0.3
  --stop_short 0.4
  --stop_short 0.5

  --max_long 1 --max_short 0
  --max_long 2 --max_short 0 *
  --max_long 3 --max_short 0

  --max_long 0 --max_short 1
  --max_long 0 --max_short 2
  --max_long 0 --max_short 3
"""


# best number of long and long stops
PARAMETER_SETS="""
  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 2 --max_short 0 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 2 --max_short 1 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 3 --max_short 0 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 3 --max_short 1 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 4 --max_short 0 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 4 --max_short 1 --hard_stop_long 1 --hard_stop_short 1 *

  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 5 --max_short 0 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 5 --max_short 1 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.3 --stop_short 0.05 --max_long 1 --max_short 1 --hard_stop_long 1 --hard_stop_short 1 (*)
  --signal mom12-1m-a --period 21 --stop_long 0.3 --stop_short 0.05 --max_long 2 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.3 --stop_short 0.05 --max_long 3 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.3 --stop_short 0.05 --max_long 4 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.3 --stop_short 0.05 --max_long 5 --max_short 1 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 1 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 2 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 3 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 5 --max_short 1 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.5 --stop_short 0.05 --max_long 1 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.5 --stop_short 0.05 --max_long 2 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.5 --stop_short 0.05 --max_long 3 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.5 --stop_short 0.05 --max_long 4 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.5 --stop_short 0.05 --max_long 5 --max_short 1

  --signal mom12-1m-a --period 21 --stop_long 1 --stop_short 1 --max_long 2 --max_short 0 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 1 --stop_short 1 --max_long 2 --max_short 1 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 1 --stop_short 1 --max_long 3 --max_short 0 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 1 --stop_short 1 --max_long 3 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
"""

# Trails: number short and stops: 0.05, 1-2 short, one or four long
PARAMETER_SETS="""
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 1 --max_short 1 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 1 --max_short 2 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 3 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 1 --max_short 3 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 4 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 1 --max_short 4 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.1 --max_long 4 --max_short 1 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.1 --max_long 1 --max_short 1 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.1 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.1 --max_long 1 --max_short 2 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.1 --max_long 4 --max_short 3 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.1 --max_long 1 --max_short 3 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.1 --max_long 4 --max_short 4 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.1 --max_long 1 --max_short 4 --hard_stop_long 1 --hard_stop_short 1
"""

# Trail: signals, period: 12-1m-a, 21 is best
PARAMETER_SETS="""
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-b --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12m --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1
  --signal mom6m --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-b --period 30 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12m --period 30 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1
  --signal mom6m --period 30 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-b --period 40 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12m --period 40 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1
  --signal mom6m --period 40 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1

  --signal mom12-1m-b --period 14 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1
  --signal mom12m --period 14 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1
  --signal mom6m --period 14 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1
"""

PARAMETER_SETS="""
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1 --leverage 1
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1 --leverage 1.5
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1 --leverage 2
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1 --leverage 2.5
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1 --leverage 3
"""

PARAMETER_SETS="""
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1 --leverage 2

  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1 --leverage 2
  --signal mom12-1m-a --period 21 --stop_long 0.3 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1 --leverage 2
  --signal mom12-1m-a --period 21 --stop_long 0.4 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 1 --hard_stop_short 1 --leverage 2

  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 0.5 --hard_stop_short 1 --leverage 2
  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 0.4 --hard_stop_short 1 --leverage 2
  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 0.3 --hard_stop_short 1 --leverage 2
  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 0.2 --hard_stop_short 1 --leverage 2

  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 0.1 --hard_stop_short 1 --leverage 2 *
  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 0.1 --hard_stop_short 1 --leverage 2.5
  --signal mom12-1m-a --period 21 --stop_long 0.2 --stop_short 0.05 --max_long 4 --max_short 2 --hard_stop_long 0.1 --hard_stop_short 1 --leverage 3
"""

OUTPUT_FILE="sector-rotation.trail.csv"
if [ "$#" -gt 0 ]; then
  OUTPUT_FILE="$1"
fi


echo > "$OUTPUT_FILE"

# filter out empty lines from PARAMETER_SETS
echo "$PARAMETER_SETS" | grep -v '^[[:space:]]*$' | while IFS= read -r PARAMETER_SET; do
  echo "Running with parameters: $PARAMETER_SET"

  t=$(mktemp)
  uv run vertex-ai-main.py sector-rotation.ipynb --html /dev/null --csv "$t" $PARAMETER_SET
  
  r=$(mktemp)
  xsv cat rows "$t" "$OUTPUT_FILE" | xsv sort -s ts > "$r"
  mv "$r" "$OUTPUT_FILE"

  rm "$t"
done
