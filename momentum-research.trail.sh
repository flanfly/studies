if [ -z "$1" ]; then
  RESULTS_DIR="momentum-research-results-$(date +%Y%m%d-%H%M%S)"
  echo "No results directory specified. Using: $RESULTS_DIR"
  mkdir -p "$RESULTS_DIR"
else
  RESULTS_DIR="$1"
  mkdir -p "$RESULTS_DIR"
fi

parallel \
  --results "$RESULTS_DIR" \
  --bar \
  --joblog "$RESULTS_DIR"/job.log \
  uv run vertex-ai-main.py momentum-research.ipynb \
  --csv "$RESULTS_DIR/metrics_{1}_{2}_{3}.csv" \
  --html "$RESULTS_DIR/metrics_{1}_{2}_{3}.html" \
  --days_holding {1} \
  --top_percentile {2} \
  --bottom_percentile {3} \
  ::: 1 3 7 14 20 30 60 \
  ::: 0.5 0.75 0.9 0.95 0.99 \
  ::: 0.5 0.25 0.1 0.05 None
