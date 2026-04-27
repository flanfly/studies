if [ -z "$1" ]; then
  RESULTS_DIR="cs-crypto-momentum-results-$(date +%Y%m%d-%H%M%S)"
  echo "No results directory specified. Using: $RESULTS_DIR"
  mkdir -p "$RESULTS_DIR"
else
  RESULTS_DIR="$1"
  mkdir -p "$RESULTS_DIR"
fi

parallel \
  --results "$RESULTS_DIR" \
  --bar \
  --verbose \
  --joblog "$RESULTS_DIR"/job.log \
  uv run vertex-ai-main.py cs-crypto-momentum.ipynb \
  --csv "$RESULTS_DIR/metrics_{1}_{2}_{3}_{4}_{5}.csv" \
  --html "$RESULTS_DIR/metrics_{1}_{2}_{3}_{4}_{5}.html" \
  --days_holding {1} \
  --n_buckets {2} \
  --volume_cutoff {3} \
  --days_momentum {4} \
  --ema_slow {5} \
  --start_date 2023-01-01 \
  ::: 1 3 7 14 \
  ::: 4 5 7 10 \
  ::: 100_000 1_000_000 5_000_000 10_000_000 100_000_000 \
  ::: 14 20 30 40 50 \
  ::: 40 50 60 80 100 120

