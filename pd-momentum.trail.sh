if [ -z "$1" ]; then
  RESULTS_DIR="pd-momentum-results-$(date +%Y%m%d-%H%M%S)"
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
  uv run vertex-ai-main.py pd-momentum.ipynb \
  --csv "$RESULTS_DIR/metrics_year={1}_hold={2}_mom={3}_mcap={4}_long={5}_short={6}.csv" \
  --html "$RESULTS_DIR/metrics_year={1}_hold={2}_mom={3}_mcap={4}_long={5}_short={6}.html" \
  --start_year {1} \
  --holding_days {2} \
  --momentum_days {3} \
  --min_mcap {4} \
  --long_decile {5} \
  --short_decile {6} \
  ::: 2020 \
  ::: 1 3 7 14 21 30 \
  ::: 7 14 30 \
  ::: 1000000 \
  ::: 8 9 \
  ::: -1 \
