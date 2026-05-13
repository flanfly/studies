if [ -z "$1" ]; then
  RESULTS_DIR="momentum-research-backtest-results-$(date +%Y%m%d-%H%M%S)"
  echo "No results directory specified. Using: $RESULTS_DIR"
  mkdir -p "$RESULTS_DIR"
else
  RESULTS_DIR="$1"
  mkdir -p "$RESULTS_DIR"
fi

cp momentum-research-backtest.ipynb "$RESULTS_DIR/input.ipynb"

parallel \
  --results "$RESULTS_DIR" \
  --bar \
  --joblog "$RESULTS_DIR"/job.log \
  uv run vertex-ai-main.py momentum-research-backtest.ipynb \
  --csv "$RESULTS_DIR/metrics_{1}_{2}_{3}_{4}.csv" \
  --html "$RESULTS_DIR/metrics_{1}_{2}_{3}_{4}.html" \
  --days_holding {1} \
  --top_percentile {2} \
  --bottom_percentile {3} \
  --days_momentum {4} \
  ::: 1 3 7 14 20 30 60 \
  ::: 0.5 0.75 0.9 0.95 0.99 \
  ::: 0.5 0.25 0.1 0.05 None \
  ::: 3 5 7 14 21 30 40 50 60 80 100 120 150 220
#
# Results: 1 day holding, 1,3,7, days momentum outperforms <2020, then loses
# money. >=2020 30 day holding, 50 day momentum outperforms.
# Both no stops and highly concentrated

# parallel \
#   --results "$RESULTS_DIR" \
#   --bar \
#   --joblog "$RESULTS_DIR"/job.log \
#   uv run vertex-ai-main.py momentum-research-backtest.ipynb \
#   --csv "$RESULTS_DIR/metrics_{1}_{2}_{3}_{4}.csv" \
#   --html "$RESULTS_DIR/metrics_{1}_{2}_{3}_{4}.html" \
#   --days_holding {1} \
#   --top_percentile {2} \
#   --bottom_percentile {3} \
#   --days_momentum {4} \
#   ::: 25 30 35 40 45 50 55 \
#   ::: 0.95 0.99 \
#   ::: 0.05 None \
#   ::: 35 40 45 50 55 60 65 70 75
#
# Results: tighter grid reveals the 30 day holding and 50 day momentum remians
# the sweet spot with a decrease in performance around that on both axes. There
# is a another sweet spot around 45 days holding and 45 days momentum but the
# neighborhood unterperforms drastically or is even negative.

# parallel \
#   --results "$RESULTS_DIR" \
#   --bar \
#   --joblog "$RESULTS_DIR"/job.log \
#   uv run vertex-ai-main.py momentum-research-backtest.ipynb \
#   --csv "$RESULTS_DIR/metrics_{1}_{2}_{3}_{4}.csv" \
#   --html "$RESULTS_DIR/metrics_{1}_{2}_{3}_{4}.html" \
#   --days_holding {1} \
#   --top_percentile {2} \
#   --bottom_percentile {3} \
#   --days_momentum {4} \
#   ::: 25 26 27 28 29 30 31 32 33 34 35 \
#   ::: 0.95 0.99 \
#   ::: 0.05 None \
#   ::: 45 46 47 48 49 50 51 52 53 54 55
#
# Results: same as with the previous grid. 30/50 is best

# parallel \
#   --results "$RESULTS_DIR" \
#   --bar \
#   --joblog "$RESULTS_DIR"/job.log \
#   uv run vertex-ai-main.py momentum-research-backtest.ipynb \
#   --csv "$RESULTS_DIR/metrics_{1}_{2}_{3}_{4}.csv" \
#   --html "$RESULTS_DIR/metrics_{1}_{2}_{3}_{4}.html" \
#   --days_holding {1} \
#   --top_percentile {2} \
#   --bottom_percentile {3} \
#   --days_momentum {4} \
#   ::: 1 3 7 14 20 30 60 \
#   ::: 0.5 0.75 0.9 0.95 0.99 \
#   ::: None \
#   ::: 3 5 7 14 21 30 40 50 60 80 100 120 150 220
# 
