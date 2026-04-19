parallel \
  --results "$1" \
  --bar \
  --joblog "$1"/job.log \
  uv run vertex-ai-main.py crypto-momentum.ipynb \
  --csv "$1/metrics_{1}_{2}_{3}_{4}_{5}.csv" \
  --html "$1/metrics_{1}_{2}_{3}_{4}_{5}.html" \
  --signal {1} \
  --gate {2} \
  --interval_days {3} \
  --max_long {4} \
  --max_short {5} \
  --show false \
  ::: mom12-1m-a mom12-1m-b mom12m mom6m mom3m mom2m mom1m \
  ::: mom12m-andor-6m ema50d \
  ::: 1 3 7 14 20 30 60 \
  ::: 1 2 3 4 5 6 \
  ::: 0 1 2 3 4 5
