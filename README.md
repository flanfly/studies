# Studies

[momentum.ipynb]

[btc-momentum.ipynb]
[duckdb.ipynb]
[ideas.ipynb]

Trade 30d and counter trade 1d
------------------------------
* compute momentum/ret correlation and t-statistics for high/low AR, high/low vol, bull/bear
* combine significant features in linear or more complex models
* test against binance data
* incorporate stop loss and check path dependencies on lower timescale data from binance

Run chrono-2 inference:

```bash
uv run papermill chronos-2.ipynb chronos-2-results.ipynb -p lookback 30 --log-output
```
