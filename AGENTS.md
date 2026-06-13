Overview
========

This is a monorepo containing various Jupyter notebooks and Python scripts to
research, develop and productize trading strategies for equities, derivatives
and cryptocurrencies.

Guidelines
----------

* Do not do any code changes unless explicitly instructed. Do not commit
  changes unless explicitly instructed.
* Use `uv` to manage script depdendencies. Run scripts with `uv run <script>`.
* Each strategy comes as a Jupyter notebook, most have a paired Python script
  using Jupytext. Make sure to update the notebook after modifying the script
  and vice versa using `jupytext --sync <notebook>`.

Selected Scripts
----------------

* sync-datastore.py: Downloads historical market data from Binance and stores
  them on r2 for further analysis.
* yf.py, fred.py: download historical market data from Yahoo Finance and FRED,
  respectively.
* polarity/poll-polarity.sh: Downloads metrics and price data from Polarity Digital.
* live/main.py: Downloads OHLCV candles, pair margin borrow rates and perpetual
  funding rates from the largest CEX and DEX venues.

