import duckdb
import dotenv
import os
import polars as pl

dotenv.load_dotenv()

access_key = os.getenv("R2_ACCESS_KEY")
secret_key = os.getenv("R2_SECRET_KEY")
account_id = os.getenv("R2_ACCOUNT_ID")

con = duckdb.connect(database=':memory:')

con.execute('''
create or replace secret r2 (
    type r2,
    provider config,
    key_id ?, secret ?, account_id ?
);
''', [access_key, secret_key, account_id])

df = con.sql('''
select
    epoch_ms(epoch(ts)::bigint) ts,
    open, high, low, close, closetime, volume, qty_volume, trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore, symbol, year
from read_parquet('r2://studies-data-eu/binance-spot-1m/all/*/*.parquet', hive_partitioning = true)
where year = 2025
limit 100
''')
print(df)
