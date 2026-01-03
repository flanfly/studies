import duckdb
from dotenv import load_dotenv
import os
import polars as pl

load_dotenv()

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

con.execute('set enable_progress_bar=true;')
con.execute('''set max_memory='27GB';''')
con.execute('''set temp_directory = './duckdb_temp/';''')
con.execute('set preserve_insertion_order = false;')
con.execute('''set max_temp_directory_size='210GiB';''')
con.execute('''
copy (
    select
        date_trunc('day', timestamp) AS ts,
        symbol,
        arg_min(open, timestamp) AS open,
        MAX(high) AS high,
        MIN(low) AS low,
        arg_max(close, timestamp) AS close,
        SUM(volume) AS volume,
        SUM(taker_buy_quote_asset_volume) AS taker_buy_quote_asset_volume,
        SUM(taker_buy_base_asset_volume) AS taker_buy_base_asset_volume
    from (
        select ts as timestamp, open, high, low, close, volume, taker_buy_quote_asset_volume, taker_buy_base_asset_volume, symbol
        from read_parquet('r2://studies-data-eu/binance-spot-1m/all/*/*.parquet', hive_partitioning = true)
        where year in (2024, 2025)
        --limit 100
    )
    group by ts, symbol
    order by ts, symbol
) to 'r2://studies-data-eu/binance-spot-1m/1d-4.parquet' (
    FORMAT PARQUET, 
    COMPRESSION 'zstd',
    OVERWRITE_OR_IGNORE true
);
''')
