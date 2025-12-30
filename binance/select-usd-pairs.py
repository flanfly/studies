import duckdb

con = duckdb.connect(database=':memory:')

df = con.sql('''
    select * from read_parquet('../binance-spot-1m/all/all-year=2017.parquet', filename=true) limit 10
''').pl()

print(df)
