import polars as pl
import argparse
import sys
import re
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Process crypto data files."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="The output file path",
        default="output.parquet"
    )

    parser.add_argument(
        "input_files",
        nargs="*",
        help="One or more input csv"
    )

    args = parser.parse_args()

    # logic check: ensure at least one input file exists
    if not args.input_files:
        print("Error: No input files provided.")
        parser.print_help()
        sys.exit(1)

    lst = []
    for p in tqdm(args.input_files, desc='reading files'):
        match = re.match(r'([A-Z0-9]+)-1m-(20[0-9]{2})-([0-9]{2})\.csv', os.path.basename(p))
        if match is None:
            l.error(f'{p} is not a valid file name')
            return

        sym = match.group(1)

        try:
            lf = pl.scan_csv(
                p,
                has_header=False,
                new_columns=['ts','open','high','low','close','volume','closetime','qty_volume','trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'],
                schema_overrides={
                    "ts": pl.Int64,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "closetime": pl.Int64,
                    "qty_volume": pl.Float64,
                    "trades": pl.Int64,
                    "taker_buy_base_asset_volume": pl.Float64,
                    "taker_buy_quote_asset_volume": pl.Float64,
                    "ignore": pl.Float64,
                }
            ).with_columns(
                [
                    pl.col("close").cast(pl.Float64, strict=False),
                    pl.col("volume").cast(pl.Float64, strict=False),
                    pl.col("ts").cast(pl.Datetime("ms")),
                    pl.lit(sym.lower()).alias("symbol"),
                ]
            )

            lst.append(lf)
        except Exception as e:
            print(f"Error processing file {p}: {e}")
            continue

    pl.concat(tqdm(lst, desc='writing frames')).sink_parquet(args.output, compression='gzip')

if __name__ == "__main__":
    main()
