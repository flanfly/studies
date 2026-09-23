import sys
from os import getenv

from schwab.client import Client
from schwab.auth import easy_client
import httpx
import asyncio

from typing import Annotated
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

import polars as pl
import pandas as pd

import datetime as dt
import dateutil as du
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.table import Table

from dotenv import load_dotenv
import logging as l
from tqdm.contrib.logging import logging_redirect_tqdm

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

load_dotenv()

EOM_PERIOD = 5
BOM_PERIOD = 5


def nyse_holidays():
    tables = pd.read_html("https://www.nyse.com/trade/hours-calendars")
    holidays = pl.DataFrame(tables[0])
    now = dt.datetime.now().replace(
        microsecond=0, second=0, minute=0, hour=0, tzinfo=ZoneInfo("EST")
    )

    frag = []
    for ystr in holidays.columns:
        if ystr == "Holiday":
            continue

        elif ystr.isdigit():
            year = int(ystr)
            if year < now.year or year > now.year + 3:
                l.warning(f"skipping year {ystr}")
                continue

            for row in holidays.select(["Holiday", ystr]).iter_rows():
                holiday_name, dstr = row
                try:
                    date = du.parser.parse(dstr, fuzzy=True)
                    date = date.replace(year=year, tzinfo=ZoneInfo("EST"))
                    frag.append({"name": holiday_name, "ts": date})
                except ValueError:
                    l.warning(f"failed parsing {dstr}")
                    continue

        else:
            l.warning(f"unexpected column type: {type(ystr)}")
            continue

    return pl.DataFrame(frag).sort("ts")


async def closing_prices(c, sym: str, since: dt.datetime) -> pl.DataFrame:
    resp = await c.get_price_history_every_day(sym, start_datetime=since)
    assert resp.status_code == httpx.codes.OK

    return pl.DataFrame(
        resp.json().get("candles", []),
        schema={
            "datetime": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        },
    ).select(
        ts=pl.from_epoch(pl.col("datetime"), time_unit="ms")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("EST"),
        close=pl.col("close"),
    )


def parse_ms_timestamp(v: int | float | dt.datetime) -> dt.datetime:
    if isinstance(v, (int, float)):
        return dt.datetime.fromtimestamp(v / 1000, tz=dt.timezone.utc)
    return v


Datetime = Annotated[dt.datetime, BeforeValidator(parse_ms_timestamp)]


class Quote(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ask_mic_id: str = Field(alias="askMICId")
    ask_price: float = Field(alias="askPrice")
    ask_size: int = Field(alias="askSize")
    ask_time: Datetime = Field(alias="askTime")

    bid_mic_id: str = Field(alias="bidMICId")
    bid_price: float = Field(alias="bidPrice")
    bid_size: int = Field(alias="bidSize")
    bid_time: Datetime = Field(alias="bidTime")

    trade_time: Datetime = Field(alias="tradeTime")


async def quotes(c, symbols: list[str]) -> list[Quote]:
    resp = await c.get_quotes(symbols)
    assert resp.status_code == httpx.codes.OK

    quotes = [sym["quote"] for sym in resp.json().values()]
    return [Quote.model_validate(q) for q in quotes]


async def historical_data(c, spy=None, ief=None) -> pl.DataFrame:
    month_sel = [
        pl.col("ts").dt.year().alias("year"),
        pl.col("ts").dt.month().alias("month"),
    ]

    td = dt.datetime.now().replace(
        microsecond=0, second=0, minute=0, hour=0, tzinfo=ZoneInfo("EST")
    )

    # Labelled calendar (all days): weekends and NYSE holidays keep their name
    # but are excluded from every trading-day count below.
    range_start = td - dt.timedelta(days=90)
    range_end = td + dt.timedelta(days=30)
    days = (
        pl.DataFrame(
            pl.datetime_range(range_start, range_end, interval="1d", eager=True).alias(
                "ts"
            )
        )
        .join(nyse_holidays(), on="ts", how="full")
        .with_columns(
            ts=pl.coalesce("ts", "ts_right"),
            # non-null only on a NYSE holiday (never "", so it cannot be
            # clobbered the way an empty-string weekend marker would)
            holiday=pl.col("name"),
        )
        .sort("ts")
        # drop holiday rows the NYSE table contributes outside the window
        .filter((pl.col("ts") >= range_start) & (pl.col("ts") <= range_end))
        .with_columns(
            name=pl.when(pl.col("ts").dt.weekday().is_in([6, 7]))
            .then(pl.lit("Weekend"))
            .otherwise(pl.col("holiday").fill_null(""))  # "" = trading day
        )
        .select("ts", "name")
    )

    # Trading sessions only: all indices, returns and positions are computed
    # here so non-trading days can never be counted.
    sessions = (
        days.filter(pl.col("name") == "")
        .with_columns(month_sel)
        .with_columns(
            # 1-based index of the trading day within its month
            trading_day=pl.int_range(1, pl.len() + 1).over(month_sel),
            # total number of trading days in the month
            num_trading_days=pl.len().over(month_sel),
        )
        .with_columns(
            # trading days left after today: 0 = last trading day,
            # eom_period = signal day (one day before the eom leg opens)
            rem_trading_days=pl.col("num_trading_days")
            - pl.col("trading_day")
        )
        .with_columns(decision_day=pl.col("rem_trading_days") == EOM_PERIOD)
        .select("ts", "decision_day", "rem_trading_days", "trading_day")
    )

    start_date = sessions["ts"].min()

    if spy is None or ief is None:
        spy, ief = await asyncio.gather(
            closing_prices(c, "SPY", start_date),
            closing_prices(c, "IEF", start_date),
        )

    df = (
        sessions.join(spy.rename({"close": "spy"}), on="ts", how="left")
        .join(ief.rename({"close": "ief"}), on="ts", how="left")
        .with_columns(
            # previous *trading* day's close (the frame holds trading days only,
            # so shift(1) cannot land on a weekend/holiday null)
            spy_yday=pl.col("spy").shift(1),
            ief_yday=pl.col("ief").shift(1),
        )
        .with_columns(
            # mtd return today (rel. close of the last trading day of the
            # previous month: the first trading day's yday)
            spy_mtd=pl.col("spy") / pl.col("spy_yday").first().over(month_sel) - 1,
            ief_mtd=pl.col("ief") / pl.col("ief_yday").first().over(month_sel) - 1,
        )
        .with_columns(
            # month to date return of the portfolio
            mtd=0.6 * (1 + pl.col("spy_mtd"))
            + 0.4 * (1 + pl.col("ief_mtd")),
        )
        .with_columns(
            # expected bond buying/selling going into eom
            # (trading_day/rem_trading_days come from `days`, in trading-day space)
            pressure_bps=(0.4 - 0.4 * (1 + pl.col("ief_mtd")) / pl.col("mtd"))
            * 1e4,
        )
        .with_columns(
            # pin the signal to the decision day and carry it across month-end
            # through the bom leg: the frame holds trading days only, so a row
            # limit of eom_period + bom_period spans exactly 5+5 trading days
            # (bom days read the *previous* month's signal, as in the research)
            effective_pressure_bps=pl.when(pl.col("rem_trading_days") == EOM_PERIOD)
            .then(pl.col("pressure_bps"))
            .otherwise(None)
            .forward_fill(limit=EOM_PERIOD + BOM_PERIOD)
        )
        .with_columns(
            eom_pos=(
                pl.when(pl.col("effective_pressure_bps") <= -50)
                .then(pl.lit("long spy"))
                .otherwise(pl.lit("long tlt"))
            ),
            bom_pos=(
                pl.when(pl.col("effective_pressure_bps") >= 50)
                .then(pl.lit("long spy, short tlt"))
                .otherwise(pl.lit("cash"))
            ),
        )
        .with_columns(
            # Convention (matches calendar-leveraged.py): `instruction` is the
            # position to be in from the close of that day; `decision` '*' marks
            # the day on which a trade executes at the close.
            #   rem == eom_period              -> enter eom leg at close
            #   rem in 1..eom_period-1         -> hold eom leg
            #   rem == 0                       -> switch eom -> bom at close (last day)
            #   trading_day in 1..bom_period-1 -> hold bom leg
            #   trading_day == bom_period      -> exit bom leg at close
            instruction=(
                pl.when(pl.col("effective_pressure_bps").is_null())
                .then(None)
                .otherwise(
                    pl.when(
                        (pl.col("rem_trading_days") <= EOM_PERIOD)
                        & (pl.col("rem_trading_days") > 0)
                    )
                    .then("eom_pos")
                    .otherwise(
                        pl.when(
                            (pl.col("rem_trading_days") == 0)
                            | (pl.col("trading_day") < BOM_PERIOD)
                        )
                        .then("bom_pos")
                        .otherwise(pl.lit("cash"))
                    )
                )
            ),
            decision=(
                pl.when(
                    (pl.col("rem_trading_days") == EOM_PERIOD)
                    | (pl.col("rem_trading_days") == 0)
                    | (pl.col("trading_day") == BOM_PERIOD)
                )
                .then(pl.lit("*"))
                .otherwise(pl.lit(""))
            ),
        )
        .select(
            [
                "ts",
                "trading_day",
                "rem_trading_days",
                "effective_pressure_bps",
                "instruction",
                "decision",
                "decision_day",
            ]
        )
    )

    # re-attach the labelled non-trading days (weekends/holidays) for display
    # only; they carry no indices, prices or instructions
    return days.join(df, on="ts", how="left")


async def main():
    c = easy_client(
        api_key=getenv("SCHWAB_CLIENT_ID"),
        app_secret=getenv("SCHWAB_CLIENT_SECRET"),
        callback_url="https://127.0.0.1:8182",
        token_path="token.json",
        asyncio=True,
    )

    now = dt.datetime.now()
    td = now.replace(microsecond=0, second=0, minute=0, hour=0, tzinfo=ZoneInfo("EST"))
    start_date = td - dt.timedelta(days=90)

    spy, ief, q = await asyncio.gather(
        closing_prices(c, "SPY", start_date),
        closing_prices(c, "IEF", start_date),
        quotes(c, ["SPY", "IEF"]),
    )
    spy_q, ief_q = q

    sdf = lambda p: pl.DataFrame([[td, p]], schema=["ts", "close"], orient="row")
    spy = pl.concat(
        [
            spy,
            sdf(spy_q.ask_price),
        ]
    )
    ief = pl.concat(
        [
            ief,
            sdf(ief_q.ask_price),
        ]
    )

    df = await historical_data(c, spy, ief)

    nyc_now = dt.datetime.now(ZoneInfo("America/New_York"))
    if nyc_now.hour > 16:
        next_close = (nyc_now + dt.timedelta(days=1)).replace(
            microsecond=0, second=59, minute=59, hour=15
        )
    else:
        next_close = nyc_now.replace(microsecond=0, second=59, minute=59, hour=15)
    rem = (next_close - nyc_now).total_seconds()

    # horizon: last decision day before today through the bom exit after the
    # next decision day. The two legs span eom_period + bom_period trading
    # days, which can be ~14 calendar days across weekends; use a safe buffer.
    hori_start = df.filter((pl.col("ts") < td) & pl.col("decision_day"))["ts"].max()
    hori_end = df.filter((pl.col("ts") > td) & pl.col("decision_day"))[
        "ts"
    ].min() + dt.timedelta(days=2 * (EOM_PERIOD + BOM_PERIOD))

    console = Console()

    table = Table(show_header=True)
    table.add_column("Date")
    table.add_column("Pressure")
    table.add_column("Position")
    table.add_column("")

    iter = df.filter(
        (pl.col("ts") >= hori_start) & (pl.col("ts") <= hori_end)
    ).iter_rows(named=True)
    for row in iter:
        date = f'{row["ts"].strftime("%d-%b")} {row["name"]}'.rstrip()
        pressure = (
            f"{row['effective_pressure_bps']:.0f} bps"
            if row["effective_pressure_bps"] is not None
            else ""
        )
        position = row["instruction"] if row["instruction"] else "-"
        decision = row["decision"] if row["decision"] else ""

        if row["ts"] < td:
            style = "dim"
        elif row["ts"] == td and decision != "":
            style = "red underline"
        elif row["ts"] == td:
            style = "underline"
        elif decision != "":
            style = "red"
        else:
            style = None

        table.add_row(date, pressure, position, decision, style=style)

    console.print(table)
    console.print(
        f'\nTime in NYC: {nyc_now.strftime("%H:%M")}, {int(rem // 3600)}h{int((rem % 3600) // 60)}m until close.'
    )


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
            sys.exit(1)
