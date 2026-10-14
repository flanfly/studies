use core::fmt;
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::Hash;
use std::iter;
use std::pin::pin;
use std::time::{SystemTime, UNIX_EPOCH};

use log::warn;

use async_stream::try_stream;
use futures::stream::{Stream, StreamExt};

use chrono::DateTime;

use polars::prelude::*;

use serde::{Deserialize, Serialize};

use anyhow::{Result, anyhow};

type Str = Cow<'static, str>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Spot,
    Future,
}

impl Kind {
    pub fn to_str(&self) -> &'static str {
        match self {
            Kind::Spot => "spot",
            Kind::Future => "fut",
        }
    }
}

// venue-specific identifer: <kind>:<venue>:<symbol>
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Instrument(Str);

impl Instrument {
    pub fn new(kind: Kind, venue: impl AsRef<str>, symbol: impl AsRef<str>) -> Self {
        Self(format!("{}:{}:{}", kind.to_str(), venue.as_ref(), symbol.as_ref()).into())
    }
    pub fn kind(&self) -> Kind {
        if self.0.starts_with("fut:") {
            Kind::Future
        } else if self.0.starts_with("spot:") {
            Kind::Spot
        } else {
            panic!("invalid instrument {}", self.0);
        }
    }
}

// global pair identifer: btc, eth,...
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Pair(Str);

// ms unix utc timestamp
#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Timestamp(u64);

impl Timestamp {
    pub fn now() -> Result<Self> {
        Ok(Self(
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64,
        ))
    }

    pub fn masked_min(&self) -> Self {
        Timestamp((self.0 / 60_000) * 60_000)
    }

    pub fn ms(&self) -> u64 {
        self.0
    }

    pub fn is_null(&self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dt = DateTime::from_timestamp_millis(self.0 as i64);
        let dtstr = dt
            .expect("invalid timestamp")
            .format("%Y-%m-%dT%H:%M:%SZ")
            .to_string();

        write!(f, "{}", dtstr)
    }
}

struct Lots(u64);

struct Ticks(u64);

#[derive(Debug, Clone)]
struct Quote {
    pub instrument: Instrument,
    pub pair: Pair,
    pub received_at: Timestamp,
    pub observed_at: Timestamp,
    pub bid: f64,
    pub ask: f64,
    pub bid_qty: u64,
    pub ask_qty: u64,
}

struct Currency {
    id: Str,
    venue: Str,
    full_name: Str,
}

enum Status {
    Active,
    CantTrade,
    CantDeposit,
}

struct Symbology {
    pub instrument: Instrument,
    pub pair: Pair,
    pub status: Status,
    pub kind: Kind,
    pub base: Currency,
    pub quote: Currency,
    pub lot_size: f64,
    pub tick_size: f64,
    pub multiplier: f64,
    pub maker_fee_rate: f64,
    pub taker_fee_rate: f64,
    pub taker_fix_fee: f64,
    pub maker_fix_fee: f64,
    pub received_at: Timestamp,
}

struct Funding {
    pub received_at: Timestamp,
    pub instrument: Instrument,
    // XXX
}

struct Settlement {
    pub received_at: Timestamp,
    pub instrument: Instrument,
    // XXX
}

enum Event {
    Symbology(Symbology),
    Quote(Quote),
    Funding(Funding),
    Settlement(Settlement),
}

/// Generic KuCoin API envelope: `{ "code": "200000", "data": ... }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KucoinResponse<T> {
    code: String,
    data: T,
}

/// A single currency entry from `GET /api/v3/currencies`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct KucoinCurrency {
    currency: String,
    name: String,
    full_name: String,
    precision: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct KucoinSpotSymbol {
    symbol: String,
    name: String,
    base_currency: String,
    quote_currency: String,
    base_min_size: String,
    quote_min_size: String,
    base_max_size: String,
    quote_max_size: String,
    base_increment: String,
    quote_increment: String,
    price_increment: String,
    price_limit_rate: String,
    enable_trading: bool,
    fee_category: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct KucoinFutureSymbol {
    pub symbol: String,
    pub display_symbol: String,
    pub root_symbol: String,
    #[serde(rename = "type")]
    pub asset_type: String,
    pub base_currency: String,
    pub display_base_currency: String,
    pub quote_currency: String,
    pub settle_currency: String,
    pub lot_size: u64,
    pub tick_size: f64,
    pub multiplier: f64,
    pub maker_fee_rate: f64,
    pub taker_fee_rate: f64,
    pub taker_fix_fee: f64,
    pub maker_fix_fee: f64,
    pub status: String,
}

/// `GET /api/v3/exchangeInfo` -- the exchange's trading rules.
///
/// Only the fields we consume are modelled; serde ignores the rest
/// (`rateLimits`, `orderTypes`, `permissions`, `conceptPlates`, ...)
/// so the struct keeps working as MEXC adds fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MexcExchangeInfo {
    timezone: String,
    server_time: u64,
    symbols: Vec<MexcSymbol>,
}

/// A single tradable pair from `exchangeInfo`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MexcSymbol {
    symbol: String,
    status: String,
    base_asset: String,
    base_asset_precision: u32,
    quote_asset: String,
    quote_asset_precision: u32,
    is_spot_trading_allowed: bool,
    is_margin_trading_allowed: bool,
    base_size_precision: String,
    quote_amount_precision: String,
    max_quote_amount: String,
    maker_commission: String,
    taker_commission: String,
    full_name: String,
    filters: Vec<MexcFilter>,
}

/// A trading rule attached to a symbol. MEXC emits different filter
/// types (`PRICE_FILTER`, `LOT_SIZE`, `MIN_NOTIONAL`, ...) in the same
/// array, so every field is optional and we key off `filter_type`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MexcFilter {
    filter_type: String,
    #[serde(default)]
    min_price: Option<String>,
    #[serde(default)]
    max_price: Option<String>,
    #[serde(default)]
    tick_size: Option<String>,
    #[serde(default)]
    min_qty: Option<String>,
    #[serde(default)]
    max_qty: Option<String>,
    #[serde(default)]
    step_size: Option<String>,
    #[serde(default)]
    min_notional: Option<String>,
    #[serde(default)]
    bid_multiplier_up: Option<String>,
    #[serde(default)]
    ask_multiplier_down: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    //let (kucoin, mexc) = try_join!(
    //    reqwest::get("https://api-futures.kucoin.com/api/v1/contracts/active")
    //        .await?
    //        .json::<KucoinResponse<Vec<KucoinFutureSymbol>>>(),
    //    reqwest::get("https://api.mexc.com/api/v3/exchangeInfo?symbol=GMEUSDT")
    //        .await?
    //        .json::<MexcExchangeInfo>()
    //)?;

    //let mut base_mc = vec![];
    //let mut base_kc = vec![];

    //for c in kucoin.data.into_iter() {
    //    base_kc.push(c.base_currency);
    //}

    //for s in mexc.symbols.into_iter() {
    //    if s.base_asset == "GME" {
    //        println!("{:#?}", s);
    //        return Ok(());
    //    }
    //    base_mc.push(s.base_asset);
    //}

    //base_kc.sort();
    //base_kc.dedup();
    //base_mc.sort();
    //base_mc.dedup();

    //println!("{:?}", base_kc);
    //println!("{:?}", base_mc);

    let strm = simulate(signals(continuous(resample(join_causality(
        from_parquet("mexc", Kind::Spot, "../spot-book-resampled-10min.parquet")?,
        from_parquet(
            "kucoin",
            Kind::Future,
            "../future-book-resampled-10min.parquet",
        )?,
    )))));
    let mut strm = pin!(strm);

    let mut pnl_bps = 0.0;
    while let Some(sig) = strm.next().await {
        pnl_bps += sig?;
        println!("{:.02}%", pnl_bps / 10e4);
    }

    Ok(())
}

fn instrument_and_pair(
    venue: impl AsRef<str>,
    kind: Kind,
    symbol: impl AsRef<str>,
) -> (Instrument, Pair) {
    let symbol = symbol.as_ref().to_lowercase();
    let venue = venue.as_ref().to_lowercase();

    match (venue.as_ref(), kind, symbol.as_ref()) {
        ("mexc", Kind::Spot, "gme") => (
            Instrument::new(Kind::Spot, "kucoin", "gmememe"),
            Pair("gmememe".into()),
        ),
        (venue, kind, symbol) => (
            Instrument::new(kind, venue, symbol),
            Pair(symbol.to_string().into()),
        ),
    }
}

fn join_causality<S>(strm1: S, strm2: S) -> impl Stream<Item = Result<Quote>>
where
    S: Stream<Item = Result<Quote>>,
{
    try_stream! {
        let mut s1 = pin!(strm1);
        let mut s2 = pin!(strm2);

        let mut left = s1.next().await;
        let mut right = s2.next().await;

        loop {
            let take_left = match (&left, &right) {
                (Some(Ok(l)), Some(Ok(r))) => l.observed_at <= r.observed_at,
                (Some(_), _) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };

            if take_left {
                yield left.take().unwrap()?;
                left = s1.next().await;
            } else {
                yield right.take().unwrap()?;
                right = s2.next().await;
            }
        }
    }
}

fn from_parquet(
    venue: impl AsRef<str>,
    kind: Kind,
    path: impl AsRef<str>,
) -> Result<impl Stream<Item = Result<Quote>>> {
    // Fallible setup happens eagerly so scan errors surface as a normal Err.
    let lf = LazyFrame::scan_parquet(path.as_ref().into(), ScanArgsParquet::default())?;
    let strm = try_stream! {
        let mut bi = lf.sort(vec!["ts"],SortMultipleOptions::default()  ).collect_batches(Engine::Auto, true, None, true)?;
        let venue = venue.as_ref().to_lowercase();

        while let Some(b) = bi.next() {
            let b = b?;

            let now = Timestamp::now()?;
            let symbol = b.column("symbol")?.str()?;
            let ts = b.column("ts")?.datetime()?;

            // bug in the python script
            let bid = b.column("ask_price")?.f64()?;
            let bid_qty = b.column("ask_qty")?.i64()?;
            let ask = b.column("bid_price")?.f64()?;
            let ask_qty = b.column("bid_qty")?.i64()?;

            // normalize the physical timestamp to ms
            let to_ms = match ts.time_unit() {
                TimeUnit::Milliseconds => 1,
                TimeUnit::Microseconds => 1_000,
                TimeUnit::Nanoseconds => 1_000_000,
            };

            let i = symbol
                .iter()
                .zip(ts.physical().iter())
                .zip(bid.iter())
                .zip(bid_qty.iter())
                .zip(ask.iter())
                .zip(ask_qty.iter());

            for (((((sym, ts), bid), bid_qty), ask), ask_qty) in i {
                let sym = sym.ok_or(anyhow!("missing symbol col"))?;
                let ts = ts.ok_or(anyhow!("missing ts col"))?;

                // drop sentinel rows: src_ts epoch-zero leaks into ts as 1970
                // dates and would create absurd multi-year "gaps"
                if ts <= 0 || (ts as u64) / to_ms < 1_000_000_000_000 {
                    continue;
                }
                let bid = bid.ok_or(anyhow!("missing bid col"))?;
                let bid_qty = bid_qty.ok_or(anyhow!("missing bid_qty col"))?.max(0) as u64;
                let ask = ask.ok_or(anyhow!("missing ask col"))?;
                let ask_qty = ask_qty.ok_or(anyhow!("missing ask_qty col"))?.max(0) as u64;

                let (instrument, pair) = instrument_and_pair(&venue,kind,sym);

                yield Quote {
                    instrument,
                    pair,
                    received_at: now,
                    observed_at: Timestamp((ts as u64) / to_ms),
                    bid,
                    ask,
                    bid_qty,
                    ask_qty,
                };
            }
        }
    };

    Ok(strm)
}

fn resample<S>(strm: S) -> impl Stream<Item = Result<Quote>>
where
    S: Stream<Item = Result<Quote>>,
{
    try_stream! {
        let mut latest_quote = HashMap::<Instrument, Quote>::default();
        let mut strm = pin!(strm);

        while let Some(r) = strm.next().await {
            let mut q = r?;
            let is_new = !latest_quote.contains_key(&q.instrument);
            let ent = latest_quote
                .entry(q.instrument.clone())
                .or_insert_with(|| q.clone());

            if is_new || ent.observed_at < q.observed_at {
                // emit the freshest quote, stamped at its own (minute-masked)
                // timestamp, and keep it as the new latest for this instrument
                q.observed_at = q.observed_at.masked_min();
                *ent = q.clone();
                yield q;
            }

        }
    }
}

fn continuous<S>(strm: S) -> impl Stream<Item = Result<Quote>>
where
    S: Stream<Item = Result<Quote>>,
{
    try_stream! {
        let mut latest_quote = HashMap::<Instrument, Quote>::default();
        let mut strm = pin!(strm);

        while let Some(r) = strm.next().await {
            let q = r?;
                let ent = latest_quote
                    .entry(q.instrument.clone())
                    .or_insert_with(|| q.clone());

                // first-ever sighting: ent is a clone of q, so nothing to
                // fill; just emit it
                if ent.observed_at >= q.observed_at {
                    yield q;
                    continue;
                }

                // forward-fill the gap at 10-min steps (matching the data's
                // cadence), but only up to MAX_FILL_MS; beyond that the old
                // quote is stale and the gap stays a hole (filling it would
                // just freeze old prices for potentially millions of minutes).
                let start = ent.observed_at.ms();
                let end = q.observed_at.ms();
                let fill_end = end.min(start + MAX_FILL_MS);

                // start at start + FILL_STEP: the ent minute itself was already
                // emitted (chain-once of the previous cycle), re-emitting it
                // would duplicate every observation minute.
                const FILL_STEP: u64 = 10 * 60_000;
                let ff = (start + FILL_STEP..fill_end)
                    .step_by(FILL_STEP as usize)
                    .into_iter()
                    .map(|ts| {
                        let mut qq = ent.clone();
                        qq.observed_at = Timestamp(ts);
                        qq
                    })
                    .chain(iter::once(q.clone()));

                for qq in ff {
                    yield qq;
                }
                *ent = q;
            }

    }
}

#[derive(Clone, Copy, Debug)]
struct Row {
    pub ts: Timestamp,
    pub spot_bid: f64,
    pub spot_ask: f64,
    pub fut_bid: f64,
    pub fut_ask: f64,

    // derived
    pub spot_mid: f64,
    pub fut_mid: f64,

    // bps
    pub spot_spread: f64,
    pub fut_spread: f64,

    // rel to spot
    pub basis: f64,

    // bps
    pub cost: f64,
}

impl Default for Row {
    fn default() -> Self {
        Self {
            ts: Default::default(),
            basis: f64::NAN,
            spot_bid: f64::NAN,
            spot_ask: f64::NAN,
            fut_bid: f64::NAN,
            fut_ask: f64::NAN,
            spot_mid: f64::NAN,
            fut_mid: f64::NAN,
            spot_spread: f64::NAN,
            fut_spread: f64::NAN,
            cost: f64::NAN,
        }
    }
}
const WINDOW: usize = 24 * 60 * 60 / 600; // 24h of 10min observations = 144 slots

/// don't forward-fill quotes older than this (ms); past it, the data is
/// considered stale and the gap is left as a hole instead of being padded
/// with millions of frozen quotes.
const MAX_FILL_MS: u64 = 24 * 60 * 60 * 1000;

#[derive(Debug)]
struct Signal {
    pub created_at: Timestamp,
    pub pair: Pair,
    pub net_basis: f64,
    pub triggered: bool,
}

fn signals<S>(strm: S) -> impl Stream<Item = Result<Signal>>
where
    S: Stream<Item = Result<Quote>>,
{
    let half_spread_capture = 0.0;
    let fut_taker_bps = 6.0;
    let fut_maker_bps = 2.0;
    let spot_taker_bps = 0.0;
    let spot_maker_bps = 0.0;

    try_stream! {
        let mut rings = HashMap::<Pair, [Row; WINDOW]>::default();
        let mut strm = pin!(strm);

        while let Some(r) = strm.next().await {
            let q = r?;
            let ring = rings
                .entry(q.pair.clone())
                .or_insert_with(|| [Row::default(); WINDOW]);
            let i = ((q.observed_at.ms() / 600_000) % WINDOW as u64) as usize;
            let mut row = ring[i];
            let ts = q.observed_at.masked_min();

            if row.ts < ts {
                row = Default::default();
            } else if row.ts > ts {
                warn!("stale quote {:?}", q);
            }

            row.ts = q.observed_at.masked_min();
            match q.instrument.kind() {
                Kind::Spot => {
                    row.spot_bid = q.bid;
                    row.spot_ask = q.ask;
                }
                Kind::Future => {
                    row.fut_bid = q.bid;
                    row.fut_ask = q.ask;
                }
            }

            row.spot_mid = (row.spot_bid + row.spot_ask) / 2.0;
            row.spot_spread = (row.spot_ask - row.spot_bid) / row.spot_mid * 1e4;

            row.fut_mid = (row.fut_bid + row.fut_ask) / 2.0;
            row.fut_spread = (row.fut_ask - row.fut_bid) / row.fut_mid * 1e4;

            row.basis = (row.fut_mid - row.spot_mid) / row.spot_mid * 1e4;

            if row.spot_spread > row.fut_spread {
                row.cost = row.fut_spread / 2.0 + fut_taker_bps
                    - row.spot_spread / 2.0 * half_spread_capture
                    + spot_maker_bps;
            } else {
                row.cost = row.spot_spread / 2.0 + spot_taker_bps
                    - row.fut_spread / 2.0 * half_spread_capture
                    + fut_maker_bps;
            }

            ring[i] = row;

            // mean basis
            let (mu_s, mu_c) = ring
                .iter()
                .filter_map(
                    |&Row { ts, basis, .. }| {
                        if ts.is_null() { None } else { Some(basis) }
                    },
                )
                .fold((0.0, 0.0), |(s, c), b| (s + b, c + 1.0));
            let mu = mu_s / mu_c;
            let net_dev = row.basis - mu - 2.0 * row.cost;
            let net_basis = row.basis - 2.0 * row.cost;

            if net_dev.is_finite() {
                let signal = Signal{
                    created_at: Timestamp::now()?,
                    pair: q.pair.clone(),
                    net_basis,
                    triggered: net_dev > 200.0,
                };

                yield signal;
            }
        }
    }
}

struct Position {
    pair: Pair,
    entry_net_basis: f64,
    entry_ts: Timestamp,
    current_net_basis: f64,
    current_ts: Timestamp,
}

const HOLDING_MS: u64 = 3 * 60 * 60 * 1000; // 3h in ms

fn simulate<S>(strm: S) -> impl Stream<Item = Result<f64>>
where
    S: Stream<Item = Result<Signal>>,
{
    try_stream! {
        let mut strm = pin!(strm);
        let mut book = HashMap::<Pair,Position>::default();

        while let Some(sig) = strm.next().await {
            let sig = sig?;

            match book.get_mut(&sig.pair) {
                Some(pos) => {
                    pos.current_ts = sig.created_at;
                    pos.current_net_basis = sig.net_basis;

                    if sig.created_at.ms() <= pos.entry_ts.ms() + HOLDING_MS {
                        yield pos.entry_net_basis - pos.current_net_basis;
                    }
                }
                None => {
                    let pos = Position{
                        pair: sig.pair.clone(),
                        entry_net_basis: sig.net_basis,
                        entry_ts: sig.created_at,
                        current_net_basis: sig.net_basis,
                        current_ts: sig.created_at
                    };
                    book.insert(sig.pair.clone(), pos);
                }
            }

            if !sig.triggered {
                book.remove(&sig.pair);
            }
        }
    }
}
