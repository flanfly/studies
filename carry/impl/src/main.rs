use core::fmt;
use std::array;
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::Hash;
use std::iter;
use std::pin::pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::format::parse;
use log::{info, warn};

use async_stream::try_stream;
use futures::stream::{FuturesUnordered, Stream, StreamExt};

use chrono::DateTime;

use polars::prelude::*;

use serde::{Deserialize, Serialize};

use anyhow::{Result, anyhow};

use simba::simd::{SimdValue, f64x8};

use tokio::select;
use tokio::sync::Semaphore;
use tokio::task::spawn_blocking;

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
    // bar close timestamp
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

//enum Event {
//    Symbology(Symbology),
//    Quote(Quote),
//    Funding(Funding),
//    Settlement(Settlement),
//}

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

    let strm = simulate(arma(basis(join_causality(
        from_parquet(
            "mexc",
            Kind::Spot,
            Duration::parse("10m"),
            "../spot-book-resampled-10min.parquet",
        )?,
        from_parquet(
            "kucoin",
            Kind::Future,
            Duration::parse("10m"),
            "../future-book-resampled-10min.parquet",
        )?,
    ))));
    let mut strm = pin!(strm);

    let mut pnl_bps = 0.0;
    let mut book = HashMap::<Pair, f64>::default();
    while let Some(ev) = strm.next().await {
        let ev = ev?;
        let ts = match ev {
            Event::Open {
                created_at,
                pair,
                basis,
                cost,
                ..
            } => {
                println!("[{}] open position on {}", created_at, pair.0);
                book.insert(pair, basis - cost);
                created_at
            }
            Event::Update { created_at, .. } => created_at,
            Event::Close {
                created_at,
                pair,
                entry_basis,
                basis,
                ..
            } => {
                book.remove(&pair);
                pnl_bps += entry_basis - basis;
                println!("[{}] close position on {}", created_at, pair.0);
                created_at
            }
        };
        println!(
            "[{}] PnL {:.2}% ({:.2}% unrealized)",
            ts,
            pnl_bps / 10e4,
            book.values().sum::<f64>() / 10e4
        );
        //pnl_bps += pnl.position_return;
        //println!(
        //    "{}: close {} -> {:.02}%",
        //    pnl.pair,
        //    pnl.observed_at,
        //    pnl_bps / 10e4
        //);
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
    every: Duration,
    path: impl AsRef<str>,
) -> Result<impl Stream<Item = Result<Quote>>> {
    // Fallible setup happens eagerly so scan errors surface as a normal Err.
    let lf = LazyFrame::scan_parquet(path.as_ref().into(), ScanArgsParquet::default())?;
    let strm = try_stream! {
    let venue = venue.as_ref().to_lowercase();
    let agg = lf
        .sort(["ts"], SortMultipleOptions::default())
        .filter(col("ts").dt().year().gt_eq(lit(2026)))
        .group_by_dynamic(
            col("ts"),
            [col("symbol")],
            DynamicGroupOptions {
                every,
                period: every,
                offset: Duration::parse("0s"),
                label: Label::Right,
                closed_window: ClosedWindow::Left,
                start_by: StartBy::WindowBound,
                include_boundaries: false,
                ..Default::default()
            },
        )
        // bug in the python script
        .agg([
            col("bid_price").last().alias("ask"),
            col("bid_qty").last().alias("ask_qty"),
            col("ask_price").last().alias("bid"),
            col("ask_qty").last().alias("bid_qty"),
         ])
        .with_columns([((col("bid") + col("ask")) / lit(2.0)).alias("mid"), ]) ;

    let grid = agg
        .clone()
        .select([col("symbol").unique()])
        .cross_join(
            agg
            .clone()
            .select([
                datetime_range(
                    Some(col("ts").min()),
                    Some(col("ts").max()),
                    Some(every),
                    None,
                    ClosedWindow::Both,
                    None,
                    None,
                )?.alias("ts")
            ]),
            None);

    let keys = [col("symbol"), col("ts")];
    let mut bi = grid
        .join(agg, keys.clone(), keys, JoinArgs::new(JoinType::Left))
        .sort(["symbol", "ts"], SortMultipleOptions::default())
        .with_columns([
            col("ask").fill_null_with_strategy(FillNullStrategy::Forward(None)).over([col("symbol")])?,
            col("ask_qty").fill_null_with_strategy(FillNullStrategy::Forward(None)).over([col("symbol")])?,
            col("bid").fill_null_with_strategy(FillNullStrategy::Forward(None)).over([col("symbol")])?,
            col("bid_qty").fill_null_with_strategy(FillNullStrategy::Forward(None)).over([col("symbol")])?,
        ])
        .filter(col("ask").is_not_null())
        .sort(["ts", "symbol"], SortMultipleOptions::default())
        .collect_batches(Engine::Auto, true, None, true)?;

        while let Some(b) = bi.next() {
            let b = b?;

            let now = Timestamp::now()?;
            let symbol = b.column("symbol")?.str()?;
            let ts = b.column("ts")?.datetime()?;

            let bid = b.column("bid")?.f64()?;
            let bid_qty = b.column("bid_qty")?.i64()?;
            let ask = b.column("ask")?.f64()?;
            let ask_qty = b.column("ask_qty")?.i64()?;

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
struct Basis {
    // latest input bar's closing ts
    pub observed_at: Timestamp,
    // inference ts
    pub created_at: Timestamp,
    pub pair: Pair,
    pub basis: f64,
    pub cost: f64,
}

fn basis<S>(strm: S) -> impl Stream<Item = Result<Basis>>
where
    S: Stream<Item = Result<Quote>>,
{
    let half_spread_capture = 0.0;
    let fut_taker_bps = 6.0;
    let fut_maker_bps = 2.0;
    let spot_taker_bps = 0.0;
    let spot_maker_bps = 0.0;

    try_stream! {
        let mut rows = HashMap::<Pair, Row>::default();
        let mut strm = pin!(strm);

        while let Some(r) = strm.next().await {
            let q = r?;
            let row = rows
                .entry(q.pair.clone())
                .or_insert_with(Row::default);
            let ts = q.observed_at.masked_min();

            if row.ts > ts {
                warn!("stale data recv {:?}", q);
            }

            row.ts = ts;

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

            if row.basis.is_finite() {
                let basis = Basis{
                    observed_at: row.ts,
                    created_at: Timestamp::now()?,
                    pair: q.pair.clone(),
                    cost: row.cost,
                    basis : row.basis,
                };

                yield basis;
            }
        }
    }
}

#[derive(Debug)]
struct Signal {
    // latest input bar's closing ts
    pub observed_at: Timestamp,
    // inference ts
    pub created_at: Timestamp,
    pub pair: Pair,
    pub deviation: f64,
    pub cost: f64,
    pub basis: f64,
}

#[derive(Clone)]
struct InferenceAux {
    pub observed_at: Timestamp,
    pub pair: Pair,
    // basis net of round-trip cost, the series the ARMA is fitted on
    pub basis: f64,
    pub cost: f64,
}

fn arma<S>(strm: S) -> impl Stream<Item = Result<Signal>>
where
    S: Stream<Item = Result<Basis>>,
{
    let max_iter = 100;
    let lambda = 10e-3;
    let delta_bic_threshold = 1.0;

    // pairs co-fitted in a single SIMD ARMA call
    const LANES: usize = 8;

    try_stream! {
        // per-pair rolling window of basis minus two times entry cost (assumes entry = exit cost)
        let mut y = HashMap::<Pair, Box<[f64; WINDOW]>>::default();
        // last fitted ARMA params, used to warm-start the next window
        let mut phi = HashMap::<Pair, f64>::default();
        let mut theta = HashMap::<Pair, f64>::default();
        let mut aux = HashMap::<Pair, InferenceAux>::default();

        let mut ts: Option<Timestamp> = None;

        let mut res_strm = FuturesUnordered::new();
        let mut q_strm = pin!(strm);

        let sem = Arc::new(Semaphore::new(
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2),
        ));

        // tokio::select! evaluates each handler as plain code inside the
        // generated async block, so `?` cannot be used there; stash the error
        // and propagate it after the select.
        let mut fail: Option<anyhow::Error> = None;

        loop {
            select!{
                Some(r) = q_strm.next() => {
                    match r {
                        Err(e) => fail = Some(e),
                        Ok(q) => {
                            let bar = q.observed_at;

                            match ts {
                                None => ts = Some(bar),
                                Some(prev) if bar < prev => warn!("late quote: {:?}", q),
                                Some(prev) if bar > prev => {
                                    let win = ((prev.ms() / 600_000) % WINDOW as u64) as usize;
                                    let entries: Vec<(&Pair, &Box<[f64; WINDOW]>)> = y
                                        .iter()
                                        .filter(|(_, yw)| yw.iter().cloned().all(f64::is_finite))
                                        .collect();

                                    for chunk in entries.chunks(LANES) {
                                        // oldest -> newest for the window ending at `ts`
                                        let yv = (0..WINDOW)
                                            .map(|k| {
                                                let idx = (win + 1 + k) % WINDOW;
                                                f64x8::from(std::array::from_fn(|l| {
                                                    chunk
                                                        .get(l)
                                                        .map(|(_, yw)| yw[idx])
                                                        .unwrap_or(f64::NAN)
                                                }))
                                            })
                                            .collect::<Vec<_>>();
                                        let mut phiv = [f64x8::from(std::array::from_fn(|l| {
                                            chunk
                                                .get(l)
                                                .and_then(|(p, _)| phi.get(*p).copied())
                                                .filter(|v| v.is_finite())
                                                .unwrap_or(0.0)
                                        }))];
                                        let mut thetav = [f64x8::from(std::array::from_fn(|l| {
                                            chunk
                                                .get(l)
                                                .and_then(|(p, _)| theta.get(*p).copied())
                                                .filter(|v| v.is_finite())
                                                .unwrap_or(0.0)
                                        }))];
                                        let auxv: Box<[Option<InferenceAux>]> = (0..LANES)
                                            .map(|l| {
                                                chunk
                                                    .get(l)
                                                    .and_then(|(p, _)| aux.get(*p))
                                                    .cloned()
                                            })
                                            .collect();

                                        info!("flush ARMA buffer");
                                        let permit = sem
                                            .clone()
                                            .acquire_owned()
                                            .await
                                            .expect("semaphore not closed");
                                        res_strm.push(spawn_blocking(move || {
                                            let _permit = permit;
                                            fast_arma::arma::<f64x8, 1, 1>(
                                                &yv,
                                                &mut phiv,
                                                &mut thetav,
                                                lambda,
                                                max_iter,
                                            )
                                            .map(|fit| (auxv, fit, phiv, thetav))
                                        }));
                                    }

                                    ts = Some(bar);
                                }
                                Some(_) => {
                                    let i = ((bar.ms() / 600_000) % WINDOW as u64) as usize;
                                    let wy = y
                                        .entry(q.pair.clone())
                                        .or_insert_with(|| Box::new([f64::NAN; WINDOW]));
                                    wy[i] = q.basis - q.cost;

                                    aux.insert(q.pair.clone(), InferenceAux {
                                        observed_at: bar,
                                        pair: q.pair.clone(),
                                        basis: q.basis,
                                        cost: q.cost,
                                    });
                                }
                            }
                        }
                    }
                }

                Some(r) = res_strm.next() => {
                    match r {
                        Err(e) => fail = Some(e.into()),
                        Ok(Err(e)) => fail = Some(e),
                        Ok(Ok((auxv, fit, phiv, thetav))) => {
                            let now = Timestamp::now().unwrap_or_default();
                            let dbic = fit.delta_bic();

                            for (l, aux) in auxv.iter().enumerate() {
                                let Some(aux) = aux else { continue };

                                phi.insert(aux.pair.clone(), phiv[0].extract(l));
                                theta.insert(aux.pair.clone(), thetav[0].extract(l));

                                if dbic.extract(l) > delta_bic_threshold {
                                    let net_dev = aux.basis - 2.0*aux.cost - thetav[0].extract(l);
                                    yield Signal{
                                        observed_at: aux.observed_at,
                                        created_at: now,
                                        deviation: net_dev,
                                        basis: aux.basis,
                                        cost: aux.cost,
                                        pair: aux.pair.clone(),
                                    };
                                }
                            }
                        }
                    }
                }

                else => break,
            }

            if let Some(e) = fail.take() {
                Err(e)?;
            }
        }
    }
}

struct Position {
    pair: Pair,
    entry_basis: f64,
    entry_ts: Timestamp,
    entry_cost: f64,
    last_basis: f64,
    last_deviation: f64,
    last_cost: f64,
    last_ts: Timestamp,
}

#[derive(Debug)]
enum Event {
    Open {
        created_at: Timestamp,
        pair: Pair,
        basis: f64,
        cost: f64,
        deviation: f64,
    },
    Update {
        created_at: Timestamp,
        pair: Pair,
        entry_at: Timestamp,
        entry_basis: f64,
        basis: f64,
        cost: f64,
        deviation: f64,
    },
    Close {
        created_at: Timestamp,
        pair: Pair,
        entry_at: Timestamp,
        entry_basis: f64,
        basis: f64,
        cost: f64,
        deviation: f64,
    },
}

fn simulate<S>(strm: S) -> impl Stream<Item = Result<Event>>
where
    S: Stream<Item = Result<Signal>>,
{
    let threshold = 200.0;
    let holding_ms = 3 * 60 * 60 * 1000; // 3h in ms

    try_stream! {
        let mut strm = pin!(strm);
        let mut book = HashMap::<Pair,Position>::default();

        while let Some(sig) = strm.next().await {
            let sig = sig?;
            let now = sig.observed_at.clone();
            let pos = book.keys().cloned().collect::<Box<[_]>>();

            // close
            for pair in pos {
                if now.ms() <= book[&pair].entry_ts.ms() + holding_ms {
                    continue;
                }

                let pos = book.remove(&pair).unwrap();
                yield Event::Close {
                    created_at: now.clone(),
                    pair: pair.clone(),
                    entry_at: pos.entry_ts,
                    entry_basis: pos.entry_basis,
                    basis: pos.last_basis,
                    cost: pos.last_cost + pos.entry_cost,
                    deviation: pos.last_deviation,
               };
            }

            // open
            if !book.contains_key(&sig.pair) && sig.deviation > threshold {
                let pos = Position{
                    pair: sig.pair.clone(),
                    entry_basis: sig.basis,
                    entry_ts: now.clone(),
                    entry_cost: sig.cost,
                    last_basis: sig.basis,
                    last_ts: now.clone(),
                    last_cost: sig.cost,
                    last_deviation: sig.deviation,
                };
                book.insert(sig.pair.clone(), pos);

                yield Event::Open{
                    created_at: sig.observed_at,
                    pair: sig.pair.clone(),
                    basis: sig.basis,
                    cost: sig.cost,
                    deviation: sig.deviation,
                };
            }

            // update
            if let Some(pos) = book.get_mut(&sig.pair) {
                pos.last_ts = now;
                pos.last_basis = sig.basis;
                pos.last_deviation = sig.deviation;
                pos.last_cost = sig.cost;

                yield Event::Update{
                    created_at: now,
                    pair: pos.pair.clone(),
                    entry_at: pos.entry_ts.clone(),
                    entry_basis: pos.entry_basis,
                    basis: sig.basis,
                    cost: sig.cost,
                    deviation: sig.deviation,
                }
            }
        }
    }
}
