use core::fmt;
use std::array::from_fn;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::hash::Hash;
use std::io::Write;
use std::path::Path;
use std::pin::pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration as StdDuration, SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand};
use log::{error, info, warn};

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message as WsMessage, WebSocketUpgrade};
use axum::routing::get;

use async_stream::try_stream;
use futures::stream;
use futures::stream::{BoxStream, FuturesUnordered, SelectAll, Stream, StreamExt};
use futures::try_join;

use chrono::DateTime;

use serde::Serialize;
use serde_json::json;

use anyhow::Result;

use simba::simd::{SimdValue, f64x8};

use tokio::select;
use tokio::sync::Semaphore;
use tokio::task::spawn_blocking;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use prost::Message;

type Str = Cow<'static, str>;

mod pb {
    include!(concat!(env!("OUT_DIR"), "/events.rs"));
}
use pb::{
    Basis as PbBasis, Close as PbClose, Event, Open as PbOpen, Signal as PbSignal, Universe,
    event::Payload, universe::Market,
};

mod cex;
use cex::{kucoin_exchange_state, mexc_exchange_state, subscribe_kucoin_bbo, subscribe_mexc_bbo};

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

impl ToString for Instrument {
    fn to_string(&self) -> String {
        self.0.to_string()
    }
}

// global pair identifer: btc, eth,...
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize)]
struct Pair(Str);

impl ToString for Pair {
    fn to_string(&self) -> String {
        self.0.to_string()
    }
}

// ms unix utc timestamp
#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize)]
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

#[derive(Parser)]
#[command(name = "app", version, about = "spot/futures carry backtester")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Trade live
    Live,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // rustls has both `ring` (polars) and `aws-lc-rs` (reqwest) compiled in;
    // pick one explicitly or the websocket TLS handshake panics.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    match Cli::parse().command {
        Command::Live => live().await,
    }
}

async fn live() -> Result<()> {
    let exchange_update = StdDuration::from_hours(4);
    let timescale = StdDuration::from_mins(10);
    let holding = StdDuration::from_hours(3);

    // An empty source that is closed when Ctrl-C is received. Ending it
    // finalizes every downstream stage, so `live` can return.
    let strm = stream::pending().take_until(tokio::signal::ctrl_c());
    let strm = subscribe_exchange_state(exchange_update, strm);
    let strm = filter_pairs(strm);
    let strm = subscribe_bbo(strm);
    let strm = aggregate_quotes(timescale, strm);
    let strm = basis(strm);
    let strm = arma::<2, _>(timescale, strm);
    let strm = filter_signals(200.0, strm);
    let strm = simulate(holding, strm);
    let (strm, app) = serve(strm.boxed_local());
    let strm = persist("events/", 10_000_000, strm);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    let j = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            error!("serve failed: {e}");
        }
    });

    let mut strm = pin!(strm);
    let mut clock = tokio::time::interval(StdDuration::from_secs(1));
    let mut events = 0;
    loop {
        select! {
            _ = clock.tick() => {
                println!("{events} events/s");
                events = 0;
            }

            ev = strm.next() => {
                match ev {
                    Some(Ok(_)) => events += 1,
                    Some(Err(err)) => error!("stream error: {err}"),
                    None => {
                        info!("interrupted, streams finalized");
                        break;
                    }
                }
            }
        }
    }

    j.abort();
    Ok(())
}

fn mkid() -> String {
    static ID: AtomicU64 = AtomicU64::new(1);
    ID.fetch_add(1, Ordering::Relaxed).to_string()
}

fn subscribe_exchange_state<S>(interval: StdDuration, strm: S) -> impl Stream<Item = Result<Event>>
where
    S: Stream<Item = Result<Event>>,
{
    try_stream! {
        let mut strm = pin!(strm);
        let mut clock = tokio::time::interval(interval);
        let mut fail: Option<anyhow::Error> = None;

        loop {
            select!{
                ev = strm.next() => {
                    match ev {
                        Some(Ok(ev)) => yield ev,
                        Some(Err(e)) => fail = Some(e),
                        None => break,
                    }
                }

                _ = clock.tick() => {
                    let joined = try_join!(kucoin_exchange_state(), mexc_exchange_state());
                    let Ok((kucoin_assets, (mexc_assets, mexc_rate_limit))) = joined else {
                        error!("exchange state failed");
                        continue;
                    };

                    let kucoin_mkt = Market{
                        exchange: "kucoin".to_string(),
                        kind: Kind::Future.to_str().to_string(),
                        messages_per_second: 10,
                        subscriptions_per_connection: 25,
                        assets: kucoin_assets,
                    };
                    let mexc_mkt = Market{
                        exchange: "mexc".to_string(),
                        kind: Kind::Spot.to_str().to_string(),
                        messages_per_second: mexc_rate_limit,
                        subscriptions_per_connection: 30,
                        assets: mexc_assets,
                    };

                    yield Event{
                        id: mkid(),
                        timestamp: Timestamp::now().unwrap_or_default().ms(),
                        payload: Some(Payload::Universe(Universe{
                            markets: vec![kucoin_mkt, mexc_mkt],
                        })),
                    };
                }
            }

            if let Some(e) = fail.take() {
                Err(e)?;
            }
        }
    }
}

fn subscribe_bbo<S>(strm: S) -> impl Stream<Item = Result<Event>>
where
    S: Stream<Item = Result<Event>>,
{
    try_stream! {
        let mut strm = pin!(strm);
        // exchange -> (current pairs, cancel token for its bbo streams)
        let mut subs = HashMap::<String, (HashSet<Pair>, CancellationToken)>::default();
        let mut bbos = SelectAll::<BoxStream<'static, Result<Event>>>::new();
        let mut fail: Option<anyhow::Error> = None;

        loop {
            select!{
                ev = strm.next() => {
                    match ev {
                        Some(Ok(ev)) => {
                            if let Some(Payload::Universe(u)) = &ev.payload {
                                for m in &u.markets {
                                    let new: HashSet<Pair> = m
                                        .assets
                                        .iter()
                                        .map(|a| Pair(a.pair.clone().into()))
                                        .collect();
                                    let changed = subs
                                        .get(&m.exchange)
                                        .map(|(p, _)| p != &new)
                                        .unwrap_or(true);

                                    if changed {
                                        if let Some((_, c)) = subs.remove(&m.exchange) {
                                            c.cancel();
                                        }

                                        let cancel = CancellationToken::new();
                                        let per_conn = m.subscriptions_per_connection.max(1) as usize;
                                        for chunk in m.assets.chunks(per_conn) {
                                            let tickers = chunk
                                                .iter()
                                                .map(|a| {
                                                    (
                                                        Pair(a.pair.clone().into()),
                                                        Instrument(a.instrument.clone().into()),
                                                    )
                                                })
                                                .collect::<Vec<_>>();

                                            let s: BoxStream<'static, Result<Event>> = match m.exchange.as_str() {
                                                "kucoin" => subscribe_kucoin_bbo(tickers, cancel.clone()).boxed(),
                                                "mexc" => subscribe_mexc_bbo(tickers, m.messages_per_second, cancel.clone()).boxed(),
                                                other => panic!("unknown exchange {other}"),
                                            };
                                            bbos.push(s);
                                        }

                                        info!(
                                            "subscribed to {} {} symbols across {} connections",
                                            new.len(),
                                            m.exchange,
                                            bbos.len(),
                                        );
                                        subs.insert(m.exchange.clone(), (new, cancel));
                                    }
                                }
                            }
                            yield ev;
                        }
                        Some(Err(e)) => fail = Some(e),
                        None => break,
                    }
                }

                Some(q) = bbos.next() => {
                    match q {
                        Ok(q) => yield q,
                        Err(e) => fail = Some(e),
                    }
                }
            }

            if let Some(e) = fail.take() {
                Err(e)?;
            }
        }
    }
}

fn persist<S>(
    base: impl Into<String>,
    batch_size: usize,
    strm: S,
) -> impl Stream<Item = Result<Event>>
where
    S: Stream<Item = Result<Event>>,
{
    let base = base.into();

    try_stream! {
        let mut strm = pin!(strm);

        let mut fname = String::new();
        let mut raw = Vec::<u8>::default();
        let mut batch = 0usize;

        let flush = |fname: &str, raw: &mut Vec<u8>| -> Result<()> {
            if let Some(dir) = Path::new(fname).parent() {
                if !dir.as_os_str().is_empty() {
                    fs::create_dir_all(dir)?;
                }
            }
            let mut file = File::create(fname)?;
            file.write_all(raw)?;
            raw.clear();
            info!("{fname} written");
            Ok(())
        };

        while let Some(ev) = strm.next().await {
            let ev = ev?;

            // name the file after the first event of the batch
            if batch == 0 {
                let ts = DateTime::from_timestamp_millis(ev.timestamp as i64)
                    .map(|d| d.to_rfc3339())
                    .unwrap_or_default();
                fname = format!("{base}{ts}-{}.events.pb", mkid());
            }

            ev.encode_length_delimited(&mut raw)?;
            batch += 1;

            if batch >= batch_size {
                flush(&fname, &mut raw)?;
                batch = 0;
            }

            yield ev;
        }

        if batch > 0 {
            flush(&fname, &mut raw)?;
        }
    }
}

/*
fn quotes() -> impl Stream<Item = Result<Quote>> {
    // mexc silently drops every subscription beyond 30 channels on one connection
    let mexc_subscriptions_count = 30;
    let kucoin_subscriptions_count = 25;
    let poll_interval = StdDuration::from_secs(3600);

    stream! {
        let mut pairs = HashSet::<Pair>::default();
        let mut strms: Vec<BoxStream<'static, Quote>> = Vec::new();

        loop {
            let (Ok((mut mexc, rate_limit)), Ok(mut kucoin)) = join!(mexc_exchange_state(), kucoin_exchange_state()) else {
                error!("symbol update failed");
                sleep(poll_interval).await;
                continue;
            };

            // pairs listed on both venues
            let common: Vec<Pair> = mexc
                .keys()
                .filter(|p| kucoin.contains_key(*p))
                .cloned()
                .collect();
            let s = common.iter().cloned().collect::<HashSet<_>>();
            if strms.is_empty() || pairs != s {
                pairs = s;
                strms = common
                    .chunks(mexc_subscriptions_count)
                    .map(|chunk| {
                        let tickers = chunk.iter().map(|p| (p.clone(), mexc.remove(p).unwrap())).collect::<Vec<_>>();
                        subscribe_mexc_bbo(tickers, rate_limit, CancellationToken::new()).boxed()
                    })
                    .chain(
                        common
                        .chunks(kucoin_subscriptions_count)
                        .map(|chunk| {
                            let tickers = chunk.iter().map(|p| (p.clone(), kucoin.remove(p).unwrap())).collect::<Vec<_>>();
                            subscribe_kucoin_bbo(tickers, CancellationToken::new()).boxed()
                        })
                    )
                    .collect();
            }

            info!(
                "subscribe to {} pairs over {} connections",
                pairs.len(),
                strms.len()
            );

            let mut quotes = select_all(strms.iter_mut());
            let deadline = sleep(poll_interval);
            tokio::pin!(deadline);
            loop {
                select!{
                    _ = &mut deadline => break,
                    q = quotes.next() => {
                        if let Some(q) = q {
                            yield Ok(q);
                        }
                    }
                }
            }
        }
    }
}

fn resample<S>(interval: StdDuration, strm: S) -> impl Stream<Item = Result<Quote>>
where
    S: Stream<Item = Result<Quote>>,
{
    try_stream! {
        let mut strm = pin!(strm);
        let mut quotes = HashMap::<Instrument, Quote>::default();

        loop {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
            let phase = StdDuration::from_nanos((now.as_nanos() % interval.as_nanos()) as u64);
            let delay = interval - phase;
            let next = Timestamp((now + delay).as_millis() as u64);
            let mut fail = None;

            select!{
                _ = sleep(delay) => {
                    for (_, q) in quotes.iter() {
                        let mut q = q.clone();
                        q.observed_at = next;
                        q.received_at = Timestamp::now().unwrap();
                        q.is_aggregation =true;
                        yield q;
                    }
                }

                q = strm.next() => {
                    match q {
                        Some(Ok(q)) => {
                            quotes.insert(q.instrument.clone(), q.clone());
                            yield q;
                        }
                        Some(Err(e)) => fail = Some(e),
                        None => {}
                    }
                }
            }

            if let Some(e) = fail {
                Err(e)?;
            }
        }
    }
}

*/
/*
async fn history(spot: PathBuf, futures: PathBuf) -> Result<()> {
    let strm = simulate(arma(basis(join_causality(
        from_parquet("mexc", Kind::Spot, Duration::parse("10m"), spot)?,
        from_parquet("kucoin", Kind::Future, Duration::parse("10m"), futures)?,
    ))));
    let mut strm = pin!(strm);

    let mut pnl_bps = 0.0;
    let mut book = HashMap::<Pair, f64>::default();
    while let Some(ev) = strm.next().await {
        let ev = ev?;
        let ts = match ev {
            Position::Open {
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
            Position::Update { created_at, .. } => created_at,
            Position::Close {
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
    }

    Ok(())
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
    path: impl AsRef<std::path::Path>,
) -> Result<impl Stream<Item = Result<Quote>>> {
    // Fallible setup happens eagerly so scan errors surface as a normal Err.
    let path = path.as_ref().to_string_lossy().into_owned();
    let lf = LazyFrame::scan_parquet(PlRefPath::from(path.as_str()), ScanArgsParquet::default())?;
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

                let (instrument, pair) = instrument_and_pair(&venue,kind,sym,sym);

                yield Quote {
                    is_aggregation: false,
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

*/
fn aggregate_quotes<S>(interval: StdDuration, strm: S) -> impl Stream<Item = Result<Event>>
where
    S: Stream<Item = Result<Event>>,
{
    try_stream! {
        let mut strm = pin!(strm);
        // instrument -> most recent live (non-aggregated) quote event
        let mut quotes = HashMap::<String, Event>::default();

        loop {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
            let phase = StdDuration::from_nanos((now.as_nanos() % interval.as_nanos()) as u64);
            let delay = interval - phase;
            let next = Timestamp((now + delay).as_millis() as u64);
            let mut fail = None;

            select!{
                _ = tokio::time::sleep(delay) => {
                    for (_, q) in quotes.iter() {
                        let mut q = q.clone();
                        q.id = mkid();
                        q.timestamp = Timestamp::now().unwrap_or_default().ms();
                        if let Some(Payload::Quote(pq)) = q.payload.as_mut() {
                            pq.observed_at = next.ms();
                            pq.is_aggregation = true;
                        }
                        yield q;
                    }
                }

                ev = strm.next() => {
                    match ev {
                        Some(Ok(ev)) => {
                            if let Some(Payload::Quote(pq)) = &ev.payload {
                                if !pq.is_aggregation {
                                    quotes.insert(pq.instrument.clone(), ev.clone());
                                }
                            }
                            yield ev;
                        }
                        Some(Err(e)) => fail = Some(e),
                        None => break,
                    }
                }
            }

            if let Some(e) = fail {
                Err(e)?;
            }
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

/*
#[derive(Debug, Clone, Serialize)]
struct Basis {
    pub is_aggregation: bool,
    // latest input bar's closing ts
    pub observed_at: Timestamp,
    // inference ts
    pub created_at: Timestamp,
    pub pair: Pair,
    pub basis: f64,
    pub cost: f64,

    pub spot_bid: f64,
    pub spot_ask: f64,
    pub spot_mid: f64,
    pub spot_spread: f64,

    pub fut_bid: f64,
    pub fut_ask: f64,
    pub fut_mid: f64,
    pub fut_spread: f64,
}
*/

fn basis<S>(strm: S) -> impl Stream<Item = Result<Event>>
where
    S: Stream<Item = Result<Event>>,
{
    let half_spread_capture = 0.0;
    let fut_taker_bps = 6.0;
    let fut_maker_bps = 2.0;
    let spot_taker_bps = 0.0;
    let spot_maker_bps = 0.0;

    try_stream! {
        let mut live = HashMap::<Pair, Row>::default();
        let mut agg = HashMap::<Pair, Row>::default();
        let mut strm = pin!(strm);

        while let Some(r) = strm.next().await {
            let ev = r?;

            let Some(Payload::Quote(q)) = &ev.payload else {
                yield ev;
                continue
            };

            let pair = Pair(q.pair.clone().into());
            let observed_at = Timestamp(q.observed_at);

            let rows = if q.is_aggregation { &mut agg } else { &mut live };
            let row = rows.entry(pair.clone()).or_insert_with(Row::default);
            let ts = observed_at.masked_min();

            if row.ts > ts && q.is_aggregation {
                warn!("stale aggregation recv {:?}", q);
            }

            row.ts = ts;

            match Instrument(q.instrument.clone().into()).kind() {
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

            yield Event{
                id: mkid(),
                timestamp: Timestamp::now().unwrap_or_default().ms(),
                payload: Some(Payload::Basis(PbBasis{
                    is_aggregation: q.is_aggregation,
                    observed_at: row.ts.ms(),
                    pair: pair.to_string(),
                    basis: row.basis,
                    cost: row.cost,
                    spot_bid: row.spot_bid,
                    spot_ask: row.spot_ask,
                    spot_mid: row.spot_mid,
                    spot_spread: row.spot_spread,
                    fut_bid: row.fut_bid,
                    fut_ask: row.fut_ask,
                    fut_mid: row.fut_mid,
                    fut_spread: row.fut_spread,
                })),
            };
        }
    }
}

/*
#[derive(Debug, Clone, Serialize)]
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
*/

#[derive(Clone)]
struct InferenceAux {
    pub observed_at: Timestamp,
    pub pair: Pair,
    // basis net of round-trip cost, the series the ARMA is fitted on
    pub basis: f64,
    pub cost: f64,
}

fn arma<const W: usize, S>(interval: StdDuration, strm: S) -> impl Stream<Item = Result<Event>>
where
    S: Stream<Item = Result<Event>>,
{
    let max_iter = 100;
    let lambda = 10e-3;
    let delta_bic_threshold = 1.0;
    let bar_ms = interval.as_millis() as u64;

    // pairs co-fitted in a single SIMD ARMA call
    const LANES: usize = 8;

    try_stream! {
        // per-pair rolling window of basis minus two times entry cost (assumes entry = exit cost)
        let mut y = HashMap::<Pair, Box<[f64; W]>>::default();
        // last fitted ARMA params, used to warm-start the next window
        let mut phi = HashMap::<Pair, f64>::default();
        let mut theta = HashMap::<Pair, f64>::default();
        let mut aux = HashMap::<Pair, InferenceAux>::default();

        let mut ts: Option<Timestamp> = None;

        let mut res_strm = FuturesUnordered::new();
        let mut b_strm = pin!(strm);

        let sem = Arc::new(Semaphore::new(
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2),
        ));

        // tokio::select! evaluates each handler as plain code inside the
        // generated async block, so `?` cannot be used there; stash the error
        // and propagate it after the select.
        let mut fail: Option<anyhow::Error> = None;

        loop {
            select!{
                Some(r) = b_strm.next() => {
                    match r {
                        Err(e) => fail = Some(e),
                        Ok(ev) => {
                            if let Some(Payload::Basis(b)) = &ev.payload {
                                if !b.is_aggregation {
                                    yield ev;
                                    continue;
                                }

                                let bar = Timestamp(b.observed_at);

                                match ts {
                                    None => ts = Some(bar),
                                    Some(prev) if bar < prev => warn!("late quote: {:?}", b),
                                    Some(prev) if bar > prev => {
                                        let win = ((prev.ms() / bar_ms) % W as u64) as usize;
                                        let entries: Vec<(&Pair, &Box<[f64; W]>)> = y
                                            .iter()
                                            .filter(|(_, yw)| yw.iter().cloned().all(f64::is_finite))
                                            .collect();

                                        for chunk in entries.chunks(LANES) {
                                            // oldest -> newest for the window ending at `ts`
                                            let yv = (0..W)
                                                .map(|k| {
                                                    let idx = (win + 1 + k) % W;
                                                    f64x8::from(from_fn(|l| {
                                                        chunk
                                                            .get(l)
                                                            .map(|(_, yw)| yw[idx])
                                                            .unwrap_or(f64::NAN)
                                                    }))
                                                })
                                                .collect::<Vec<_>>();
                                            let mut phiv = [f64x8::from(from_fn(|l| {
                                                chunk
                                                    .get(l)
                                                    .and_then(|(p, _)| phi.get(*p).copied())
                                                    .filter(|v| v.is_finite())
                                                    .unwrap_or(0.0)
                                            }))];
                                            let mut thetav = [f64x8::from(from_fn(|l| {
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
                                        let i = ((bar.ms() / bar_ms) % W as u64) as usize;
                                        let pair = Pair(b.pair.clone().into());
                                        let wy = y
                                            .entry(pair.clone())
                                            .or_insert_with(|| Box::new([f64::NAN; W]));
                                        wy[i] = b.basis - b.cost;

                                        aux.insert(pair.clone(), InferenceAux {
                                            observed_at: bar,
                                            pair,
                                            basis: b.basis,
                                            cost: b.cost,
                                        });
                                    }
                                }
                            }

                            // pass every event through unchanged
                            yield ev;
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
                                    yield Event{
                                        id: mkid(),
                                        timestamp: now.ms(),
                                        payload: Some(Payload::Signal(PbSignal{
                                            window_start: aux.observed_at.ms().saturating_sub((W as u64 - 1) * bar_ms),
                                            window_end: aux.observed_at.ms(),
                                            deviation: net_dev,
                                            basis: aux.basis,
                                            cost: aux.cost,
                                            pair: aux.pair.to_string(),
                                        })),
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

fn filter_pairs<S>(strm: S) -> impl Stream<Item = Result<Event>>
where
    S: Stream<Item = Result<Event>>,
{
    try_stream! {
        let mut strm = pin!(strm);

        let mut has_futures = HashSet::<String>::default();
        let mut has_spot = HashSet::<String>::default();

        while let Some(r) = strm.next().await {
            let mut ev = r?;

            if let Some(Payload::Universe(u)) = &mut ev.payload {
                // XXX: delisted/halted handling?
                for mkt in &u.markets {
                    let pp = mkt.assets.iter().map(|a| a.pair.clone());
                    match mkt.kind.as_ref() {
                        "spot" => has_spot.extend(pp),
                        "fut" => has_futures.extend(pp),
                        k => warn!("unknown market kind {k}"),
                    }
                }

                for mkt in &mut u.markets {
                    mkt.assets.retain(|a| {
                        has_spot.contains(&a.pair) && has_futures.contains(&a.pair)
                    });
                }
            }

            yield ev;
        }
    }
}

fn filter_signals<S>(min_deviation: f64, strm: S) -> impl Stream<Item = Result<Event>>
where
    S: Stream<Item = Result<Event>>,
{
    try_stream! {
        let mut strm = pin!(strm);

        while let Some(r) = strm.next().await {
            let ev = r?;

            // drop signals below the entry threshold, pass everything else through
            if let Some(Payload::Signal(s)) = &ev.payload {
                if s.deviation <= min_deviation {
                    continue;
                }
            }

            yield ev;
        }
    }
}

struct ServeState {
    cancel: CancellationToken,
    latest: Mutex<HashMap<String, PbBasis>>,
    open: Mutex<HashMap<String, PbOpen>>,
    closed: Mutex<Vec<PbClose>>,
}

fn serve<S>(strm: S) -> (impl Stream<Item = Result<Event>>, Router)
where
    S: Stream<Item = Result<Event>>,
{
    let push_interval = StdDuration::from_millis(500);

    let cancel = CancellationToken::new();
    let state = Arc::new(ServeState {
        cancel: cancel.clone(),
        latest: Mutex::new(HashMap::new()),
        open: Mutex::new(HashMap::new()),
        closed: Mutex::new(Vec::new()),
    });
    let ws_handler = move |ws: WebSocketUpgrade, State(state): State<Arc<ServeState>>| async move {
        ws.on_upgrade(move |mut socket| async move {
            loop {
                let payload = {
                    let latest = state.latest.lock().unwrap();
                    let open = state.open.lock().unwrap();
                    let closed = state.closed.lock().unwrap();

                    json!({
                        "latest": &*latest,
                        "open": &*open,
                        "closed": &*closed,
                    })
                    .to_string()
                };
                if socket.send(WsMessage::Text(payload.into())).await.is_err() {
                    break;
                }
                select! {
                    _ = sleep(push_interval) => {}
                    _ = state.cancel.cancelled() => {
                        let _ = socket.send(WsMessage::Close(None)).await;
                        break;
                    }
                }
            }
        })
    };
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .with_state(state.clone());

    let strm = try_stream! {
        let mut strm = pin!(strm);

        while let Some(ev) = strm.next().await {
            let ev = ev?;

            match &ev.payload {
                Some(Payload::OpenPosition(o)) => {
                    state.open.lock().unwrap().insert(o.pair.clone(), o.clone());
                }
                Some(Payload::ClosePosition(c)) => {
                    state.open.lock().unwrap().remove(&c.pair);
                    state.closed.lock().unwrap().push(c.clone());
                }
                Some(Payload::Basis(b)) if !b.is_aggregation => {
                    state.latest.lock().unwrap().insert(b.pair.clone(), b.clone());
                }
                _ => {}
            }

            yield ev;
        }

        cancel.cancel();
    };

    (strm, app)
}

/*
fn persist_signals<S>(base: impl Into<String>, strm: S) -> impl Stream<Item = Result<Signal>>
where
    S: Stream<Item = Result<Signal>>,
{
    let batch_size = 10;
    let base = base.into();

    try_stream! {
        let mut strm = pin!(strm);

        let mut pair = Vec::<String>::with_capacity(batch_size);
        let mut observed_at = Vec::<i64>::with_capacity(batch_size);
        let mut created_at = Vec::<i64>::with_capacity(batch_size);
        let mut deviation = Vec::<f64>::with_capacity(batch_size);
        let mut cost = Vec::<f64>::with_capacity(batch_size);
        let mut basis = Vec::<f64>::with_capacity(batch_size);

        let mut batch_ts: Option<Timestamp> = None;

        let flush = |ts: Timestamp, mut df: DataFrame| -> Result<()> {
            if let Some(dir) = Path::new(&base).parent() {
                if !dir.as_os_str().is_empty() {
                    fs::create_dir_all(dir)?;
                }
            }
            let fname = format!("{base}-{ts}-{}.parquet", mkid());
            let file = File::create(&fname)?;
            ParquetWriter::new(file)
                .with_compression(ParquetCompression::Zstd(None))
                .with_row_group_size(Some(batch_size.max(64 * 1024)))
                .finish(&mut df)?;
            info!("{fname} written");
            Ok(())
        };

        while let Some(s) = strm.next().await {
            let s = s?;

            pair.push(s.pair.0.to_string());
            observed_at.push(s.observed_at.ms() as i64);
            created_at.push(s.created_at.ms() as i64);
            deviation.push(s.deviation);
            cost.push(s.cost);
            basis.push(s.basis);

            batch_ts = Some(batch_ts.map(|t| t.min(s.observed_at)).unwrap_or(s.observed_at));

            if pair.len() >= batch_size {
                let df = df! {
                    "pair" => mem::take(&mut pair),
                    "observed_at" => mem::take(&mut observed_at),
                    "created_at" => mem::take(&mut created_at),
                    "deviation" => mem::take(&mut deviation),
                    "cost" => mem::take(&mut cost),
                    "basis" => mem::take(&mut basis),
                }?;
                flush(batch_ts.unwrap_or_default(), df)?;
                batch_ts = None;
            }

            yield s;
        }

        if !pair.is_empty() {
            let df = df! {
                "pair" => mem::take(&mut pair),
                "observed_at" => mem::take(&mut observed_at),
                "created_at" => mem::take(&mut created_at),
                "deviation" => mem::take(&mut deviation),
                "cost" => mem::take(&mut cost),
                "basis" => mem::take(&mut basis),
            }?;
            flush(batch_ts.unwrap_or_default(), df)?;
        }
    }
}

*/
struct SimPosition {
    entry_basis: f64,
    entry_ts: Timestamp,
    entry_cost: f64,
    entry_deviation: f64,
    last_basis: f64,
    last_deviation: f64,
    last_cost: f64,
    last_ts: Timestamp,
}

/*
#[derive(Debug, Serialize)]
enum Position {
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
*/

fn simulate<S>(holding: StdDuration, strm: S) -> impl Stream<Item = Result<Event>>
where
    S: Stream<Item = Result<Event>>,
{
    try_stream! {
        let holding_ms = holding.as_millis() as u64;
        let mut strm = pin!(strm);
        let mut book = HashMap::<Pair,SimPosition>::default();

        while let Some(r) = strm.next().await {
            let ev = r?;

            if let Some(Payload::Signal(sig)) = &ev.payload {
                let pair = Pair(sig.pair.clone().into());
                let now = Timestamp(sig.window_end);
                let pos = book.keys().cloned().collect::<Box<[_]>>();

                // close positions held past the holding period
                for pair in pos {
                    if now.ms() <= book[&pair].entry_ts.ms() + holding_ms {
                        continue;
                    }

                    let pos = book.remove(&pair).unwrap();
                    yield Event{
                        id: mkid(),
                        timestamp: now.ms(),
                        payload: Some(Payload::ClosePosition(PbClose{
                            pair: pair.to_string(),
                            entry_at: pos.entry_ts.ms(),
                            entry_basis: pos.entry_basis,
                            entry_deviation: pos.entry_deviation,
                            entry_cost: pos.entry_cost,
                            exit_at: now.ms(),
                            exit_basis: pos.last_basis,
                            exit_cost: pos.last_cost,
                            exit_deviation: pos.last_deviation,
                        })),
                    };
                }

                // open
                if !book.contains_key(&pair) {
                    let pos = SimPosition{
                        entry_basis: sig.basis,
                        entry_ts: now,
                        entry_cost: sig.cost,
                        entry_deviation: sig.deviation,
                        last_basis: sig.basis,
                        last_ts: now,
                        last_cost: sig.cost,
                        last_deviation: sig.deviation,
                    };
                    book.insert(pair.clone(), pos);

                    yield Event{
                        id: mkid(),
                        timestamp: now.ms(),
                        payload: Some(Payload::OpenPosition(PbOpen{
                            pair: pair.to_string(),
                            basis: sig.basis,
                            cost: sig.cost,
                            deviation: sig.deviation,
                        })),
                    };
                }

                // update running position
                if let Some(pos) = book.get_mut(&pair) {
                    pos.last_ts = now;
                    pos.last_basis = sig.basis;
                    pos.last_deviation = sig.deviation;
                    pos.last_cost = sig.cost;
                }
            }

            // pass every event through unchanged
            yield ev;
        }
    }
}

/*
struct ServePositionsState {
    cancel: CancellationToken,
    open: Mutex<HashMap<Pair, ServePosition>>,
    closed: Mutex<Vec<ServePosition>>,
}

#[derive(Debug, Serialize)]
struct ServePosition {
    pair: Pair,
    entry_basis: f64,
    entry_ts: Timestamp,
    entry_cost: f64,
    last_basis: f64,
    last_deviation: f64,
    last_cost: f64,
    last_ts: Timestamp,
    close_basis: Option<f64>,
    close_deviation: Option<f64>,
    close_cost: Option<f64>,
    close_ts: Option<Timestamp>,
}

fn serve_positions<S>(strm: S) -> (impl Stream<Item = Result<Position>>, Router)
where
    S: Stream<Item = Result<Position>>,
{
    let push_interval = StdDuration::from_millis(500);

    let cancel = CancellationToken::new();
    let state = Arc::new(ServePositionsState {
        cancel: cancel.clone(),
        open: Mutex::new(HashMap::<Pair, ServePosition>::new()),
        closed: Mutex::new(vec![]),
    });
    let ws_handler = move |ws: WebSocketUpgrade, State(state): State<Arc<ServePositionsState>>| async move {
        ws.on_upgrade(move |mut socket| async move {
            loop {
                let payload = {
                    let open = state.open.lock().unwrap();
                    let closed = state.closed.lock().unwrap();
                    json!({ "open": &*open, "closed": &*closed }).to_string()
                };
                if socket.send(WsMessage::Text(payload.into())).await.is_err() {
                    break;
                }
                select! {
                    _ = sleep(push_interval) => {}
                    _ = state.cancel.cancelled() => {
                        let _ = socket.send(WsMessage::Close(None)).await;
                        break;
                    }
                }
            }
        })
    };
    let app = Router::new()
        .route("/p", get(ws_handler))
        .with_state(state.clone());

    let strm = try_stream! {
        let mut strm = pin!(strm);


        while let Some(e) = strm.next().await {
            let e = e?;

            match &e {
                Position::Open { created_at, pair, basis, cost, deviation } => {
                    let pos = ServePosition {
                        pair: pair.clone(),
                        entry_basis: *basis,
                        entry_ts: created_at.clone(),
                        entry_cost: *cost,
                        last_basis: *basis,
                        last_deviation: *deviation,
                        last_cost: *cost,
                        last_ts: created_at.clone(),
                        close_basis: None,
                        close_deviation: None,
                        close_cost: None,
                        close_ts: None,
                    };
                    state.open.lock().unwrap().insert(pair.clone(), pos);
                }
                Position::Update { created_at, pair, basis, cost, deviation, .. } => {
                    let mut open = state.open.lock().unwrap();
                    if let Some(pos) = open.get_mut(pair) {
                        pos.last_basis = *basis;
                        pos.last_deviation = *deviation;
                        pos.last_cost = *cost;
                        pos.last_ts = created_at.clone();
                    }
                }
                Position::Close { created_at, pair, basis, cost, deviation, .. } => {
                    let mut pos = state.open.lock().unwrap().remove(pair);
                    if let Some(pos) = pos.as_mut() {
                        pos.last_basis = *basis;
                        pos.last_deviation = *deviation;
                        pos.last_ts = created_at.clone();
                        pos.close_basis = Some(*basis);
                        pos.close_deviation = Some(*deviation);
                        pos.close_cost = Some(*cost);
                        pos.close_ts = Some(created_at.clone());
                    }
                    if let Some(pos) = pos {
                        state.closed.lock().unwrap().push(pos);
                    }
                }
            }

            yield e;
        }

        cancel.cancel();
    };

    (strm, app)
}
*/
