use core::fmt;
use std::array::from_fn;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::hash::Hash;
use std::mem;
use std::path::{Path, PathBuf};
use std::pin::pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration as StdDuration, Instant, SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand};
use governor::{Quota, RateLimiter};
use log::{error, info, warn};
use nonzero_ext::nonzero;

use async_stream::{stream, try_stream};
use futures::stream::{BoxStream, FuturesUnordered, Stream, StreamExt, select_all};
use futures::{SinkExt, join};

use chrono::DateTime;

use polars::prelude::{
    ClosedWindow, DataFrame, Duration, DynamicGroupOptions, Engine, FillNullStrategy, JoinArgs,
    JoinType, Label, LazyFrame, ParquetCompression, ParquetWriter, PlRefPath, ScanArgsParquet,
    SortMultipleOptions, StartBy, TimeUnit, col, datetime_range, df, lit,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use anyhow::{Result, anyhow};

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message as WsMessage, WebSocketUpgrade};
use axum::routing::get;

use simba::simd::{SimdValue, f64x8};

use tokio::select;
use tokio::sync::Semaphore;
use tokio::task::spawn_blocking;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;

use prost::Message as _;

type Str = Cow<'static, str>;

pub mod mexcpb {
    include!(concat!(env!("OUT_DIR"), "/mexc_proto.rs"));
}
use mexcpb::PushDataV3ApiWrapper;

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
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize)]
struct Pair(Str);

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

#[derive(Debug, Clone)]
struct Quote {
    pub is_aggregation: bool,
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

/// Generic KuCoin API envelope: `{ "code": "200000", "data": ... }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KucoinResponse<T> {
    code: String,
    data: T,
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
    #[serde(default)]
    rate_limits: Vec<MexcRateLimit>,
    symbols: Vec<MexcSymbol>,
}

/// A rate-limit rule from `exchangeInfo.rateLimits`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MexcRateLimit {
    rate_limit_type: String,
    interval: String,
    interval_num: u32,
    limit: u32,
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

#[derive(Parser)]
#[command(name = "app", version, about = "spot/futures carry backtester")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Replay historical order books from parquet files
    History {
        /// spot order-book parquet path
        #[arg(short, long)]
        spot: PathBuf,
        /// futures order-book parquet path
        #[arg(short, long)]
        futures: PathBuf,
    },
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
        Command::History { spot, futures } => history(spot, futures).await,
        Command::Live => live().await,
    }
}

async fn live() -> Result<()> {
    let strm = simulate(arma(serve(
        "0.0.0.0:3000",
        basis(resample(
            StdDuration::from_secs(600),
            persist_quotes("pq/bbo", quotes()),
        )),
    )));
    let mut strm = pin!(strm);
    while let Some(x) = strm.next().await {
        println!("{:?}", x);
    }
    Ok(())
}

/// kucoin sends price/qty as strings, e.g. `[["246.72", "40"]]`
fn de_levels<'de, D>(de: D) -> std::result::Result<Vec<(f64, f64)>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Vec::<(String, String)>::deserialize(de)?;
    Ok(raw
        .into_iter()
        .map(|(px, qty)| (px.parse().unwrap_or(f64::NAN), qty.parse().unwrap_or(0.0)))
        .collect())
}

#[derive(Deserialize)]
struct KuCoinBboEvent {
    #[serde(rename = "d")]
    data: KuCoinOrderBook,
}

#[derive(Deserialize)]
struct KuCoinOrderBook {
    #[serde(rename = "a", deserialize_with = "de_levels")]
    ask: Vec<(f64, f64)>,
    #[serde(rename = "b", deserialize_with = "de_levels")]
    bid: Vec<(f64, f64)>,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "M")]
    timestamp: u64,
}

impl Quote {
    pub fn from_kucoin(p: Pair, i: Instrument, ev: KuCoinBboEvent) -> Quote {
        let (bid, bid_qty) = ev.data.bid[0];
        let (ask, ask_qty) = ev.data.ask[0];
        Quote {
            is_aggregation: false,
            instrument: i,
            pair: p,
            received_at: Timestamp::now().unwrap_or_default(),
            // M is nanoseconds
            observed_at: Timestamp(ev.data.timestamp / 1_000_000),
            bid,
            bid_qty: bid_qty.max(0.0) as u64,
            ask,
            ask_qty: ask_qty.max(0.0) as u64,
        }
    }

    fn from_mexc(p: Pair, i: Instrument, ev: PushDataV3ApiWrapper) -> Quote {
        let mut q = Quote {
            is_aggregation: false,
            instrument: i,
            pair: p,
            received_at: Timestamp::now().unwrap_or_default(),
            // sendTime is milliseconds
            observed_at: Timestamp(ev.send_time.unwrap_or_default() as u64),
            bid: f64::NAN,
            ask: f64::NAN,
            bid_qty: 0,
            ask_qty: 0,
        };
        if let Some(mexcpb::push_data_v3_api_wrapper::Body::PublicAggreBookTicker(b)) = ev.body {
            q.bid = b.bid_price.parse().unwrap_or(f64::NAN);
            q.bid_qty = b.bid_quantity.parse::<f64>().unwrap_or(0.0).max(0.0) as u64;
            q.ask = b.ask_price.parse().unwrap_or(f64::NAN);
            q.ask_qty = b.ask_quantity.parse::<f64>().unwrap_or(0.0).max(0.0) as u64;
        }
        q
    }
}

#[derive(Serialize)]
struct MexcSubscribe {
    method: String,
    params: Vec<String>,
    id: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct KuCoinSubscribe {
    id: String,
    action: String,
    channel: String,
    trade_type: String,
    symbol: String,
    depth: String,
    rpi_filter: usize,
}

fn mkid() -> String {
    static ID: AtomicU64 = AtomicU64::new(1);
    ID.fetch_add(1, Ordering::Relaxed).to_string()
}

fn subscribe_kucoin_bbo(
    tickers: Vec<(Pair, Instrument)>,
    cancel: CancellationToken,
) -> impl Stream<Item = Quote> {
    let handshake_timeout = StdDuration::from_secs(10);
    let max_attempts: usize = 3;

    let conn_limit = RateLimiter::direct(Quota::per_second(nonzero!(1u32)));

    stream! {
        'outer: loop {
            conn_limit.until_ready().await;

            let Ok((conn, _)) = connect_async("wss://x-push-futures.kucoin.com").await else {
                error!("kucoin connection failed");
                continue;
            };
            let (mut tx, mut rx) = conn.split();

            // read welcome
            let Ok(Some(Ok(welmsg))) = timeout(handshake_timeout, rx.next()).await else {
                error!("kucoin handshake failed");
                continue;
            };
            let Some(wel) = welmsg.into_text().ok().and_then(|s| serde_json::from_str::<Value>(s.as_str()).ok()) else {
                error!("kucoin handshake ill-formed");
                continue;
            };

            let ping_interval = StdDuration::from_millis(wel.get("pingInterval").and_then(|v| v.as_u64()).unwrap_or(18_000));
            // no burst: a 30-message burst makes kucoin reject some subscriptions
            let sub_limit = RateLimiter::direct(Quota::per_second(nonzero!(30u32)).allow_burst(nonzero!(1u32)));
            let mut out_queue = VecDeque::<(usize, KuCoinSubscribe)>::default();
            // id -> (deadline, json)
            let mut ack_queue = HashMap::<String, (Instant, usize, KuCoinSubscribe)>::default();
            // lowercased venue symbol -> (pair, instrument)
            let lookup: HashMap<String, (Pair, Instrument)> = tickers
                .iter()
                .map(|(p, i)| {
                    let symbol = i.0.rsplit(':').next().unwrap_or_default().to_lowercase();
                    (symbol, (p.clone(), i.clone()))
                })
                .collect();

            // fill out queue with subscription msgs
            out_queue.extend(tickers.iter().map(|(_, i)| (0, KuCoinSubscribe {
                id: mkid(),
                action: "subscribe".to_string(),
                channel: "obu".to_string(),
                trade_type: "FUTURES".to_string(),
                symbol: i.0.rsplit(':').next().unwrap_or_default().to_uppercase(),
                depth: "1".to_string(),
                rpi_filter: 0,
            })));

            loop {
                let next_resend = ack_queue
                    .values()
                    .map(|(t, _, _)| *t)
                    .min()
                    .map(|t| t.saturating_duration_since(Instant::now()))
                    .unwrap_or_default();

                select!{
                    _ = cancel.cancelled() => return,

                    _ = sub_limit.until_ready(), if !out_queue.is_empty() => {
                        let (attempts, msg) = out_queue.pop_front().unwrap();
                        match serde_json::to_string(&msg) {
                            Ok(payload) => {
                                if let Err(e) = tx.send(Message::text(payload)).await {
                                    error!("kucoin send failed: {e}");
                                    continue;
                                }
                                ack_queue.insert(msg.id.clone(), (Instant::now() + StdDuration::from_secs(5), attempts, msg));
                            }
                            Err(e) => error!("kucoin encode failed: {e}"),
                        }
                    }

                    _ = sleep(next_resend), if !ack_queue.is_empty() => {
                        let now = Instant::now();
                        let expired: Vec<String> = ack_queue
                            .iter()
                            .filter(|(_, (t, _, _))| *t <= now)
                            .map(|(k, _)| k.clone())
                            .collect();
                        for k in expired {
                            if let Some((_, attempts, msg)) = ack_queue.remove(&k) {
                                warn!("kucoin resend (no ack): {}", msg.symbol);
                                out_queue.push_back((attempts, msg));
                            }
                        }
                    }

                    m = rx.next() => {
                        let s = match m {
                            Some(Ok(Message::Text(s))) => s,
                            // kucoin pushes its JSON payloads as binary frames
                            Some(Ok(Message::Binary(payload))) => match String::from_utf8(payload.to_vec()) {
                                Ok(s) => s.into(),
                                Err(e) => {
                                    error!("kucoin ill-formed binary: {e}");
                                    continue
                                }
                            },
                            Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => continue,
                            Some(Ok(Message::Frame(_))) => {
                                warn!("kucoin frame");
                                continue
                            }
                            Some(Ok(Message::Close(_))) | None => {
                                error!("kucoin disconnected");
                                continue 'outer;
                            }
                            Some(Err(err)) => {
                                error!("kucoin recv failed: {}", err);
                                continue 'outer;
                            }
                        };
                        let Ok(m) = serde_json::from_str::<Value>(s.as_str()) else {
                            error!("kucoin ill-formed message {s}");
                            continue;
                        };
                        if m.get("T").and_then(|v| v.as_str()) == Some("obu.FUTURES") {
                            let Ok(ev) = serde_json::from_value::<KuCoinBboEvent>(m) else {
                                error!("kucoin ill-formed obu message");
                                continue;
                            };
                            if ev.data.bid.is_empty() || ev.data.ask.is_empty() {
                                continue;
                            }
                            let key = ev.data.symbol.to_lowercase();
                            let Some((p, i)) = lookup.get(&key) else {
                                error!("kucoin unknown symbol {}", ev.data.symbol);
                                continue;
                            };
                            yield Quote::from_kucoin(p.clone(), i.clone(), ev);
                        } else if let Some(r) = m.get("result") {
                            // remove from ack
                            let id = m.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                            match ack_queue.remove(id) {
                                Some((_, attempts, sub)) => {
                                    if r.as_bool() == Some(true) {
                                        // subscribed
                                    } else if attempts + 1 < max_attempts {
                                        error!("kucoin subscription failed, retrying: {}", sub.symbol);
                                        out_queue.push_back((attempts + 1, sub));
                                    } else {
                                        error!("kucoin subscription failed, giving up: {}", sub.symbol);
                                    }
                                }
                                None => error!("kucoin ack for unknown msg"),
                            }
                        } else if m.get("op").and_then(|v| v.as_str()) == Some("pong") {
                            // do nothing
                        } else {
                            error!("kucoin unknown msg shape {m:?}");
                        }
                    }

                    /* send ping (ignore limit) */
                    _ = sleep(ping_interval) => {
                        let _ = tx.send(Message::text(json!({"id": mkid(), "op": "ping"}).to_string())).await;
                    }
                }
            }
        }
    }
}

fn subscribe_mexc_bbo(
    tickers: Vec<(Pair, Instrument)>,
    rate_limit: u32,
    cancel: CancellationToken,
) -> impl Stream<Item = Quote> {
    let max_attempts: usize = 3;

    let conn_limit = RateLimiter::direct(Quota::per_second(nonzero!(1u32)));

    stream! {
        'outer: loop {
            conn_limit.until_ready().await;

            let Ok((conn, _)) = connect_async("wss://wbs-api.mexc.com/ws").await else {
                error!("mexc connection failed");
                continue;
            };
            let (mut tx, mut rx) = conn.split();

            // mexc sends no welcome frame; it just drops the connection if it sees no PING within 30s
            let ping_interval = StdDuration::from_secs(30);
            let quota = Quota::per_second(std::num::NonZeroU32::new(rate_limit).unwrap_or(nonzero!(1u32)));
            let limit = RateLimiter::direct(quota.allow_burst(nonzero!(1u32)));
            let mut out_queue = VecDeque::<(usize, MexcSubscribe)>::default();
            // id -> (deadline, attempts, sub)
            let mut ack_queue = HashMap::<String, (Instant, usize, MexcSubscribe)>::default();
            // lowercased venue symbol -> (pair, instrument)
            let lookup: HashMap<String, (Pair, Instrument)> = tickers
                .iter()
                .map(|(p, i)| {
                    let symbol = i.0.rsplit(':').next().unwrap_or_default().to_lowercase();
                    (symbol, (p.clone(), i.clone()))
                })
                .collect();

            // fill out queue with subscription msgs
            out_queue.extend(tickers.iter().map(|(_, i)| (0, MexcSubscribe {
                method: "SUBSCRIPTION".to_string(),
                params: vec![format!("spot@public.aggre.bookTicker.v3.api.pb@100ms@{}", i.0.rsplit(':').next().unwrap_or_default().to_uppercase())],
                id: mkid(),
            })));

            loop {
                let next_resend = ack_queue
                    .values()
                    .map(|(t, _, _)| *t)
                    .min()
                    .map(|t| t.saturating_duration_since(Instant::now()))
                    .unwrap_or_default();

                select!{
                    _ = cancel.cancelled() => return,

                    _ = limit.until_ready(), if !out_queue.is_empty() => {
                        let (attempts, msg) = out_queue.pop_front().unwrap();
                        match serde_json::to_string(&msg) {
                            Ok(payload) => {
                                if let Err(e) = tx.send(Message::text(payload)).await {
                                    error!("mexc send failed: {e}");
                                    continue;
                                }
                                ack_queue.insert(msg.id.clone(), (Instant::now() + StdDuration::from_secs(5), attempts, msg));
                            }
                            Err(e) => error!("mexc encode failed: {e}"),
                        }
                    }

                    _ = sleep(next_resend), if !ack_queue.is_empty() => {
                        let now = Instant::now();
                        let expired: Vec<String> = ack_queue
                            .iter()
                            .filter(|(_, (t, _, _))| *t <= now)
                            .map(|(k, _)| k.clone())
                            .collect();
                        for k in expired {
                            if let Some((_, attempts, msg)) = ack_queue.remove(&k) {
                                warn!("mexc resend (no ack): {}", msg.params[0]);
                                out_queue.push_back((attempts, msg));
                            }
                        }
                    }

                    m = rx.next() => {
                        let m = match m {
                            Some(Ok(m)) => m,
                            Some(Err(err)) => {
                                error!("mexc recv failed: {}", err);
                                continue 'outer;
                            }
                            None => {
                                error!("mexc disconnected");
                                continue 'outer;
                            }
                        };
                        let m = match m {
                            Message::Binary(payload) => {
                                let Ok(ev) = PushDataV3ApiWrapper::decode(&payload[..]) else {
                                    error!("mexc ill-formed push");
                                    continue;
                                };
                                let key = ev.symbol.as_deref().unwrap_or_default().to_lowercase();
                                let Some((p, i)) = lookup.get(&key) else {
                                    error!("mexc unknown symbol {:?}", ev.symbol);
                                    continue;
                                };
                                yield Quote::from_mexc(p.clone(), i.clone(), ev);
                                continue;
                            }
                            Message::Text(m) => m,
                            _ => continue,
                        };
                        let Ok(m) = serde_json::from_str::<Value>(m.as_str()) else {
                            error!("mexc ill-formed message {m}");
                            continue;
                        };
                        if m.get("msg").and_then(|v| v.as_str()) == Some("PONG") {
                            // do nothing
                        } else if let Some(code) = m.get("code") {
                            let id = m.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                            match ack_queue.remove(id) {
                                Some((_, attempts, sub)) => {
                                    if code.as_i64() == Some(0) {
                                        // subscribed
                                    } else if attempts + 1 < max_attempts {
                                        error!("mexc subscription failed, retrying: {}", sub.params[0]);
                                        out_queue.push_back((attempts + 1, sub));
                                    } else {
                                        error!("mexc subscription failed, giving up: {}", sub.params[0]);
                                    }
                                }
                                None => error!("mexc ack for unknown msg"),
                            }
                        } else {
                            error!("mexc unknown msg shape {m:?}");
                        }
                    }

                    /* send ping (ignore limit) */
                    _ = sleep(ping_interval) => {
                        let _ = tx.send(Message::text(json!({"method": "PING"}).to_string())).await;
                    }
                }
            }
        }
    }
}

struct ServeState {
    cancel: CancellationToken,
    latest: Mutex<HashMap<Pair, Basis>>,
}

fn serve<S>(listen: impl Into<String>, strm: S) -> impl Stream<Item = Result<Basis>>
where
    S: Stream<Item = Result<Basis>>,
{
    let listen = listen.into();
    let push_interval = StdDuration::from_millis(500);

    try_stream! {
        let mut strm = pin!(strm);

        let cancel = CancellationToken::new();
        let state = Arc::new(ServeState {
            cancel: cancel.clone(),
            latest: Mutex::new(HashMap::<Pair, Basis>::new()),
        });

        let ws_handler =
            move |ws: WebSocketUpgrade, State(state): State<Arc<ServeState>>| async move {
                ws.on_upgrade(move |mut socket| async move {
                    loop {
                        let payload = {
                            let latest = state.latest.lock().unwrap();
                            serde_json::to_string(&*latest).unwrap_or_else(|_| "{}".to_string())
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
        let listener = tokio::net::TcpListener::bind(&listen).await?;

        let shutdown = cancel.clone();
        let h = tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app)
                .with_graceful_shutdown(shutdown.cancelled_owned())
                .await
            {
                error!("serve failed: {e}");
            }
        });

        while let Some(q) = strm.next().await {
            let q = q?;

            if !q.is_aggregation {
                state.latest.lock().unwrap().insert(q.pair.clone(), q.clone());
            }
            yield q;
        }

        cancel.cancel();
        h.await?;
    }
}

fn persist_quotes<S>(base: impl Into<String>, strm: S) -> impl Stream<Item = Result<Quote>>
where
    S: Stream<Item = Result<Quote>>,
{
    let batch_size = 10_000_000;
    let base = base.into();

    try_stream! {
        let mut strm = pin!(strm);

        let mut instrument = Vec::<String>::with_capacity(batch_size);
        let mut pair = Vec::<String>::with_capacity(batch_size);
        let mut received_at = Vec::<i64>::with_capacity(batch_size);
        let mut observed_at = Vec::<i64>::with_capacity(batch_size);
        let mut ask = Vec::<f64>::with_capacity(batch_size);
        let mut ask_qty = Vec::<i64>::with_capacity(batch_size);
        let mut bid = Vec::<f64>::with_capacity(batch_size);
        let mut bid_qty = Vec::<i64>::with_capacity(batch_size);

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

        while let Some(q) = strm.next().await {
            let q = q?;
            if q.is_aggregation {
                yield q;
                continue;
            }

            instrument.push(q.instrument.0.to_string());
            pair.push(q.pair.0.to_string());
            received_at.push(q.received_at.ms() as i64);
            observed_at.push(q.observed_at.ms() as i64);
            ask.push(q.ask);
            ask_qty.push(q.ask_qty as i64);
            bid.push(q.bid);
            bid_qty.push(q.bid_qty as i64);

            batch_ts = Some(batch_ts.map(|t| t.min(q.observed_at)).unwrap_or(q.observed_at));

            if instrument.len() >= batch_size {
                let df = df! {
                    "instrument" => mem::take(&mut instrument),
                    "pair" => mem::take(&mut pair),
                    "received_at" => mem::take(&mut received_at),
                    "observed_at" => mem::take(&mut observed_at),
                    "ask" => mem::take(&mut ask),
                    "ask_qty" => mem::take(&mut ask_qty),
                    "bid" => mem::take(&mut bid),
                    "bid_qty" => mem::take(&mut bid_qty),
                }?;
                flush(batch_ts.unwrap_or_default(), df)?;
                batch_ts = None;
            }

            yield q;
        }

        if !instrument.is_empty() {
            let df = df! {
                "instrument" => mem::take(&mut instrument),
                "pair" => mem::take(&mut pair),
                "received_at" => mem::take(&mut received_at),
                "observed_at" => mem::take(&mut observed_at),
                "ask" => mem::take(&mut ask),
                "ask_qty" => mem::take(&mut ask_qty),
                "bid" => mem::take(&mut bid),
                "bid_qty" => mem::take(&mut bid_qty),
            }?;
            flush(batch_ts.unwrap_or_default(), df)?;
        }
    }
}

async fn kucoin_exchange_state() -> Result<HashMap<Pair, Instrument>> {
    let res = reqwest::get("https://api-futures.kucoin.com/api/v1/contracts/active")
        .await?
        .json::<KucoinResponse<Vec<KucoinFutureSymbol>>>()
        .await?;
    let s: HashMap<Pair, Instrument> = res
        .data
        .into_iter()
        .filter_map(|s| {
            if s.status != "Open" {
                return None;
            }
            if s.quote_currency.to_lowercase() != "usdt" {
                return None;
            }
            let (i, p) = instrument_and_pair("kucoin", Kind::Future, &s.base_currency, &s.symbol);
            Some((p, i))
        })
        .collect();
    Ok(s)
}

async fn mexc_exchange_state() -> Result<(HashMap<Pair, Instrument>, u32)> {
    let res = reqwest::get("https://api.mexc.com/api/v3/exchangeInfo")
        .await?
        .json::<MexcExchangeInfo>()
        .await?;

    let rate_limit = res
        .rate_limits
        .iter()
        .find(|r| r.interval.eq_ignore_ascii_case("second"))
        .map(|r| r.limit)
        .unwrap_or(30);

    let ret: HashMap<Pair, Instrument> = res
        .symbols
        .into_iter()
        .filter_map(|s| {
            if s.quote_asset.to_lowercase() != "usdt" {
                return None;
            }
            if !s.is_spot_trading_allowed {
                return None;
            }
            if s.status != "1" {
                return None;
            }
            let (i, p) = instrument_and_pair("mexc", Kind::Spot, &s.base_asset, &s.symbol);
            Some((p, i))
        })
        .collect();
    Ok((ret, rate_limit))
}

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
    }

    Ok(())
}

fn instrument_and_pair(
    venue: impl AsRef<str>,
    kind: Kind,
    base: impl AsRef<str>,
    symbol: impl AsRef<str>,
) -> (Instrument, Pair) {
    let symbol = symbol.as_ref().to_lowercase();
    let venue = venue.as_ref().to_lowercase();
    let base = base.as_ref().to_lowercase();

    match (venue.as_ref(), kind, base.as_ref()) {
        ("mexc", Kind::Spot, "gme") => (
            Instrument::new(Kind::Spot, "mexc", symbol),
            Pair("gmememe".into()),
        ),
        ("mexc", Kind::Spot, "wen") => (
            Instrument::new(Kind::Spot, "mexc", symbol),
            Pair("wenmeme".into()),
        ),
        (venue, kind, _) => (
            Instrument::new(kind, venue, symbol),
            Pair(base.to_string().into()),
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
        let mut live = HashMap::<Pair, Row>::default();
        let mut agg = HashMap::<Pair, Row>::default();
        let mut strm = pin!(strm);

        while let Some(r) = strm.next().await {
            let q = r?;
            let rows = if q.is_aggregation { &mut agg } else { &mut live };
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

            let basis = Basis{
                is_aggregation: q.is_aggregation,
                observed_at: row.ts,
                created_at: Timestamp::now()?,
                pair: q.pair.clone(),
                cost: row.cost,
                basis : row.basis,

                spot_bid: row.spot_bid,
                spot_ask: row.spot_ask,
                spot_mid: row.spot_mid,
                spot_spread: row.spot_spread,

                fut_bid: row.fut_bid,
                fut_ask: row.fut_ask,
                fut_mid: row.fut_mid,
                fut_spread: row.fut_spread,
            };

            yield basis;
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
                        Ok(b) => {
                            if !b.is_aggregation {
                                continue;
                            }

                            let bar = b.observed_at;

                            match ts {
                                None => ts = Some(bar),
                                Some(prev) if bar < prev => warn!("late quote: {:?}", b),
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
                                    let i = ((bar.ms() / 600_000) % WINDOW as u64) as usize;
                                    let wy = y
                                        .entry(b.pair.clone())
                                        .or_insert_with(|| Box::new([f64::NAN; WINDOW]));
                                    wy[i] = b.basis - b.cost;

                                    aux.insert(b.pair.clone(), InferenceAux {
                                        observed_at: bar,
                                        pair: b.pair.clone(),
                                        basis: b.basis,
                                        cost: b.cost,
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
