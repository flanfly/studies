use std::collections::{HashMap, VecDeque};
use std::time::{Duration as StdDuration, Instant};

use anyhow::Result;
use async_stream::stream;
use futures::stream::Stream;
use futures::{SinkExt, StreamExt};
use governor::{Quota, RateLimiter};
use log::{error, warn};
use nonzero_ext::nonzero;
use prost::Message as _;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::select;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;

use crate::pb::universe::market::{Asset, asset};
use crate::pb::{Event, Quote as PbQuote, event::Payload};
use crate::{Instrument, Kind, Pair, Timestamp, mkid};

mod mexcpb {
    include!(concat!(env!("OUT_DIR"), "/mexc_proto.rs"));
}
use mexcpb::PushDataV3ApiWrapper;

#[derive(Debug, Clone)]
pub(crate) struct Quote {
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
    fn from_kucoin(p: Pair, i: Instrument, ev: KuCoinBboEvent) -> Quote {
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

    /// Wrap the raw tick into the unified, serializable event envelope.
    pub(crate) fn into_event(self) -> Event {
        Event {
            id: mkid(),
            timestamp: self.received_at.ms(),
            payload: Some(Payload::Quote(PbQuote {
                is_aggregation: self.is_aggregation,
                observed_at: self.observed_at.ms(),
                pair: self.pair.to_string(),
                instrument: self.instrument.to_string(),
                bid: self.bid,
                bid_qty: self.bid_qty,
                ask: self.ask,
                ask_qty: self.ask_qty,
            })),
        }
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

pub(crate) fn subscribe_kucoin_bbo(
    tickers: Vec<(Pair, Instrument)>,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<Event>> {
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
                            yield Ok(Quote::from_kucoin(p.clone(), i.clone(), ev).into_event());
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

pub(crate) fn subscribe_mexc_bbo(
    tickers: Vec<(Pair, Instrument)>,
    rate_limit: u32,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<Event>> {
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
                                yield Ok(Quote::from_mexc(p.clone(), i.clone(), ev).into_event());
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
    trade_side_type: u32,
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

pub(crate) async fn kucoin_exchange_state() -> Result<Vec<Asset>> {
    let res = reqwest::get("https://api-futures.kucoin.com/api/v1/contracts/active")
        .await?
        .json::<KucoinResponse<Vec<KucoinFutureSymbol>>>()
        .await?;
    let a = res
        .data
        .into_iter()
        .filter_map(|s| {
            if s.quote_currency.to_lowercase() != "usdt" {
                return None;
            }

            let (i, p) = instrument_and_pair("kucoin", Kind::Future, &s.base_currency, &s.symbol);

            Asset {
                pair: p.to_string(),
                instrument: i.to_string(),
                state: match s.status.as_ref() {
                    "Init" => asset::State::Halted as i32,
                    "Open" => asset::State::Open as i32,
                    "BeingSettled" => asset::State::Delisted as i32,
                    "Settled" => asset::State::Delisted as i32,
                    "Paused" => asset::State::Halted as i32,
                    "Closed" => asset::State::Delisted as i32,
                    "CancelOnly" => asset::State::Halted as i32,
                    _ => asset::State::Unspecified as i32,
                },
            }
            .into()
        })
        .collect();

    Ok(a)
}

pub(crate) async fn mexc_exchange_state() -> Result<(Vec<Asset>, u32)> {
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

    let a = res
        .symbols
        .into_iter()
        .filter_map(|s| {
            if s.quote_asset.to_lowercase() != "usdt" {
                return None;
            }

            let (i, p) = instrument_and_pair("mexc", Kind::Spot, &s.base_asset, &s.symbol);

            Asset {
                pair: p.to_string(),
                instrument: i.to_string(),
                state: match (
                    s.trade_side_type,
                    s.status.as_ref(),
                    s.is_spot_trading_allowed,
                ) {
                    (_, _, false) => asset::State::Delisted as i32,
                    (1, "1", true) => asset::State::Open as i32,
                    (1, "2", true) => asset::State::Halted as i32,
                    (1, "3", true) => asset::State::Delisted as i32,
                    (2, _, _) => asset::State::Halted as i32,
                    (3, _, _) => asset::State::Halted as i32,
                    (4, _, _) => asset::State::Halted as i32,
                    _ => asset::State::Unspecified as i32,
                },
            }
            .into()
        })
        .collect();

    Ok((a, rate_limit))
}

pub(crate) fn instrument_and_pair(
    venue: impl AsRef<str>,
    kind: Kind,
    base: impl AsRef<str>,
    symbol: impl AsRef<str>,
) -> (Instrument, Pair) {
    let symbol = symbol.as_ref().to_lowercase();
    let venue = venue.as_ref().to_lowercase();
    let base = base.as_ref().to_lowercase();

    match (venue.as_ref(), kind, base.as_ref()) {
        ("kucoin", Kind::Future, "purr") => (
            Instrument::new(Kind::Future, "kucoin", symbol),
            Pair("purrhl".into()),
        ),
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
