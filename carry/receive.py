import asyncio
import anyio
import random
import time
from httpx_ws import WebSocketNetworkError, WebSocketDisconnect

import datetime as dt
import json
import base64
import hmac
import hashlib

import polars as pl

from collections.abc import Iterable
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, computed_field
from typing import Union, Annotated, Literal, Any

import httpx
from httpx import AsyncClient
from httpx_ws import aconnect_ws

import secrets
from more_itertools import batched

from pb import PushDataV3ApiWrapper

import dotenv  # type: ignore

dotenv.load_dotenv()

import logging as l
import sys

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
for n in ("httpx",):
    l.getLogger(n).setLevel(l.WARNING)


class ExchangeInfo(BaseModel):
    class Filter(BaseModel):
        filterType: str = ""
        bidMultiplierUp: float | None = None
        askMultiplierDown: float | None = None

    class Symbol(BaseModel):
        symbol: str
        status: str  # "1" = trading; MEXC uses string codes, not the Binance enum
        fullName: str = ""
        st: bool = False  # special-treatment / risk flag

        baseAsset: str
        quoteAsset: str
        baseAssetPrecision: int
        quotePrecision: int
        quoteAssetPrecision: int
        baseCommissionPrecision: int
        quoteCommissionPrecision: int

        orderTypes: list[str] = []
        permissions: list[str] = []
        filters: list["ExchangeInfo.Filter"] = []

        isSpotTradingAllowed: bool = False
        isMarginTradingAllowed: bool = False
        tradeSideType: int | None = None

        baseSizePrecision: float
        quoteAmountPrecision: float
        quoteAmountPrecisionMarket: float | None = None
        maxQuoteAmount: float | None = None
        maxQuoteAmountMarket: float | None = None

        makerCommission: float
        takerCommission: float

        contractAddress: str = ""
        conceptPlateIds: list[int] = []
        firstOpenTime: dt.datetime | None = None

    class RateLimit(BaseModel):
        rateLimitType: str | None = None
        interval: str | None = None
        intervalNum: int | None = None
        limit: int | None = None

    timezone: str
    serverTime: dt.datetime
    rateLimits: list["ExchangeInfo.RateLimit"] = []
    exchangeFilters: list[dict[str, Any]] = []
    symbols: list["ExchangeInfo.Symbol"] = []


class MexcAuth(httpx.Auth):
    def __init__(self, api_key: str, api_secret: str, recv_window: int = 5000):
        self.api_key = api_key or ""
        self.api_secret = (api_secret or "").encode("utf-8")
        self.recv_window = recv_window
        if not (self.api_key and self.api_secret):
            l.warning(
                "API credentials incomplete. Access is restricted to public endpoints."
            )

    def auth_flow(self, req: httpx.Request):
        if not (self.api_key and self.api_secret) or req.extensions.get("public"):
            yield req
            return

        timestamp = str(int(dt.datetime.now().timestamp() * 1000))

        # 1. Existing query string, exactly as httpx already encoded it
        raw_q = req.url.query.decode("ascii")

        # 2. Append auth params -- MEXC v3 signs the query string for every verb,
        #    POST included. The body is never part of the payload.
        parts = [p for p in (raw_q, f"timestamp={timestamp}") if p]
        if self.recv_window:
            parts.append(f"recvWindow={self.recv_window}")
        payload = "&".join(parts)

        # 3. Sign, then append signature last -- it must not be re-encoded
        signature = hmac.new(
            self.api_secret, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        req.url = req.url.copy_with(
            query=f"{payload}&signature={signature}".encode("ascii")
        )

        # 4. Auth header. No passphrase, no key-version.
        req.headers["X-MEXC-APIKEY"] = self.api_key
        yield req


class KuCoinAuth(httpx.Auth):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        broker_partner: str = "",
        key_version: str = "3",
    ):
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.api_passphrase = api_passphrase or ""
        self.key_version = key_version

        # Pre-encrypt passphrase if credentials are provided
        if self.api_passphrase and self.api_secret:
            self.passphrase_sign = self._sign(
                self.api_passphrase.encode("utf-8"),
                self.api_secret.encode("utf-8"),
            )
        else:
            self.passphrase_sign = ""

        if not all([self.api_key, self.api_secret, self.api_passphrase]):
            l.warning(
                "API credentials incomplete. Access is restricted to public endpoints."
            )

    def _sign(self, plain: bytes, key: bytes) -> str:
        """Helper to generate HMAC-SHA256 signature encoded in Base64."""
        hm = hmac.new(key, plain, hashlib.sha256)
        return base64.b64encode(hm.digest()).decode("utf-8")

    def auth_flow(self, req: httpx.Request):
        """HTTPX Auth Flow Interceptor"""
        # Ensure standard JSON header for KuCoin API calls
        req.headers["Content-Type"] = "application/json"

        now = dt.datetime.now()
        timestamp = str(int(now.timestamp() * 1000))

        # 1. Extract path + query params (e.g. /api/v1/trade-fees?symbols=BTC-USDT)
        raw_url = req.url.raw_path.decode("ascii")

        # 2. Extract request body payload
        body_str = req.content.decode("utf-8") if req.content else ""

        # 3. Build string to sign: timestamp + METHOD + raw_url + body
        str_to_sign = f"{timestamp}{req.method.upper()}{raw_url}{body_str}"

        # 4. Generate API signature
        signature = self._sign(
            str_to_sign.encode("utf-8"), self.api_secret.encode("utf-8")
        )

        # 5. Set KuCoin authentication headers
        req.headers["KC-API-KEY"] = self.api_key
        req.headers["KC-API-PASSPHRASE"] = self.passphrase_sign
        req.headers["KC-API-TIMESTAMP"] = timestamp
        req.headers["KC-API-SIGN"] = signature
        req.headers["KC-API-KEY-VERSION"] = self.key_version
        req.headers["X-SITE-TYPE"] = "global"

        yield req


class PublicToken(BaseModel):
    class Instance(BaseModel):
        endpoint: str
        encrypt: bool
        protocol: str
        pingInterval: int
        pingTimeout: int

    class Data(BaseModel):
        token: str
        instanceServers: list["PublicToken.Instance"]

    code: str
    data: Data


class Contracts(BaseModel):
    class Data(BaseModel):
        symbol: str
        displaySymbol: str
        rootSymbol: str
        type: Literal["FFWCSX", "FFICSX"]
        firstOpenDate: int
        expireDate: int | None
        settleDate: int | None
        baseCurrency: str
        displayBaseCurrency: str
        quoteCurrency: str
        settleCurrency: str
        maxOrderQty: int
        marketMaxOrderQty: int
        maxPrice: float
        lotSize: int
        tickSize: float
        indexPriceTickSize: float
        multiplier: float
        initialMargin: float
        maintainMargin: float
        maxRiskLimit: int
        minRiskLimit: int
        riskStep: int
        makerFeeRate: float
        takerFeeRate: float
        takerFixFee: float
        makerFixFee: float
        settlementFee: float | None = None
        isDeleverage: bool
        isQuanto: bool
        isInverse: bool
        markMethod: Literal["FairPrice"] | None
        fairMethod: Literal["FundingRate"] | None
        fundingBaseSymbol: str | None
        fundingQuoteSymbol: str | None
        fundingRateSymbol: str | None
        indexSymbol: str
        settlementSymbol: str | None
        status: Literal[
            "Init", "Open", "BeingSettled", "Settled", "Paused", "Closed", "CancelOnly"
        ]
        fundingFeeRate: float | None
        predictedFundingFeeRate: float | None = None
        dailyInterestRate: float | None
        fundingRateGranularity: int | None
        effectiveFundingRateCycleStartTime: int | None
        currentFundingRateGranularity: int | None
        fundingRateCap: float | None
        fundingRateFloor: float | None
        period: Literal[0, 1] | None
        openInterest: str
        turnoverOf24h: float
        volumeOf24h: float
        markPrice: float
        indexPrice: float
        lastTradePrice: float | None
        nextFundingRateTime: int | None
        nextFundingRateDateTime: int | None
        maxLeverage: int
        sourceExchanges: list[str]
        premiumsSymbol1M: str
        premiumsSymbol8H: str
        fundingBaseSymbol1M: str | None
        fundingQuoteSymbol1M: str | None
        lowPrice: float
        highPrice: float
        priceChgPct: float
        priceChg: float
        k: float
        m: float
        f: float
        mmrLimit: float
        mmrLevConstant: float
        supportCross: bool
        buyLimit: float
        sellLimit: float
        adjustK: float | None = None
        adjustM: float | None = None
        adjustMmrLevConstant: float | None = None
        adjustActiveTime: int | None = None
        crossRiskLimit: float
        marketStage: str
        preMarketToPerpDate: int | None = None
        orderPriceRange: float
        marketType: Literal["CRYPTO", "NASDAQ", "OTHER"] = "CRYPTO"

    code: str
    data: list["Contracts.Data"]


class FundingRate(BaseModel):
    class Data(BaseModel):
        fundingRate: float
        timestamp: int
        granularity: int

    topic: str
    type: Literal["message"]
    subject: Literal["funding.rate"]
    data: Data


class Ticker(BaseModel):
    class Data(BaseModel):
        symbol: str
        sequence: int
        bestBidSize: int
        bestBidPrice: str
        bestAskSize: int
        bestAskPrice: str
        ts: int

    topic: str
    type: Literal["message"]
    subject: Literal["tickerV2"]
    data: Data


class Trade(BaseModel):
    class Data(BaseModel):
        symbol: str
        sequence: int
        side: str
        size: int
        price: str
        takerOrderId: str
        makerOrderId: str
        tradeId: str
        ts: int

    topic: str
    type: Literal["message"]
    subject: Literal["match"]
    data: Data


class Acknowledge(BaseModel):
    type: Literal["ack"]
    id: str


class Pong(BaseModel):
    type: Literal["pong"]
    id: str


class Error(BaseModel):
    type: Literal["error"]
    id: str
    code: int
    data: str


class Welcome(BaseModel):
    type: Literal["welcome"]
    id: str


Message = Annotated[
    Union[
        Welcome,
        Acknowledge,
        Pong,
        Error,
        Annotated[Union[FundingRate, Ticker, Trade], Field(discriminator="subject")],
    ],
    Field(discriminator="type"),
]


async def kc_public_token(client: AsyncClient) -> PublicToken:
    resp = await client.post("https://api-futures.kucoin.com/api/v1/bullet-public")
    resp.raise_for_status()

    model = PublicToken(**resp.json())

    if model.code != "200000":
        raise ValueError(f"public token: {model}")
    if len(model.data.instanceServers) == 0:
        raise ValueError(f"public token: {model}")

    return model


async def kc_fut_contracts(client: AsyncClient) -> Contracts:
    resp = await client.get("https://api-futures.kucoin.com/api/v1/contracts/active")
    resp.raise_for_status()
    model = Contracts(**resp.json())
    if model.code != "200000":
        raise ValueError(f"contracts: {model}")
    return model


async def mc_book(client: AsyncClient) -> pl.DataFrame:
    resp = await client.get("https://api.mexc.com/api/v3/ticker/bookTicker")
    resp.raise_for_status()

    return pl.DataFrame(
        resp.json(),
        schema={
            "symbol": pl.Utf8,
            "bidPrice": pl.Float64,
            "bidQty": pl.Float64,
            "askPrice": pl.Float64,
            "askQty": pl.Float64,
        },
    )


async def mc_exchange_info(client: AsyncClient) -> ExchangeInfo:
    resp = await client.get("https://api.mexc.com/api/v3/exchangeInfo")
    resp.raise_for_status()

    return ExchangeInfo(**resp.json())


async def kc_all_futures(client: AsyncClient) -> pl.DataFrame:
    resp = await client.get("https://api-futures.kucoin.com/api/v1/allTickers")
    resp.raise_for_status()

    return pl.DataFrame(
        resp.json()["data"],
        schema={
            "symbol": pl.Utf8,
            "bestBidPrice": pl.Float64,
            "bestBidSize": pl.Float64,
            "bestAskPrice": pl.Float64,
            "bestAskSize": pl.Float64,
        },
    )


class KuCoinSocket:
    def __init__(self, client: AsyncClient | None = None, topics: Iterable[str] = ()):
        self.client = client or AsyncClient()
        self.topics = set(topics)

    def _make_id(self) -> str:
        return secrets.token_urlsafe(16)

    async def get_fees(self, symbols: Iterable[str]) -> pl.DataFrame:
        # No authenticated fee endpoint wired up here; return an empty frame.
        return pl.DataFrame(
            schema={"symbol": pl.Utf8, "maker": pl.Float64, "taker": pl.Float64}
        )

    def subscribe(self, topic: str):
        self.topics.add(topic)

    async def listen(self):
        while True:
            tok = await kc_public_token(self.client)
            ep = f"{tok.data.instanceServers[0].endpoint}?token={tok.data.token}"

            async with aconnect_ws(ep, self.client) as ws:
                async for msg in self._listen_inner(ws):
                    yield msg

    async def _listen_inner(self, ws):
        in_flight = {
            t["id"]: t
            for t in [
                {"topic": t, "id": self._make_id(), "ts": None} for t in self.topics
            ]
        }

        while True:
            now = dt.datetime.now()
            for req in in_flight.values():
                if req["ts"] is not None and now - req["ts"] < dt.timedelta(seconds=30):
                    continue
                msg = {
                    "id": req["id"],
                    "type": "subscribe",
                    "topic": req["topic"],
                    "response": True,
                }
                l.debug(f"send {msg}")
                await ws.send_json(msg)
                req["ts"] = now

            raw = await ws.receive_text()
            l.debug(f"recv raw: {raw}")
            try:
                msg = TypeAdapter(Message).validate_json(raw)
                match msg:
                    case Acknowledge(id=id) as ack:
                        if id in in_flight:
                            del in_flight[id]
                        l.debug(f"ack: {ack}")
                        continue

                    case Error(id=id) as err:
                        l.warning(f"error: {err}")
                        continue

                    case Welcome():
                        pass

                    case _:
                        yield msg

            except ValidationError as e:
                l.error(f"recv: {e}")
                continue
            l.debug(f"recv: {msg}")
            yield msg


async def handler(ws):
    from websockets.exceptions import ConnectionClosed

    i = 0

    try:
        while True:
            await ws.send(json.dumps({"hello": f"world: {i}"}))
            await asyncio.sleep(1)
            i += 1
    except ConnectionClosed:
        pass


class MEXCSocket:
    def __init__(self, client: AsyncClient | None = None, topics: Iterable[str] = ()):
        self.client = client or AsyncClient()
        self.topics = list(topics)

    async def _subscribe(self, ws, topics):
        msg = {
            "method": "SUBSCRIPTION",
            "params": topics,
        }
        l.info(f"subscribe {msg}")
        await ws.send_json(msg)

    async def listen(self):
        async with aconnect_ws("wss://wbs-api.mexc.com/ws", self.client) as ws:
            await self._subscribe(ws, self.topics)

            while True:
                match await ws.receive():
                    case str(msg):
                        msg = json.loads(msg)
                        if ack["id"] != 0 or ack["code"] != 0:
                            raise ValueError(f"failed to subscribe: {ack}")

                    case bytes(msg):
                        yield PushDataV3ApiWrapper.FromString(msg)


class PairState(BaseModel):
    funding: float | None = None
    spot_ask_price: float | None = None
    spot_ask_qty: float | None = None
    spot_bid_price: float | None = None
    spot_bid_qty: float | None = None
    future_ask_price: float | None = None
    future_ask_qty: float | None = None
    future_bid_price: float | None = None
    future_bid_qty: float | None = None
    last_future: float | None = None
    last_spot: float | None = None
    mark: float | None = None
    index: float | None = None

    @computed_field
    @property
    def spot_mid(self) -> float | None:
        if self.spot_ask_price is not None and self.spot_bid_price is not None:
            return (self.spot_ask_price + self.spot_bid_price) / 2
        return None
        if self.spot_ask_price is None or self.spot_ask_price == 0:
            return self.spot_bid_price
        if self.spot_bid_price is None or self.spot_bid_price == 0:
            return self.spot_ask_price
        return (self.spot_ask_price + self.spot_bid_price) / 2

    @computed_field
    @property
    def spot_spread_bps(self) -> float | None:
        if self.spot_ask_price is None or self.spot_bid_price is None:
            return None
        if self.spot_ask_price == 0 or self.spot_bid_price == 0:
            return None

        return (self.spot_ask_price - self.spot_bid_price) / self.spot_bid_price * 1e4

    @computed_field
    @property
    def future_mid(self) -> float | None:
        if self.future_ask_price is None or self.future_ask_price == 0:
            return self.future_bid_price
        if self.future_bid_price is None or self.future_bid_price == 0:
            return self.future_ask_price
        return (self.future_ask_price + self.future_bid_price) / 2

    @computed_field
    @property
    def future_spread_bps(self) -> float | None:
        if self.future_ask_price is None or self.future_bid_price is None:
            return None
        if self.future_ask_price == 0 or self.future_bid_price == 0:
            return None

        return (
            (self.future_ask_price - self.future_bid_price)
            / self.future_bid_price
            * 1e4
        )

    @computed_field
    @property
    def basis_bps(self) -> float | None:
        future_mid = self.future_mid
        spot_mid = self.spot_mid
        if future_mid is not None and spot_mid is not None:
            return (spot_mid - future_mid) / future_mid * 1e4
        return None


class Log:
    def __init__(self, stem: str, schema: dict[str, pl.DataType], chunk_size=1_000_000):
        self.chunk_size = chunk_size
        self.schema = schema
        self.queue = asyncio.Queue(maxsize=1024)
        self.buffer = []
        self.stem = stem
        self.counter = 0

    async def log(self, row):
        await self.queue.put(row)

    async def run(self):
        try:
            while True:
                self.buffer.append(await self.queue.get())
                if len(self.buffer) >= self.chunk_size:
                    self.flush()

        except asyncio.QueueShutDown:
            return

    def flush(self):
        fn = f"{self.stem}{self.counter:04d}.parquet"
        l.info(f"write {fn}")
        pl.DataFrame(self.buffer, orient="row", schema=self.schema).write_parquet(fn)

        self.buffer = []
        self.counter += 1


async def kc_ping(ws, interval: int):
    try:
        while True:
            await ws.send_json(
                {
                    "id": secrets.token_urlsafe(8),
                    "type": "ping",
                }
            )
            await asyncio.sleep(interval)

    except Exception as e:
        l.debug(f"kc_ping: socket closed, ending ping loop: {e!r}")
        return


async def mc_ping(ws, interval: int):
    try:
        while True:
            await ws.send_json(
                {
                    "method": "PING",
                }
            )
            await asyncio.sleep(interval)

    except Exception as e:
        l.debug(f"mc_ping: socket closed, ending ping loop: {e!r}")
        return


async def kc_websocket(client: AsyncClient, topics: list[str], out: asyncio.Queue):
    backoff = 1.0
    while True:
        t0 = time.monotonic()
        try:
            tok = await kc_public_token(client)
            srv = tok.data.instanceServers[0]
            ep = f"{srv.endpoint}?token={tok.data.token}"

            async with aconnect_ws(
                ep, client, keepalive_ping_timeout_seconds=None
            ) as ws:

                for t in topics:
                    await ws.send_json(
                        {
                            "id": secrets.token_urlsafe(8),
                            "type": "subscribe",
                            "topic": t,
                            "response": True,
                        }
                    )
                    await asyncio.sleep(0.15)

                ping = asyncio.create_task(kc_ping(ws, srv.pingInterval / 1000 / 2))

                try:
                    while True:
                        raw = await ws.receive_text()
                        try:
                            msg = TypeAdapter(Message).validate_json(raw)
                        except ValidationError as e:
                            l.error(f"kc parse: {e}")
                            continue

                        match msg:
                            case Welcome() | Acknowledge():
                                continue
                            case Error() as err:
                                l.error(f"kc: {err}")
                                continue
                            case _:
                                await out.put(msg)
                finally:
                    ping.cancel()
                    await asyncio.gather(ping, return_exceptions=True)

        except* (WebSocketNetworkError, WebSocketDisconnect) as e:
            if time.monotonic() - t0 > 30:
                backoff = 1.0  # healthy run, restart the backoff ladder
            delay = backoff * random.uniform(0.5, 1.5)
            backoff = min(backoff * 2, 30)
            l.warning(f"kc_websocket: connection lost, reconnect in {delay:.1f}s")
            await asyncio.sleep(delay)


async def mc_websocket(client: AsyncClient, topics: list[str], out: asyncio.Queue):
    from wsproto.events import TextMessage, BytesMessage

    backoff = 1.0
    while True:
        t0 = time.monotonic()
        try:
            async with aconnect_ws("wss://wbs-api.mexc.com/ws", client) as ws:
                await ws.send_json({"method": "SUBSCRIPTION", "params": topics})

                ping = asyncio.create_task(mc_ping(ws, 15))

                try:
                    while True:
                        match await ws.receive():
                            case TextMessage(data=raw):
                                ack = json.loads(raw)
                                if ack.get("code", 0) != 0:
                                    raise ValueError(f"mexc: {ack}")
                            case BytesMessage(data=raw):
                                await out.put(PushDataV3ApiWrapper.FromString(raw))
                finally:
                    ping.cancel()
                    await asyncio.gather(ping, return_exceptions=True)

        except* (WebSocketNetworkError, WebSocketDisconnect) as e:
            if time.monotonic() - t0 > 30:
                backoff = 1.0  # healthy run, restart the backoff
            delay = backoff * random.uniform(0.5, 1.5)
            backoff = min(backoff * 2, 30)
            l.warning(f"mc_websocket: connection lost, reconnect in {delay:.1f}s")
            await asyncio.sleep(delay)


BookSchema = {
    "ts": pl.Datetime,
    "symbol": pl.Utf8,
    "bid_price": pl.Float64,
    "bid_qty": pl.Int64,
    "ask_price": pl.Float64,
    "ask_qty": pl.Int64,
}
TradesSchema = {
    "ts": pl.Datetime,
    "symbol": pl.Utf8,
    "price": pl.Float64,
    "qty": pl.Int64,
    "side": pl.Utf8,
}

FundingSchema = {
    "ts": pl.Datetime,
    "symbol": pl.Utf8,
    "funding": pl.Float64,
}


async def exchange_data(
    states: dict[str, PairState],
    spot_book: Log,
    spot_trades: Log,
    future_book: Log,
    future_trades: Log,
    future_funding: Log,
):
    from os import getenv
    from aiostream import stream
    import itertools as it

    kc_auth = KuCoinAuth(
        api_key=getenv("KUCOIN_API_KEY"),
        api_secret=getenv("KUCOIN_API_SECRET"),
        api_passphrase=getenv("KUCOIN_API_PASSWORD"),
    )

    mc_auth = MexcAuth(
        api_key=getenv("MEXC_ACCESS_KEY"),
        api_secret=getenv("MEXC_SECRET_KEY"),
    )

    async with AsyncClient(auth=mc_auth) as mc_client:
        async with AsyncClient(auth=kc_auth) as kc_client:
            kc_pairs, mc_pairs = await asyncio.gather(
                kc_fut_contracts(kc_client), mc_exchange_info(mc_client)
            )

            kc_symbols = {
                c.symbol: c.baseCurrency.lower()
                for c in kc_pairs.data
                if c.status == "Open" and c.quoteCurrency.lower() == "usdt"
            }
            mc_symbols = {
                s.symbol: s.baseAsset.lower()
                for s in mc_pairs.symbols
                if s.status == "1"
                and s.isSpotTradingAllowed
                and s.quoteAsset.lower() == "usdt"
            }

            for sym in list(set(kc_symbols.values()) & set(mc_symbols.values())):
                states[sym] = PairState()

            mc_topic_templates = [
                "spot@public.aggre.deals.v3.api.pb@100ms@",
                "spot@public.aggre.bookTicker.v3.api.pb@100ms@",
            ]
            mc_topics = [
                f"{p[0]}{p[1]}"
                for p in it.product(mc_topic_templates, mc_symbols.keys())
                if mc_symbols[p[1]] in states
            ]

            kc_topic_templates = [
                "/contractMarket/tickerV2:",
                "/contractMarket/execution:",
            ]
            kc_topics = [
                f"{p[0]}{p[1]}"
                for p in it.product(kc_topic_templates, kc_symbols.keys())
                if kc_symbols[p[1]] in states
            ]

            queue = asyncio.Queue(maxsize=100_000)

            async with anyio.create_task_group() as tg:
                for b in batched(kc_topics, 25):
                    tg.start_soon(kc_websocket, kc_client, list(b), queue)
                for b in batched(mc_topics, 30):
                    tg.start_soon(mc_websocket, mc_client, list(b), queue)

                while True:
                    msg = await queue.get()
                    now = dt.datetime.now()
                    match msg:
                        case FundingRate() as fr:
                            symbol = kc_symbols[msg.topic.split(":")[-1]]
                            states[symbol].funding = fr.data.fundingRate * 1e4
                            await future_funding.log(
                                [ft.data.timestamp, symbol, fr.data.fundingRate * 1e4]
                            )

                        case Trade() as t:
                            symbol = kc_symbols[msg.topic.split(":")[-1]]
                            states[symbol].last_future = float(t.data.price)
                            await future_trades.log(
                                [
                                    dt.datetime.fromtimestamp(
                                        t.data.ts / 1_000_000_000
                                    ),
                                    symbol,
                                    t.data.price,
                                    t.data.size,
                                    t.data.side,
                                ]
                            )

                        case Ticker() as t:
                            symbol = kc_symbols[msg.topic.split(":")[-1]]
                            states[symbol].future_ask_price = float(t.data.bestAskPrice)
                            states[symbol].future_ask_qty = float(t.data.bestAskSize)
                            states[symbol].future_bid_price = float(t.data.bestBidPrice)
                            states[symbol].future_bid_qty = float(t.data.bestBidSize)
                            await future_book.log(
                                [
                                    dt.datetime.fromtimestamp(
                                        t.data.ts / 1_000_000_000
                                    ),
                                    symbol,
                                    t.data.bestAskPrice,
                                    t.data.bestAskSize,
                                    t.data.bestBidPrice,
                                    t.data.bestBidSize,
                                ]
                            )

                        case PushDataV3ApiWrapper() as pd:
                            symbol = mc_symbols[pd.symbol]

                            match pd.WhichOneof("body"):
                                case "publicAggreDeals":
                                    states[symbol].last_spot = float(
                                        pd.publicAggreDeals.deals[-1].price
                                    )
                                    for d in pd.publicAggreDeals.deals:
                                        await spot_trades.log(
                                            [
                                                dt.datetime.fromtimestamp(
                                                    d.time / 1_000
                                                ),
                                                symbol,
                                                d.price,
                                                d.quantity,
                                                "buy" if d.tradeType == 1 else "sell",
                                            ]
                                        )

                                case "publicAggreBookTicker":
                                    d = pd.publicAggreBookTicker
                                    states[symbol].spot_ask_price = float(d.askPrice)
                                    states[symbol].spot_ask_qty = float(d.askQuantity)
                                    states[symbol].spot_bid_price = float(d.bidPrice)
                                    states[symbol].spot_bid_qty = float(d.bidQuantity)
                                    await spot_book.log(
                                        [
                                            dt.datetime.fromtimestamp(
                                                d.lastOrderCreateTime / 1_000
                                            ),
                                            symbol,
                                            d.askPrice,
                                            d.askQuantity,
                                            d.bidPrice,
                                            d.bidQuantity,
                                        ]
                                    )
