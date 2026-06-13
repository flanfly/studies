"""Download margin borrow rates for USDT-quoted pairs from HTX and KuCoin.

Fetches both isolated and cross margin rates.

HTX uses the private /v1/margin/loan-info (isolated) and /v1/cross-margin/loan-info
(cross) endpoints. KuCoin uses /api/v3/margin/borrowRate for both, filtering by
available symbols for each margin type.

Results are written to borrow-rates.parquet with columns: symbol, exchange, type, rate.

Requires (.env):
  KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSWORD  (for KuCoin)
  HTX_ACCESS_KEY, HTX_SECRET_KEY                            (for HTX)
"""

import base64
import hashlib
import hmac
import os
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import polars as pl
import requests
from dotenv import load_dotenv

load_dotenv()

OUTPUT = "borrow-rates.parquet"


# ============================================================================
# HTX
# ============================================================================

def _htx_sign(method: str, host: str, path: str, params: dict, secret: str) -> str:
    sorted_keys = sorted(params.keys())
    encoded = "&".join(
        f"{k}={requests.utils.quote(str(params[k]), safe='')}" for k in sorted_keys
    )
    payload = f"{method}\n{host}\n{path}\n{encoded}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.b64encode(sig).decode()


def _htx_private_get(path: str, params_extra: dict | None = None) -> dict | None:
    """Make an authenticated GET request to the HTX API."""
    key = os.environ.get("HTX_ACCESS_KEY")
    secret = os.environ.get("HTX_SECRET_KEY")
    if not key or not secret:
        return None

    host = "api.huobi.pro"
    method = "GET"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    params = {
        "AccessKeyId": key,
        "SignatureMethod": "HmacSHA256",
        "SignatureVersion": "2",
        "Timestamp": timestamp,
    }
    if params_extra:
        params.update(params_extra)
    params["Signature"] = _htx_sign(method, host, path, params, secret)

    url = f"https://{host}{path}?{urlencode(params)}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _htx_fetch_isolated_rates() -> dict[str, float]:
    """{base_lower: annual_rate} from HTX isolated margin."""
    data = _htx_private_get("/v1/margin/loan-info")
    if data is None:
        print("HTX: skipping – HTX_ACCESS_KEY / HTX_SECRET_KEY not set")
        return {}
    if data.get("status") != "ok":
        print(f"HTX isolated: API error – {data}")
        return {}

    rates: dict[str, float] = {}
    for item in data.get("data", []):
        symbol: str = item.get("symbol", "")
        if not symbol.upper().endswith("USDT"):
            continue
        base = symbol[:-4].lower()
        for cur in item.get("currencies", []):
            if cur.get("currency", "").upper() == base.upper():
                rates[base] = float(cur["interest-rate"]) * 365
    return rates


def _htx_fetch_cross_rates() -> dict[str, float]:
    """{base_lower: annual_rate} from HTX cross margin.

    Cross-margin loan-info returns per-currency rates; we cross-reference with
    available cross-margin USDT symbols to know which are USDT-quoted.
    """
    loan_data = _htx_private_get("/v1/cross-margin/loan-info")
    if loan_data is None:
        return {}
    if loan_data.get("status") != "ok":
        msg = loan_data.get("err-msg", loan_data)
        print(f"HTX cross: not available – {msg}")
        return {}

    # Build a lookup of currency -> daily rate
    daily_rates: dict[str, float] = {}
    for item in loan_data.get("data", []):
        cur = item.get("currency", "").upper()
        daily_rates[cur] = float(item["interest-rate"])

    # Get cross margin symbols to filter USDT pairs
    symbols_url = "https://api.huobi.pro/v1/margin/symbols"
    try:
        resp = requests.get(symbols_url, timeout=30)
        resp.raise_for_status()
        sym_data = resp.json()
    except Exception as e:
        print(f"HTX cross: symbols request failed – {e}")
        return {}

    rates: dict[str, float] = {}
    for sym in sym_data.get("data", []):
        symbol = sym.get("symbol", "")
        if not symbol.upper().endswith("USDT"):
            continue
        base = symbol[:-4].upper()
        if base in daily_rates:
            rates[base.lower()] = daily_rates[base] * 365
    return rates


# ============================================================================
# KuCoin
# ============================================================================

def _kucoin_sign(timestamp: str, method: str, endpoint: str, body: str, secret: str) -> str:
    payload = timestamp + method + endpoint + body
    return base64.b64encode(
        hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    ).decode()


def _kucoin_passphrase_sign(passphrase: str, secret: str) -> str:
    return base64.b64encode(
        hmac.new(secret.encode(), passphrase.encode(), hashlib.sha256).digest()
    ).decode()


def _kucoin_headers(method: str, endpoint: str, body: str = "") -> dict:
    key = os.environ.get("KUCOIN_API_KEY")
    secret = os.environ.get("KUCOIN_API_SECRET")
    passphrase = os.environ.get("KUCOIN_API_PASSWORD")
    if not key or not secret or not passphrase:
        raise RuntimeError("KUCOIN_API_KEY / _SECRET / _PASSWORD not set in .env")

    now = str(int(time.time() * 1000))
    return {
        "KC-API-KEY": key,
        "KC-API-SIGN": _kucoin_sign(now, method, endpoint, body, secret),
        "KC-API-TIMESTAMP": now,
        "KC-API-PASSPHRASE": _kucoin_passphrase_sign(passphrase, secret),
        "KC-API-KEY-VERSION": "2",
    }


def _kucoin_currencies_from_symbols(endpoint: str) -> list[str]:
    """Fetch USDT base currencies from a KuCoin margin symbols endpoint (public)."""
    try:
        resp = requests.get(f"https://api.kucoin.com{endpoint}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"KuCoin symbols ({endpoint}): request failed – {e}")
        return []

    if data.get("code") != "200000":
        print(f"KuCoin symbols ({endpoint}): API error – {data}")
        return []

    # some endpoints wrap in {"items": [...]}, others return a flat list
    items = data.get("data", [])
    if isinstance(items, dict):
        items = items.get("items", [])

    currencies: list[str] = []
    for s in items:
        sym = s.get("symbol", "")
        if sym.upper().endswith("-USDT"):
            currencies.append(sym.split("-")[0])
    return currencies


def _kucoin_fetch_rates_for(currencies: list[str]) -> dict[str, float]:
    """Fetch annual borrow rates for a list of currencies (batched)."""
    rates: dict[str, float] = {}
    batch_size = 50
    for i in range(0, len(currencies), batch_size):
        batch = currencies[i : i + batch_size]
        endpoint = f"/api/v3/margin/borrowRate?currency={','.join(batch)}"
        headers = _kucoin_headers("GET", endpoint)
        try:
            resp = requests.get(f"https://api.kucoin.com{endpoint}", headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"KuCoin borrowRate batch: failed – {e}")
            continue

        if data.get("code") != "200000":
            print(f"KuCoin borrowRate: API error – {data}")
            continue

        for item in data.get("data", {}).get("items", []):
            rates[item["currency"].lower()] = float(item["annualizedBorrowRate"])
    return rates


def _kucoin_fetch_isolated_rates() -> dict[str, float]:
    currencies = _kucoin_currencies_from_symbols("/api/v1/isolated/symbols")
    if not currencies:
        return {}
    print(f"KuCoin isolated: {len(currencies)} USDT pairs")
    return _kucoin_fetch_rates_for(currencies)


def _kucoin_fetch_cross_rates() -> dict[str, float]:
    currencies = _kucoin_currencies_from_symbols("/api/v3/margin/symbols")
    if not currencies:
        return {}
    print(f"KuCoin cross: {len(currencies)} USDT pairs")
    return _kucoin_fetch_rates_for(currencies)


# ============================================================================
# Main
# ============================================================================

def main():
    rows: list[dict] = []

    for base, rate in _htx_fetch_isolated_rates().items():
        rows.append({"symbol": base, "exchange": "htx", "type": "isolated", "rate": rate})
    print(f"  HTX isolated: {len(rows)} rates")

    n = len(rows)
    for base, rate in _htx_fetch_cross_rates().items():
        rows.append({"symbol": base, "exchange": "htx", "type": "cross", "rate": rate})
    print(f"  HTX cross: {len(rows) - n} rates")

    n = len(rows)
    for base, rate in _kucoin_fetch_isolated_rates().items():
        rows.append({"symbol": base, "exchange": "kucoin", "type": "isolated", "rate": rate})
    print(f"  KuCoin isolated: {len(rows) - n} rates")

    n = len(rows)
    for base, rate in _kucoin_fetch_cross_rates().items():
        rows.append({"symbol": base, "exchange": "kucoin", "type": "cross", "rate": rate})
    print(f"  KuCoin cross: {len(rows) - n} rates")

    df = pl.DataFrame(rows).sort(["symbol", "exchange", "type"])
    df.write_parquet(OUTPUT)
    print(f"\nWrote {len(df)} rows to {OUTPUT}")
    print(df.head(10))


if __name__ == "__main__":
    main()
