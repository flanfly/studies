import json

import requests


def get_htx_margin_pairs():
    # HTX (formerly Huobi) public API endpoint for margin symbols
    # Note: The /v1/margin/loan-info and /v1/cross-margin/loan-info endpoints
    # now require API authentication (signature), no longer publicly accessible.
    # Using the public /v1/margin/symbols endpoint instead.
    margin_symbols_url = "https://api.huobi.pro/v1/margin/symbols"

    cross_pairs = []

    # 1. Fetch Cross Margin Pairs
    try:
        response = requests.get(margin_symbols_url)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "ok":
            cross_pairs = [item["symbol"] for item in data.get("data", [])]
    except Exception as e:
        print(f"Error fetching margin pairs: {e}")

    return {
        "cross_margin_pairs": cross_pairs,
    }


if __name__ == "__main__":
    htx_data = get_htx_margin_pairs()

    print(f"Total Cross Margin Pairs: {len(htx_data['cross_margin_pairs'])}")
    print(f"Sample Margin Pairs: {htx_data['cross_margin_pairs'][:10]}")

    with open("htx-margin.json", "w") as f:
        json.dump(htx_data["cross_margin_pairs"], f)
    print("\nWritten all margin pairs to htx-margin.json")
