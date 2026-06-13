import json

import requests


def get_kucoin_margin_pairs():
    # KuCoin API endpoints for margin symbols
    cross_margin_url = "https://api.kucoin.com/api/v3/margin/symbols"
    isolated_margin_url = "https://api.kucoin.com/api/v1/isolated/symbols"

    cross_pairs = []
    isolated_pairs = []

    # 1. Fetch Cross Margin Pairs
    try:
        response = requests.get(cross_margin_url)
        response.raise_for_status()
        data = response.json()

        if data.get("code") == "200000":
            payload = data.get("data", [])
            # KuCoin sometimes wraps the list in an 'items' dictionary key
            items = payload.get("items", []) if isinstance(payload, dict) else payload
            cross_pairs = [item["symbol"] for item in items]
    except Exception as e:
        print(f"Error fetching cross margin pairs: {e}")

    # 2. Fetch Isolated Margin Pairs
    try:
        response = requests.get(isolated_margin_url)
        response.raise_for_status()
        data = response.json()

        if data.get("code") == "200000":
            payload = data.get("data", [])
            items = payload.get("items", []) if isinstance(payload, dict) else payload
            isolated_pairs = [item["symbol"] for item in items]
    except Exception as e:
        print(f"Error fetching isolated margin pairs: {e}")

    # 3. Combine and deduplicate
    all_margin_pairs = list(set(cross_pairs + isolated_pairs))

    return {
        "cross_margin": cross_pairs,
        "isolated_margin": isolated_pairs,
        "all_margin_pairs": all_margin_pairs,
    }


if __name__ == "__main__":
    pairs = get_kucoin_margin_pairs()

    print(f"Total Cross Margin Pairs: {len(pairs['cross_margin'])}")
    print(f"Total Isolated Margin Pairs: {len(pairs['isolated_margin'])}")
    print(f"Total Unique Margin Pairs: {len(pairs['all_margin_pairs'])}\n")

    print("Sample Margin Pairs:")
    print(pairs["all_margin_pairs"][:10])

    with open("kucoin-margin.json", "w") as f:
        json.dump(pairs["all_margin_pairs"], f)
    print("\nWritten all unique margin pairs to kucoin-margin.json")
