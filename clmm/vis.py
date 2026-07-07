import requests
from dotenv import load_dotenv
from os import getenv
import json
import math
import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

load_dotenv()


def price_to_tick(
    price: float, d0: int, d1: int, price_is_token1_in_token0: bool = True
) -> int:
    if price_is_token1_in_token0:
        return int(np.floor(-np.log(price / 10 ** (d1 - d0)) / np.log(1.0001)))
    else:
        return int(np.floor(np.log(price / 10 ** (d0 - d1)) / np.log(1.0001)))


def tick_to_price(
    tick: int, d0: int, d1: int, price_is_token1_in_token0: bool = True
) -> float:
    if price_is_token1_in_token0:
        return (1.0001 ** (-tick)) * (10 ** (d1 - d0))
    else:
        return (1.0001**tick) * (10 ** (d0 - d1))


BLOCK_NUMBER = 25363520
SUBGRAPH_URL = "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
POOL_ADDRESS = (
    "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640".lower()
)  # USDC/ETH pool as example

query = """
query GetPoolTokens($poolId: ID!) {
  pool(id: $poolId) {
    id
    token0 {
      name
      symbol
      decimals
    }
    token1 {
      name
      symbol
      decimals
    }
  }
}
"""

# 3. Setup payload with lowercased address
payload = {"query": query, "variables": {"poolId": POOL_ADDRESS.lower()}}
headers = {"authorization": f"""Bearer {getenv('GRAPH_API_KEY')}"""}

# 4. Execute POST request
response = requests.post(SUBGRAPH_URL, json=payload, headers=headers)

if response.status_code == 200:
    data = response.json().get("data", {})
    pool_data = data.get("pool")

    if pool_data:
        t0 = pool_data["token0"]
        t1 = pool_data["token1"]

        print(f"Pool: {POOL_ADDRESS}")
        print("-" * 40)
        print(
            f"Asset 0 (Token0): {t0['name']} ({t0['symbol']}) | Decimals: {t0['decimals']}"
        )
        print(
            f"Asset 1 (Token1): {t1['name']} ({t1['symbol']}) | Decimals: {t1['decimals']}"
        )
    else:
        print("Pool address not found in this subgraph deployment.")
else:
    print(f"Query failed with status code {response.status_code}: {response.text}")

query = """
query GetLiquidityProfile($poolAddress: String!, $blockNumber: Int!, $lastTick: BigInt) {
  pool(id: $poolAddress, block: { number: $blockNumber }) {
    id
    tick
    liquidity
    sqrtPrice
    feeTier
  }
  ticks(
    first: 1000
    where: { 
      poolAddress: $poolAddress, 
      liquidityGross_gt: "0" 
      tickIdx_gt: $lastTick
    }
    block: { number: $blockNumber }
    orderBy: tickIdx
    orderDirection: asc
  ) {
    tickIdx
    liquidityGross
    liquidityNet
  }
}
"""

all_ticks = []
pool_data = None
last_tick = "-887273"
limit = 1000

print(f"Fetching data for pool {POOL_ADDRESS} at block {BLOCK_NUMBER}...")

while True:
    variables = {
        "poolAddress": POOL_ADDRESS,
        "blockNumber": BLOCK_NUMBER,
        "lastTick": last_tick,
    }

    # Send the POST request to the subgraph
    response = requests.post(
        SUBGRAPH_URL,
        json={"query": query, "variables": variables},
        headers=headers,
    )

    # Check for HTTP errors (e.g., 404, 500)
    response.raise_for_status()

    # Parse the JSON response
    data = response.json()

    # Check for GraphQL-specific errors
    if "errors" in data:
        raise Exception(f"GraphQL Error: {json.dumps(data['errors'], indent=2)}")

    # Extract the data
    response_data = data.get("data", {})

    # Grab the static pool data on the very first loop
    if not pool_data:
        pool_data = response_data.get("pool")
        if not pool_data:
            print(
                "Warning: Pool not found at this block. Check the address and block number."
            )
            break

    current_ticks = response_data.get("ticks", [])
    if not current_ticks:
        break

    all_ticks.extend(current_ticks)

    print(f"Fetched {len(current_ticks)} ticks (Total so far: {len(all_ticks)})")

    # Update last_tick for cursor pagination
    last_tick = str(current_ticks[-1]["tickIdx"])

    # Pagination logic: if we receive fewer than 1000 items, we are done.
    if len(current_ticks) < limit:
        break

print("Finished fetching all ticks!")

start_ticks = dict()
if pool_data:
    print("\n--- Pool Summary ---")
    print(f"Current Tick: {pool_data['tick']}")
    print(f"Current Active Liquidity: {pool_data['liquidity']}")
    print(f"Total Initialized Ticks: {len(all_ticks)}")

    # Print a sample of the first 3 ticks to verify
    print("\n--- First 3 Ticks Sample ---")
    print(json.dumps(all_ticks[:3], indent=2))

    for t in all_ticks:
        start_ticks[int(t["tickIdx"])] = int(t["liquidityNet"])

import polars as pl
from eth_utils import event_signature_to_log_topic

# Canonical event signatures
swap = event_signature_to_log_topic(
    "Swap(address,address,int256,int256,uint160,uint128,int24)"
)
mint = event_signature_to_log_topic(
    "Mint(address,address,int24,int24,uint128,uint256,uint256)"
)
burn = event_signature_to_log_topic("Burn(address,int24,int24,uint128,uint256,uint256)")

topic0_to_event = {
    swap: "swap",
    mint: "mint",
    burn: "burn",
}


# Fast helper for big-endian decoding of bytes to integers (Fix 2)
def be_int(b: bytes, signed: bool) -> int:
    if b is None:
        return 0
    return int.from_bytes(b, "big", signed=signed)


# 1. Filter raw logs strictly to the pool address and sort chronologically (Fix 1, 4)
pool_addr_bytes = bytes.fromhex(POOL_ADDRESS[2:])
df = (
    pl.read_parquet("ethereum__logs__*.parquet")
    .filter(pl.col("address") == pool_addr_bytes)
    .join(
        pl.read_parquet("ethereum__blocks__*.parquet").select(
            pl.col("block_number"),
            ts=pl.from_epoch(pl.col("timestamp"), time_unit="s").dt.replace_time_zone(
                "UTC"
            ),
        ),
        on=["block_number"],
    )
    .filter(pl.col("block_number") > BLOCK_NUMBER)
    .sort(["block_number", "transaction_index", "log_index"])
)

d0 = int(t0["decimals"])
d1 = int(t1["decimals"])

# Filter for the unified replay stream of Swap, Mint, Burn events (Fix 4)
events_df = df.filter(pl.col("topic0").is_in([swap, mint, burn])).with_columns(
    epoch=pl.col("ts").dt.epoch("s")
)

# Setup timeline binning (Fix 7)
N_bins = 1500 * 2  # Horizontal time resolution
M = 800 * 3  # Vertical price resolution

# Find timeline range from actual swap events
swaps_df = events_df.filter(pl.col("topic0") == swap)
if swaps_df.height > 0:
    epochs = swaps_df["epoch"].to_numpy()
    min_epoch = epochs.min()
    max_epoch = epochs.max()
else:
    # Fallback if no swaps found
    min_epoch = events_df["epoch"].min()
    max_epoch = events_df["epoch"].max()

bin_edges = np.linspace(min_epoch, max_epoch, N_bins + 1)
bin_centers_epoch = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_centers_ts = (
    pl.from_epoch(pl.Series(bin_centers_epoch, dtype=pl.Int64), time_unit="s")
    .dt.replace_time_zone("UTC")
    .to_numpy()
)

# Initialize variables for single-pass replay (Fix 4)
ticks = {int(k): int(v) for k, v in start_ticks.items()}

binned_prices = np.zeros(N_bins)
binned_ticks_states = [None] * N_bins
binned_liquidities = np.zeros(N_bins)

last_price = None
last_liq = None

# Validation / Golden Check metrics (Fix 6)
total_swaps = 0
mismatch_count = 0

print("Replaying pool events chronologically...")
bin_idx = 0

for row in events_df.iter_rows(named=True):
    t0_ = row["topic0"]
    event_epoch = row["epoch"]

    # Save tick state and price for bins that the timeline has passed
    while bin_idx < N_bins and event_epoch > bin_edges[bin_idx + 1]:
        binned_prices[bin_idx] = last_price if last_price is not None else 0.0
        binned_ticks_states[bin_idx] = ticks.copy()
        binned_liquidities[bin_idx] = last_liq if last_liq is not None else 0.0
        bin_idx += 1

    # Process Mint/Burn/Swap events
    if t0_ == mint or t0_ == burn:
        lower = be_int(row["topic2"], signed=True)
        upper = be_int(row["topic3"], signed=True)
        # Decode liquidity amount strictly as integer to avoid precision loss (Fix 2)
        if t0_ == mint:
            delta = be_int(row["data"][32:64], signed=False)
        else:
            delta = -be_int(row["data"][0:32], signed=False)

        ticks[lower] = ticks.get(lower, 0) + delta
        ticks[upper] = ticks.get(upper, 0) - delta

    elif t0_ == swap:
        # Decode Swap fields strictly in big-endian EVM format
        sqrtPriceX96_val = be_int(row["data"][64:96], signed=False)
        liquidity_val = be_int(row["data"][96:128], signed=False)
        tick_val = be_int(row["data"][128:160], signed=True)

        # Keep track of last price and in-range liquidity
        sqrtPriceX96_float = float(sqrtPriceX96_val) / (2**96)
        last_price = 1.0 / (sqrtPriceX96_float**2) * pow(10, d1 - d0)
        last_liq = float(liquidity_val)

        # Golden Check: Validate the active liquidity prefix sum matches the contract's emitted liquidity (Fix 6)
        total_swaps += 1
        computed_L = sum(net for t, net in ticks.items() if t <= tick_val)
        if computed_L != liquidity_val:
            mismatch_count += 1

# Fill remaining bins to the end of the timeline
while bin_idx < N_bins:
    binned_prices[bin_idx] = last_price if last_price is not None else 0.0
    binned_ticks_states[bin_idx] = ticks.copy()
    binned_liquidities[bin_idx] = last_liq if last_liq is not None else 0.0
    bin_idx += 1

print(f"Replay complete. Verified {total_swaps} swaps.")
print(
    f"Validation Mismatches: {mismatch_count} / {total_swaps} ({(mismatch_count/total_swaps*100.0 if total_swaps > 0 else 0.0):.2f}%)"
)

# Define the vertical price range to exactly 500 to 3000 (Fix 7)
min_price = 500.0
max_price = 3000.0

prices_grid = np.linspace(min_price, max_price, M)

# Pre-convert all price grid bins to tick indices
ticks_grid = np.array(
    [price_to_tick(p, d0, d1, price_is_token1_in_token0=True) for p in prices_grid]
)

print(f"Generating 2D active liquidity grid of shape ({M}, {N_bins})...")
liquidity_grid = np.zeros((M, N_bins))

for j in range(N_bins):
    ticks_dict = binned_ticks_states[j]
    if not ticks_dict:
        continue

    sorted_ticks = sorted(ticks_dict.keys())
    if not sorted_ticks:
        continue

    # Compute active liquidity profile across tick space using exact integer prefix sum (Fix 2)
    cum_liq = 0
    tick_liquidity = []
    for tick in sorted_ticks:
        cum_liq += ticks_dict[tick]
        tick_liquidity.append(max(0, cum_liq))

    total_pool_liq = sum(tick_liquidity)

    # Find active liquidity for each price level using fast binary search
    for i, T in enumerate(ticks_grid):
        tick_idx = bisect.bisect_right(sorted_ticks, T) - 1
        if tick_idx >= 0 and total_pool_liq > 0:
            liquidity_grid[i, j] = (
                float(tick_liquidity[tick_idx]) / total_pool_liq
            ) * 100.0
        else:
            liquidity_grid[i, j] = 0.0

# Mask exact zeros (Fix 3)
grid = liquidity_grid.astype(float)
grid[grid <= 0] = np.nan

# Define robust norm bounds to prevent floating residue from shrinking the range (Fix 3)
valid_vals = grid[~np.isnan(grid)]
if len(valid_vals) > 0:
    vmin = max(0.001, np.nanpercentile(valid_vals, 2))
    vmax = np.nanmax(valid_vals)
else:
    vmin, vmax = 0.01, 100.0

# Ensure log-norm is happy
vmin = max(vmin, 1e-4)
vmax = max(vmax, vmin * 10)

norm = colors.LogNorm(vmin=vmin, vmax=vmax)

# ==============================================================================
# PLOT 2D LIQUIDITY DEPTH HEATMAP
# ==============================================================================
print("Plotting the liquidity depth heatmap...")
fig, ax = plt.subplots(figsize=(15, 9), facecolor="#111111")
ax.set_facecolor("#111111")

# Red-to-green colormap representing absolute active liquidity depth (Fix 3)
cmap = plt.get_cmap("RdYlGn").copy()
cmap.set_bad("#111111")  # Mask zero-liquidity ranges to background color

pcm = ax.pcolormesh(
    bin_centers_ts,
    prices_grid,
    grid,
    shading="auto",
    cmap=cmap,
    norm=norm,
    alpha=1.0,  # Avoid nonlinear darkening over black background (Fix 3)
)

# Foreground line representing binned price path (Black line as requested)
ax.plot(bin_centers_ts, binned_prices, color="black", linewidth=3.0, label="Swap Price")

# Axis labels & styling
ax.set_title(
    "Uniswap V3 Historical Liquidity Depth Heatmap (USDC/ETH)",
    fontsize=16,
    color="white",
    pad=15,
    fontweight="bold",
)
ax.set_xlabel("Timestamp", fontsize=12, color="white")
ax.set_ylabel("Price (USDC per ETH)", fontsize=12, color="white")
ax.tick_params(colors="white", which="both")
ax.grid(True, linestyle="--", color="#444444", alpha=0.4)

# Colorbar for active liquidity depth formatted as percentage
cbar = fig.colorbar(pcm, ax=ax, pad=0.02, format=ticker.PercentFormatter(xmax=100.0))
cbar.ax.yaxis.set_tick_params(color="white")
cbar.set_label(
    "Active Liquidity (% of Pool's Current Active Liquidity)",
    fontsize=12,
    color="white",
    labelpad=10,
)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

ax.legend(loc="upper left", facecolor="#222222", edgecolor="white", labelcolor="white")
plt.tight_layout()

# Save the figure as an image
plt.savefig(
    "liquidity_depth_heatmap.png",
    dpi=150,
    facecolor=fig.get_facecolor(),
    edgecolor="none",
)
plt.show()
