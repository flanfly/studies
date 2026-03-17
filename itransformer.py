# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["parameters"]
input_ohlcv_file = "stables1d.parquet"
output_features_file = "itransformer_features.parquet"
output_model_file = "itransformer_model.pth"

market_pair = "BTCUSDT"

# Dev Mode: must be exactly "yes" to enable; all other values are treated as "no"
dev_mode = "yes"

# Training Parameters (all parameters are strings and converted in the next cell)
device_type = "auto"  # "cuda", "cpu", or "auto"
seq_len = "30"
batch_size = "1024"
epochs = "10"

# Labeling Parameters
horizon = "3"
rolling_vol_window = "60"
pump_threshold_sigma = "1.0"
dump_threshold_sigma = "1.0"

# Data Split Percentages
train_pct = "0.7"
val_pct = "0.1"
test_pct = "0.2"

# Learning Rates
lr_nn = "1e-3"
confidence_threshold = "0.95"

# %%
# convert string parameters and print them
import numpy as np
import torch

seq_len = int(seq_len)
batch_size = int(batch_size)
epochs = int(epochs)

horizon = int(horizon)
rolling_vol_window = int(rolling_vol_window)
pump_threshold_sigma = float(pump_threshold_sigma)
dump_threshold_sigma = float(dump_threshold_sigma)

train_pct = float(train_pct)
val_pct = float(val_pct)
test_pct = float(test_pct)

lr_nn = float(lr_nn)
confidence_threshold = float(confidence_threshold)

device_type = str(device_type).strip().lower()
if device_type == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_type)

is_dev_mode = str(dev_mode).strip().lower() == "yes"

print("Parameters:")
print(f"  input_ohlcv_file: {input_ohlcv_file}")
print(f"  output_features_file: {output_features_file}")
print(f"  market_pair: {market_pair}")
print(f"  dev_mode: {dev_mode} -> is_dev_mode={is_dev_mode}")
print(f"  device: {device}")
print(f"  seq_len: {seq_len}")
print(f"  batch_size: {batch_size}")
print(f"  epochs: {epochs}")
print(f"  horizon: {horizon}")
print(f"  rolling_vol_window: {rolling_vol_window}")
print(f"  pump_threshold_sigma: {pump_threshold_sigma}")
print(f"  dump_threshold_sigma: {dump_threshold_sigma}")
print(f"  train_pct: {train_pct}")
print(f"  val_pct: {val_pct}")
print(f"  test_pct: {test_pct}")
print(f"  lr_nn: {lr_nn}")
print(f"  confidence_threshold: {confidence_threshold}")

# %%
# derive features
import polars as pl

eps = 1e-8

# Define the features we actually use
features_list = [
    "rv_30",
    "rv_90",
    "sigma_rs",
    "hl_range",
    "vol_z_30",
    "rel_volume_30",
    "log_quote_vol",
    "log_base_vol",
    "mom_3",
    "rel_mom_3",
    "ret",
    "cc_ret",
    "lower_wick_frac",
    "wick_asym",
    "upper_wick_frac",
]

raw = (
    pl.scan_parquet(input_ohlcv_file)
    .sort(["symbol", "ts"])
    .filter(
        (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
        & (pl.col("base_volume") > 0)
        & (pl.col("quote_volume") > 0)
    )
    .with_columns(
        # base returns (all use log returns as requested)
        ret=(pl.col("close") / pl.col("open")).log(),
        cc_ret=(pl.col("close") / pl.col("close").shift(1)).log().over("symbol"),
        # bar structure (also log-based)
        hl_range=(pl.col("high") / pl.col("low")).log(),
        # wick and body ratios
        upper_wick_frac=(
            (pl.col("high") - pl.max_horizontal("open", "close"))
            / (pl.col("high") - pl.col("low") + pl.lit(eps))
        ),
        lower_wick_frac=(
            (pl.min_horizontal("open", "close") - pl.col("low"))
            / (pl.col("high") - pl.col("low") + pl.lit(eps))
        ),
        wick_asym=(
            (pl.col("high") - pl.max_horizontal("open", "close") + pl.lit(eps))
            / (pl.min_horizontal("open", "close") - pl.col("low") + pl.lit(eps))
        ).log(),
        # range-based vol (log-ratio based)
        sigma_rs=(
            (
                (pl.col("high") / pl.col("close")).log()
                * (pl.col("high") / pl.col("open")).log()
            )
            + (
                (pl.col("low") / pl.col("close")).log()
                * (pl.col("low") / pl.col("open")).log()
            )
        ).sqrt(),
        # volume / participation (log-volume based)
        log_quote_vol=(pl.col("quote_volume") + pl.lit(eps)).log(),
        log_base_vol=(pl.col("base_volume") + pl.lit(eps)).log(),
    )
)

# Use the configured market_pair as reference market series
market_ref = raw.filter(pl.col("symbol") == market_pair).select(
    ["ts", pl.col("ret").alias("ref")]
)

(
    raw.join(market_ref, on="ts", how="left")
    .with_columns(
        # if market_pair is missing for some timestamp, keep pipeline stable
        pl.col("ref").fill_null(0.0),
        ret_rel=pl.col("ret") - pl.col("ref"),
    )
    .with_columns(
        # momentum (sum of log returns is log of cumulative return)
        mom_3=pl.col("ret").rolling_sum(window_size=3).over("symbol"),
        # relative momentum (difference of log returns)
        rel_mom_3=pl.col("ret_rel").rolling_sum(window_size=3).over("symbol"),
        # realized vol (stdev of log returns)
        rv_30=pl.col("ret").rolling_std(window_size=30).over("symbol"),
        rv_90=pl.col("ret").rolling_std(window_size=90).over("symbol"),
        # volume baselines (using log-volume)
        log_quote_vol_mean_30=pl.col("log_quote_vol")
        .rolling_mean(window_size=30)
        .over("symbol"),
        log_quote_vol_std_30=pl.col("log_quote_vol")
        .rolling_std(window_size=30)
        .over("symbol"),
    )
    .with_columns(
        # volume surprise
        vol_z_30=(
            (pl.col("log_quote_vol") - pl.col("log_quote_vol_mean_30"))
            / (pl.col("log_quote_vol_std_30") + pl.lit(eps))
        ),
        rel_volume_30=pl.col("quote_volume")
        / (
            pl.col("quote_volume").rolling_mean(window_size=30).over("symbol")
            + pl.lit(eps)
        ),
    )
    .with_columns(
        # --- Volatility-adjusted labeling logic using log returns ---
        # 1. Forward Return: cumulative log returns over horizon N (starting at t+1)
        fwd_ret=pl.col("ret")
        .shift(-horizon)
        .rolling_sum(window_size=horizon)
        .over("symbol"),
        # 2. Rolling Volatility: stdev of N-day backward log returns
        # We first compute the N-day backward return series, then take its rolling std
        hist_vol=(
            pl.col("ret")
            .rolling_sum(window_size=horizon)
            .over("symbol")
            .rolling_std(window_size=rolling_vol_window)
            .over("symbol")
        ),
    )
    .with_columns(
        # 3. Standardized Forward Return (Z-score)
        target_z=pl.col("fwd_ret")
        / (pl.col("hist_vol") + pl.lit(eps))
    )
    .with_columns(
        # 4. Labeling based on sigma thresholds
        label=pl.when(pl.col("target_z") >= pump_threshold_sigma)
        .then(2)
        .when(pl.col("target_z") <= -dump_threshold_sigma)
        .then(0)
        .otherwise(1)
    )
    .select(["ts", "symbol", "fwd_ret", "target_z", "label"] + features_list)
    .sink_parquet(output_features_file)
)

print(f"Features saved to {output_features_file}")

# %%
# training the itransformer for classification
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score


# iTransformer Encoder with Feature Embeddings
class iTransformerEncoder(nn.Module):
    def __init__(self, seq_len, num_features, d_model=128, n_heads=4, num_layers=3):
        super().__init__()
        self.time_projector = nn.Linear(seq_len, d_model)

        # Learnable Feature Embedding
        self.feature_embed = nn.Parameter(torch.randn(1, num_features, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: [Batch, Time, Features]
        x = x.permute(0, 2, 1)  # [Batch, Features, Time]
        x = self.time_projector(x)  # [Batch, Features, d_model]

        # Add feature identity information
        x = x + self.feature_embed

        out = self.transformer(x)  # [Batch, Features, d_model]
        return out


class Supervised_iTransformer(nn.Module):
    def __init__(
        self, seq_len, num_features, num_classes=3, d_model=128, n_heads=4, num_layers=3
    ):
        super().__init__()
        self.encoder = iTransformerEncoder(
            seq_len, num_features, d_model, n_heads, num_layers
        )
        self.flatten_dim = num_features * d_model
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        flat_out = enc_out.flatten(start_dim=1)
        logits = self.classifier(flat_out)
        return logits


# Dataset class (Expects labels/rets to be pre-calculated)
class CryptoPanelDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: list,
        seq_len: int,
    ):
        self.seq_len = seq_len
        df = df.sort(["symbol", "ts"])
        df = df.with_columns(
            pl.int_range(0, pl.len()).over("symbol").alias("row_num_in_group")
        )
        df = df.with_row_index("global_idx")

        # Ensure windows have enough history AND have a valid future label
        valid_rows = df.filter(
            (pl.col("row_num_in_group") >= (seq_len - 1))
            & (pl.col("label").is_not_null())
        )
        self.valid_indices = valid_rows["global_idx"].to_list()

        feature_data = df.select(feature_cols).to_numpy()
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        self.features = torch.tensor(feature_data, dtype=torch.float32)
        self.labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
        self.fwd_rets = torch.tensor(df["fwd_ret"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len + 1
        x = self.features[start_idx : end_idx + 1]
        y = self.labels[end_idx]
        fwd_ret = self.fwd_rets[end_idx]
        return x, y, fwd_ret


# Load and Split Data
print("Loading data...")
full_df = pl.read_parquet(output_features_file).sort(["symbol", "ts"])
unique_ts = full_df.select("ts").unique().sort("ts")["ts"]
n_ts = len(unique_ts)
train_cutoff_ts = unique_ts[int(n_ts * train_pct)]
val_cutoff_ts = unique_ts[int(n_ts * (train_pct + val_pct))]

train_df = full_df.filter(pl.col("ts") < train_cutoff_ts)
val_df = full_df.filter(
    (pl.col("ts") >= train_cutoff_ts) & (pl.col("ts") < val_cutoff_ts)
)
test_df = full_df.filter(pl.col("ts") >= val_cutoff_ts)
print(f"Split rows - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Normalize features
print("Normalizing features...")
train_stats = train_df.select(
    [pl.col(c).mean().alias(f"{c}_mean") for c in features_list]
    + [pl.col(c).std().alias(f"{c}_std") for c in features_list]
)
means = {c: train_stats[f"{c}_mean"][0] for c in features_list}
stds = {c: train_stats[f"{c}_std"][0] for c in features_list}


def normalize_df(df, means, stds, features):
    exprs = []
    for c in features:
        mean = means.get(c, 0.0)
        std = stds.get(c, 1.0)
        if std == 0 or std is None or np.isnan(std):
            std = 1.0
        exprs.append(((pl.col(c) - mean) / std).alias(c))
    return df.with_columns(exprs)


train_df = normalize_df(train_df, means, stds, features_list)
val_df = normalize_df(val_df, means, stds, features_list)
test_df = normalize_df(test_df, means, stds, features_list)

# Datasets
train_dataset = CryptoPanelDataset(train_df, features_list, seq_len)

# Resolve training params
if is_dev_mode:
    print("!!! DEV MODE ACTIVE !!!")
    active_epochs = 1
    active_batch_size = 256
    max_train_batches = 50
else:
    active_epochs = epochs
    active_batch_size = batch_size
    max_train_batches = None

train_dataloader = DataLoader(
    train_dataset, batch_size=active_batch_size, shuffle=True, drop_last=True
)

# Compute class weights for CE loss
train_labels_for_weight = train_dataset.labels[train_dataset.valid_indices].numpy()
class_counts = np.bincount(train_labels_for_weight)
class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(
    device
)
class_weights = class_weights / class_weights.sum() * 3.0
print(f"Class weights: {class_weights}")

# Model, Loss, Optimizer
model = Supervised_iTransformer(seq_len=seq_len, num_features=len(features_list)).to(
    device
)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_nn)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# Training Loop
print(f"Starting training for {active_epochs} epochs...")
for epoch in range(active_epochs):
    model.train()
    epoch_losses = []
    for i, (x_batch, y_batch, _) in enumerate(train_dataloader):
        if max_train_batches and i >= max_train_batches:
            break
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_losses.append(loss.item())
        if (i + 1) % 10 == 0:
            print(f"Batch {i+1} | Loss: {loss.item():.4f}")

    avg_loss = np.mean(epoch_losses)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)

torch.save(model.state_dict(), output_model_file)
print(f"Model saved to {output_model_file}")

# %%
# evaluating the itransformer on validation and test set
model.eval()
print("\nEvaluating on evaluation set (Val + Test)...")

val_df_all = pl.concat([val_df, test_df])
eval_dataset = CryptoPanelDataset(val_df_all, features_list, seq_len)
eval_dataloader = DataLoader(eval_dataset, batch_size=active_batch_size, shuffle=False)

all_preds = []
all_labels = []
all_fwd_rets = []

with torch.no_grad():
    for i, (x_batch, y_batch, fwd_ret_batch) in enumerate(eval_dataloader):
        if is_dev_mode and i >= 100:
            break
        x_batch = x_batch.to(device)
        logits = model(x_batch)

        # Apply Softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # Confidence-based prediction
        preds = torch.ones(x_batch.size(0), dtype=torch.long, device=device)

        pump_mask = probs[:, 2] > confidence_threshold
        preds[pump_mask] = 2

        dump_mask = probs[:, 0] > confidence_threshold
        preds[dump_mask] = 0

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.numpy())
        all_fwd_rets.append(fwd_ret_batch.numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_labels)
fwd_rets_eval = np.concatenate(all_fwd_rets)

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(
    classification_report(
        y_true, y_pred, target_names=["Dump", "Flat", "Pump"], zero_division=0
    )
)

print("\nPerformance Stats (Forward Returns):")


def report_stats(name, mask, returns):
    subset = returns[mask]
    if len(subset) > 0:
        valid_subset = subset[~np.isnan(subset)]
        if len(valid_subset) > 0:
            print(
                f"{name:15s} | Mean: {valid_subset.mean():.4%}, Stdev: {valid_subset.std():.4%}, Count: {len(valid_subset)}"
            )
        else:
            print(f"{name:15s} | No valid return values found in subset.")
    else:
        print(f"{name:15s} | No samples found.")


report_stats("Long (Pred P)", (y_pred == 2), fwd_rets_eval)
report_stats("Short (Pred D)", (y_pred == 0), fwd_rets_eval)
report_stats("Up (Actual P)", (y_true == 2), fwd_rets_eval)
report_stats("Down (Actual D)", (y_true == 0), fwd_rets_eval)
