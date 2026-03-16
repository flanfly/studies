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
output_features_file = "scl_features.parquet"
output_nn_file = "scl_embedding_model.pth"
output_clf_file = "scl_lr_classifier.joblib"

market_pair = "BTCUSDT"

# Dev Mode: must be exactly "yes" to enable; all other values are treated as "no"
dev_mode = "yes"

# Training Parameters (used if dev_mode=False)
device_type = "auto"  # "cuda", "cpu", or "auto"
seq_len = 30
batch_size = 1024
epochs = 10

# Data Split Percentages
train_pct = 0.7
val_pct = 0.1
test_pct = 0.2

# Learning Rates
lr_nn = 1e-3
lr_linear = 1e-2

# %%
# derive features
import polars as pl
import numpy as np

eps = 1e-8

# Define the features we actually use in the SCL part
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
        # base returns
        ret=(pl.col("close") / pl.col("open")).log(),
        cc_ret=(pl.col("close") / pl.col("close").shift(1)).log().over("symbol"),
        # bar structure
        hl_range=(pl.col("high") / pl.col("low")).log(),
        # wick and body
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
        # range-based vol
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
        # volume / participation
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
        # momentum
        mom_3=pl.col("ret").rolling_sum(window_size=3).over("symbol"),
        # relative momentum
        rel_mom_3=pl.col("ret_rel").rolling_sum(window_size=3).over("symbol"),
        # realized vol
        rv_30=pl.col("ret").rolling_std(window_size=30).over("symbol"),
        rv_90=pl.col("ret").rolling_std(window_size=90).over("symbol"),
        # volume baselines
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
    .select(["ts", "symbol"] + features_list)
    .sink_parquet(output_features_file)
)

print(f"Features saved to {output_features_file}")

# %%
# training the nn for the embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Resolve Dev Mode Overrides
is_dev_mode = isinstance(dev_mode, str) and dev_mode.strip().lower() == "yes"
if is_dev_mode:
    print("!!! DEV MODE ACTIVE - Overriding parameters !!!")
    active_epochs = 1
    active_batch_size = 256
    max_train_batches = 50
else:
    active_epochs = epochs
    active_batch_size = batch_size
    max_train_batches = None

# Resolve Device
if device_type == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_type)
print(f"Using device: {device}")


# iTransformer Encoder
class iTransformerEncoder(nn.Module):
    def __init__(self, seq_len, num_features, d_model=128, n_heads=4, num_layers=3):
        super().__init__()
        self.time_projector = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.time_projector(x)
        out = self.transformer(x)
        return out


# SCL Model wrapper
class SCL_iTransformer(nn.Module):
    def __init__(
        self, seq_len, num_features, d_model=128, n_heads=4, num_layers=3, proj_dim=64
    ):
        super().__init__()
        self.encoder = iTransformerEncoder(
            seq_len, num_features, d_model, n_heads, num_layers
        )
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, proj_dim)
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        representation = enc_out.mean(dim=1)
        projected_vector = F.normalize(self.projection_head(representation), p=2, dim=1)
        return representation, projected_vector


# Supervised Contrastive Loss
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections, labels):
        device = projections.device
        batch_size = projections.shape[0]
        sim_matrix = torch.matmul(projections, projections.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss


# Dataset class
class CryptoPanelDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: list,
        seq_len: int,
        pump_thresh: float,
        dump_thresh: float,
        ret_col: str = "ret",
    ):
        self.seq_len = seq_len
        df = df.sort(["symbol", "ts"])
        df = df.with_columns(pl.col(ret_col).shift(-1).over("symbol").alias("fwd_ret"))
        df = df.drop_nulls(subset=["fwd_ret"])
        df = df.with_columns(
            pl.when(pl.col("fwd_ret") >= pump_thresh)
            .then(2)
            .when(pl.col("fwd_ret") <= dump_thresh)
            .then(0)
            .otherwise(1)
            .alias("label")
        )
        df = df.with_columns(
            pl.int_range(0, pl.len()).over("symbol").alias("row_num_in_group")
        )
        df = df.with_row_index("global_idx")
        valid_rows = df.filter(pl.col("row_num_in_group") >= (seq_len - 1))
        self.valid_indices = valid_rows["global_idx"].to_list()
        feature_data = df.select(feature_cols).to_numpy()
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        self.features = torch.tensor(feature_data, dtype=torch.float32)
        self.labels = torch.tensor(df.select("label").to_numpy(), dtype=torch.long)
        self.fwd_rets = torch.tensor(
            df.select("fwd_ret").to_numpy(), dtype=torch.float32
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len + 1
        x = self.features[start_idx : end_idx + 1]
        y = self.labels[end_idx]
        fwd_ret = self.fwd_rets[end_idx]
        return x, y, fwd_ret


def extract_embeddings(dataloader, model, device, max_samples=None):
    model.eval()
    all_embeddings, all_labels, all_fwd_rets = [], [], []
    samples_collected = 0
    with torch.no_grad():
        for x_batch, y_batch, fwd_ret_batch in dataloader:
            x_batch = x_batch.to(device)
            representation, _ = model(x_batch)
            all_embeddings.append(representation.cpu().numpy())
            all_labels.append(y_batch.numpy())
            all_fwd_rets.append(fwd_ret_batch.numpy())
            samples_collected += x_batch.size(0)
            if max_samples and samples_collected >= max_samples:
                break
    return (
        np.vstack(all_embeddings),
        np.concatenate(all_labels),
        np.concatenate(all_fwd_rets),
    )


# Initialize Model and Optimizer
model = SCL_iTransformer(seq_len=seq_len, num_features=len(features_list)).to(device)
criterion = SupConLoss(temperature=0.07).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_nn)

# Load and Split Data
print("Loading data...")
full_df = pl.read_parquet(output_features_file)
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

# Normalize features based on Training Set statistics
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


# Create raw_ret column for labeling before normalization overwrites ret
train_df = train_df.with_columns(pl.col("ret").alias("raw_ret"))
val_df = val_df.with_columns(pl.col("ret").alias("raw_ret"))
test_df = test_df.with_columns(pl.col("ret").alias("raw_ret"))

train_df = normalize_df(train_df, means, stds, features_list)
val_df = normalize_df(val_df, means, stds, features_list)
test_df = normalize_df(test_df, means, stds, features_list)

# Create Training Dataset and Loader
train_dataset = CryptoPanelDataset(train_df, features_list, seq_len, 0.05, -0.05, ret_col="raw_ret")
valid_labels = train_dataset.labels[train_dataset.valid_indices].numpy().flatten()
class_counts = np.bincount(valid_labels)
class_weights = 1.0 / (class_counts + 1e-8)
sample_weights = torch.from_numpy(class_weights[valid_labels]).double()
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_dataloader = DataLoader(
    train_dataset, batch_size=active_batch_size, sampler=sampler, drop_last=True
)

# NN Training Loop
model.train()
print(f"Starting NN training for {active_epochs} epochs...")
for epoch in range(active_epochs):
    epoch_losses = []
    for i, (x_batch, y_batch, _) in enumerate(train_dataloader):
        if max_train_batches and i >= max_train_batches:
            break
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        _, projected_vector = model(x_batch)
        loss = criterion(projected_vector, y_batch)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        if (i + 1) % 10 == 0:
            print(f"Batch {i+1} | Loss: {loss.item():.4f}")
    if epoch_losses:
        print(f"Epoch {epoch+1} | Avg Loss: {np.mean(epoch_losses):.4f}")

# Save the trained model
torch.save(model.state_dict(), output_nn_file)
print(f"Model saved to {output_nn_file}")

# %%
# training the linear classifier
# Load the model
model = SCL_iTransformer(seq_len=seq_len, num_features=len(features_list)).to(device)
model.load_state_dict(torch.load(output_nn_file, map_location=device))
model.eval()

print("\nExtracting embeddings for linear classifier training...")
train_extract_loader = DataLoader(
    train_dataset, batch_size=active_batch_size, shuffle=False
)
X_train_emb, y_train_emb, _ = extract_embeddings(
    train_extract_loader, model, device, max_samples=50000 if is_dev_mode else None
)

print(f"Fitting Logistic Regression on {len(X_train_emb)} samples...")
lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train_emb, y_train_emb.ravel())
print("Linear classifier training complete.")

# Save the trained linear classifier
joblib.dump(lr_clf, output_clf_file)
print(f"Linear classifier saved to {output_clf_file}")

# %%
# computing metrics on the test and val set using the models
# Load the model
model = SCL_iTransformer(seq_len=seq_len, num_features=len(features_list)).to(device)
model.load_state_dict(torch.load(output_nn_file, map_location=device))
model.eval()

# Load the linear classifier
lr_clf = joblib.load(output_clf_file)

print("\nEvaluating models on evaluation set (Val + Test)...")
val_df_all = pl.concat([val_df, test_df])
eval_dataset = CryptoPanelDataset(val_df_all, features_list, seq_len, 0.05, -0.05, ret_col="raw_ret")
eval_extract_loader = DataLoader(
    eval_dataset, batch_size=active_batch_size, shuffle=False
)
X_eval_emb, y_eval_emb, fwd_rets_eval = extract_embeddings(
    eval_extract_loader, model, device, max_samples=20000 if is_dev_mode else None
)

y_pred = lr_clf.predict(X_eval_emb)
acc = accuracy_score(y_eval_emb, y_pred)
print(f"Linear Classifier Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_eval_emb, y_pred, target_names=["Dump", "Flat", "Pump"], zero_division=0
    )
)

print("\nPerformance Stats (Forward Returns):")


def report_stats(name, mask, returns):
    subset = returns[mask]
    if len(subset) > 0:
        print(
            f"{name:15s} | Mean: {subset.mean():.4%}, Stdev: {subset.std():.4%}, Count: {len(subset)}"
        )
    else:
        print(f"{name:15s} | No samples found.")


report_stats("Long (Pred P)", (y_pred == 2), fwd_rets_eval)
report_stats("Short (Pred D)", (y_pred == 0), fwd_rets_eval)
report_stats("Up (Actual P)", (y_eval_emb.ravel() == 2), fwd_rets_eval)
report_stats("Down (Actual D)", (y_eval_emb.ravel() == 0), fwd_rets_eval)
