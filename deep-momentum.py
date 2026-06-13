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

# %% [markdown]
# # Lim et al.: Enhancing Time Series Momentum Strategies Using Deep Neural Networks
#
# Use a LSTM to learn the optimal momentum signals. Optimizes the Sortino ratio instead of a generic loss function.

# %%
import polars as pl
import numpy as np

horizons = [1, 3, 7, 21, 63]

df = (
    pl.read_parquet('stables-1d.parquet')
    .sort(["symbol", "ts"])
    .with_columns(
        ts=pl.col('ts').cast(pl.Datetime("ms")),
        ret=pl.col("close").pct_change().over("symbol"),
        ho = (pl.col('high') / pl.col('open')).log(),
        hc = (pl.col('high') / pl.col('close')).log(),
        lo = (pl.col('low') / pl.col('open')).log(),
        lc = (pl.col('low') / pl.col('close')).log(),
    )
    .with_columns(
        var=(pl.col('ho') * pl.col('hc')) + (pl.col('lo') * pl.col('lc'))
    )
    .with_columns(
        vol=pl.col('var').rolling_mean(window_size=60).over('symbol').mul(365).sqrt(),
    )
    .drop_nulls(subset=['vol']) 
    .with_columns(**{
        f'mom{k}d': (pl.col("close") / pl.col("close").shift(k).over("symbol") - 1.0) / (pl.col("vol") * np.sqrt(k))
        for k in horizons
    })
    # MACD = 12-day EMA - 26-day EMA
    .with_columns(
        ema12d=pl.col("close").ewm_mean(span=12, ignore_nulls=True).over("symbol"),
        ema26d=pl.col("close").ewm_mean(span=26, ignore_nulls=True).over("symbol"),
    )
    .with_columns(
        macd=pl.col("ema12d") - pl.col("ema26d"),
    )
    .with_columns(
        signal=pl.col('macd').ewm_mean(span=9, ignore_nulls=True).over("symbol"),
        target=(pl.col("ret").shift(-1) / pl.col("vol")).over("symbol")
    )
    .select(['ts','signal','macd','target','symbol', *[f'mom{k}d' for k in horizons]])
    .drop_nulls()
)

df

# %%
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import datetime as dt


class TimeSeriesMomentumDataset(Dataset):
    def __init__(self, df: pl.DataFrame, seq_len: int = 63):
        self.seq_len = seq_len
        
        # Define feature columns
        self.feature_cols = ['signal','macd', *[f'mom{k}d' for k in horizons]]
        
        # Group by symbol to create valid sequences (no cross-asset leakage)
        self.samples = []
        
        for symbol, group in df.group_by("symbol"):
            features = group.select(self.feature_cols).to_numpy()
            targets = group.select("target").to_numpy()
            
            # Create rolling windows
            for i in range(len(features) - seq_len):
                x_seq = features[i : i + seq_len]
                y_target = targets[i + seq_len - 1] # Target aligns with the end of the sequence
                self.samples.append((x_seq, y_target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class SortinoLoss(nn.Module):
    def __init__(self, target_return=0.0, eps=1e-4, annualization_factor=365):
        super(SortinoLoss, self).__init__()
        self.target_return = target_return
        
        # A much larger epsilon acts as a volatility floor
        # 1e-4 prevents the denominator from shrinking below 0.01 (1% daily vol floor)
        self.eps = eps 
        self.annualization_factor = annualization_factor

    def forward(self, positions, future_returns):
        strategy_returns = positions * future_returns

        mean_return = torch.mean(strategy_returns)

        # Downside Deviation
        downside_diff = torch.clamp(strategy_returns - self.target_return, max=0.0)
        downside_variance = torch.mean(downside_diff ** 2)
        
        # Add the larger epsilon before the square root
        downside_dev = torch.sqrt(downside_variance + self.eps)

        # Calculate Daily Sortino, then Annualize
        daily_sortino = (mean_return - self.target_return) / downside_dev
        annualized_sortino = daily_sortino * np.sqrt(self.annualization_factor)
        position_penalty = 0.01 * torch.mean(positions ** 2)

        # Return negative ratio + penalty
        return -annualized_sortino + position_penalty
        #return -annualized_sortino


class SharpeLoss(nn.Module):
    def __init__(self, risk_free_rate=0.0, eps=1e-4, annualization_factor=365, penalty_weight=0.01):
        """
        Optimizes the Annualized Sharpe Ratio.
        
        Args:
            risk_free_rate: Target return or risk-free rate (typically 0.0 for daily crypto).
            eps: Volatility floor to prevent division by zero and gradient explosions.
            annualization_factor: 365 for crypto, 252 for traditional equities/futures.
            penalty_weight: Weight of the L2 position penalty to prevent tanh saturation.
        """
        super(SharpeLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
        self.eps = eps
        self.annualization_factor = annualization_factor
        self.penalty_weight = penalty_weight

    def forward(self, positions, future_returns):
        # 1. Calculate the strategy's daily returns
        # positions: [batch_size, 1] bounded between -1 and 1
        # future_returns: [batch_size, 1]
        strategy_returns = positions * future_returns

        # 2. Calculate Expected Return (Numerator)
        mean_return = torch.mean(strategy_returns)

        # 3. Calculate Volatility (Denominator)
        # We calculate variance manually rather than using torch.var to safely add epsilon before sqrt
        variance = torch.mean((strategy_returns - mean_return) ** 2)
        volatility = torch.sqrt(variance + self.eps)

        # 4. Calculate Daily Sharpe
        daily_sharpe = (mean_return - self.risk_free_rate) / volatility

        # 5. Annualize the Sharpe Ratio
        annualized_sharpe = daily_sharpe * np.sqrt(self.annualization_factor)

        # 6. Position Magnitude Penalty
        # Encourages the model to only take 100% positions when conviction is high
        position_penalty = self.penalty_weight * torch.mean(positions ** 2)

        # Return negative Sharpe (because PyTorch minimizes loss) plus the penalty
        return -annualized_sharpe + position_penalty


class DeepMomentumLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=40, dropout_rate=0.3):
        super(DeepMomentumLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        # PyTorch native LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, x):
        # 1. Input Dropout (Variational style applies to inputs too)
        x = self.dropout(x)
        
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]
        
        norm_out = self.layer_norm(last_step_out)
        
        # 2. Output Dropout
        drop_out = self.dropout(norm_out)
        position = torch.tanh(self.fc(drop_out))
        return position


# 1. --- Chronological Data Splitting ---
# We assume 'ts' is a datetime column. 
seq_len = 63

# Train: Everything up to end of 2024
train_end_date = dt.datetime(2025, 1, 1)
train_df = df.filter(pl.col("ts") < train_end_date)

# Val (2025): Needs a 63-day warmup from 2024 to predict Jan 1, 2025
val_start_date = dt.datetime(2025, 1, 1)
val_end_date = dt.datetime(2026, 1, 1)
val_warmup_date = val_start_date - dt.timedelta(days=seq_len)

val_df = df.filter(
    (pl.col("ts") >= val_warmup_date) & 
    (pl.col("ts") < val_end_date)
)

# Test (2026): Needs a 63-day warmup from 2025 to predict Jan 1, 2026
test_start_date = dt.datetime(2026, 1, 1)
test_warmup_date = test_start_date - dt.timedelta(days=seq_len)

test_df = df.filter(pl.col("ts") >= test_warmup_date)

print(f"Train size (with warmup): {len(train_df)}")
print(f"Val size (with warmup): {len(val_df)}")
print(f"Test size (with warmup): {len(test_df)}")

# 2. --- Datasets & DataLoaders ---
batch_size = 256

train_dataset = TimeSeriesMomentumDataset(train_df, seq_len=seq_len)
val_dataset = TimeSeriesMomentumDataset(val_df, seq_len=seq_len)
test_dataset = TimeSeriesMomentumDataset(test_df, seq_len=seq_len)

# Shuffle training data, but DO NOT shuffle val/test so we can evaluate in order
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepMomentumLSTM(input_size=len(train_dataset.feature_cols), hidden_size=64).to(device)

criterion = SortinoLoss(target_return=0.0)
#criterion = SharpeLoss()


# Add Weight Decay to the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 4. --- Training Loop with Validation Tracking ---
epochs = 20

for epoch in range(epochs):
    # --- Training Phase ---
    model.train()
    epoch_train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).unsqueeze(1) # Match position shape
        
        optimizer.zero_grad()
        positions = model(batch_x)
        loss = criterion(positions, batch_y)

        loss.backward()
        # Prevents gradients from exploding due to price outliers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()
        
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    
    # --- Validation Phase (2025 Data) ---
    model.eval() # Disable dropout/batchnorm during inference
    epoch_val_loss = 0.0
    
    with torch.no_grad(): # Don't compute gradients for validation
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            positions = model(batch_x)
            loss = criterion(positions, batch_y)
            epoch_val_loss += loss.item()
            
    # Protect against empty dataloaders during testing
    avg_val_loss = epoch_val_loss / max(len(val_loader), 1)

    # Calculate position stats for the last validation batch
    pos_mean = positions.mean().item()
    pos_std = positions.std().item()
    
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Train Sortino: {-avg_train_loss:5.4f} | "
          f"Val Sortino: {-avg_val_loss:5.4f} | "
          f"Pos Mean: {pos_mean:5.2f} | Pos Std: {pos_std:5.4f}")
    
# Optional: Final Test on 2026 Data
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
        positions = model(batch_x)
        loss = criterion(positions, batch_y)
        test_loss += loss.item()
        
print(f"\nFinal Test (2026) Sortino: {-(test_loss / max(len(test_loader), 1)):5.4f}")

# %%
