import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import torch

# 1. Download intraday data
symbol = "BTC-USD"
df = yf.download(tickers=symbol, period="20d", interval="5m")

# Keep only relevant columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# 2. Exploratory Data Analysis (EDA)
print("=== EDA: Missing Data ===")
print(df.isna().sum())

print("\n=== EDA: Summary Statistics ===")
print(df.describe())

# Plot Close price
df['Close'].plot(title=f"{symbol} Close Price")
plt.show()

# Check irregularity: time deltas
df['timestamp'] = df.index
df['delta_t'] = df['timestamp'].diff().dt.total_seconds() / 60  # minutes
print("\n=== EDA: Time delta stats (minutes) ===")
print(df['delta_t'].describe())

# Check returns distribution
df['log_return'] = np.log(df['Close']).diff()
df['log_return'].plot(title="Log returns")
plt.show()
print(df['log_return'].describe())

# Check stationarity (ADF test)
adf_result = adfuller(df['log_return'].dropna())
print("\nADF Test for log returns:")
print(f"ADF statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")

# 3. Feature Engineering / Indicators
# EMA 14
df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()

# VWAP
typical_price = (df['High'] + df['Low'] + df['Close']) / 3
df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

# 4. Preprocessing for ODE-RNN

features = ['Close', 'Volume', 'EMA_14', 'VWAP']

# Remplir les NaN dans les colonnes features
df_norm = df.copy()
df_norm[features] = df_norm[features].fillna(method='ffill').fillna(0)

# Normalisation z-score
df_norm[features] = (df_norm[features] - df_norm[features].mean()) / df_norm[features].std()

df_norm['delta_t'] = df['delta_t'].fillna(5.0)  # valeur par défaut si manquante

# 2. Sequences creation
def create_sequences(data, seq_len=20, features=features):
    sequences = []
    delta_ts = []
    for i in range(len(data) - seq_len):
        seq = data[features].iloc[i:i+seq_len].values
        dt = data['delta_t'].iloc[i+1:i+seq_len+1].values  # Δt pour chaque pas
        sequences.append(seq)
        delta_ts.append(dt)
    return np.array(sequences), np.array(delta_ts)

seq_len = 20
X_seq, delta_t_seq = create_sequences(df_norm, seq_len=seq_len, features=features)

delta_t_seq = np.nan_to_num(delta_t_seq, nan=5.0)
delta_t_seq = np.maximum(delta_t_seq, 1e-3)

print("X_seq shape:", X_seq.shape)
print("delta_t_seq shape:", delta_t_seq.shape)

# 3. torch.tensor conversion
X_tensor = torch.tensor(X_seq, dtype=torch.float32)          
delta_t_tensor = torch.tensor(delta_t_seq, dtype=torch.float32)  

delta_t_tensor = delta_t_tensor.T           
X_tensor.shape

# 4. Sauvegarde
torch.save(X_tensor, "data/raw/numerical/X_tensor.pt")
torch.save(delta_t_tensor, "data/raw/numerical/delta_t_tensor.pt")

print("Tenseurs sauvegardés.")
