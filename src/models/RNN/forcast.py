import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from Deep_learning_gold_stock.src.models.RNN.GRU import GRUModel
from Deep_learning_gold_stock.src.models.RNN.ODERNN import ODE_RNN
from Deep_learning_gold_stock.src.models.RNN.LSTM import LSTMModel
from Deep_learning_gold_stock.src.models.RNN.main import eval_pred, evaluate_model
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error, r2_score


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
HIDDEN_DIM = 128 
NUM_LAYERS = 4
GRAD_CLIP = 1.0
DROPOUT = 0.3
HORIZON = 2

# Load Data
log_ret = pd.read_pickle("Deep_learning_gold_stock/data/raw/numerical/log_ret.pkl")
X_tensor = torch.load("Deep_learning_gold_stock/data/raw/numerical/X_tensor.pt").float()
features = ["log_future_var_2", "log_rv_weekly", "log_rv_monthly", "vol_5", "vol_21", "hl_range", "log_vix"]

data = pd.DataFrame()
for col, indice in zip(features, np.arange(0,8,1)): 
    data[col] = X_tensor[:,-1,indice].cpu().numpy()

volat_idx = 0

y_list = []
for i in range(len(X_tensor) - HORIZON):
    # Prédiction HORIZON steps ahead
    y_target = X_tensor[i + HORIZON, -1, volat_idx]  
    y_list.append(y_target)

y_tensor = torch.stack(y_list)  
X_tensor = X_tensor[:len(y_tensor),:,1:]  

print("X_tensor:", X_tensor.shape)
print("y_tensor:", y_tensor.shape)

X_tensor = X_tensor[:len(y_tensor)]

log_mean = y_tensor.mean()
log_std = y_tensor.std()
print("mean log return : ",log_mean)
print("std log return : ", log_std)

train_size = int(0.8 * X_tensor.shape[0])

X_train = X_tensor[:train_size]
X_test  = X_tensor[train_size:]

y_train = y_tensor[:train_size]
y_test  = y_tensor[train_size:]

mean = X_train.mean(dim=(0,1), keepdim=True)
std  = X_train.std(dim=(0,1), keepdim=True) + 1e-8

# normalisation
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

y_mean = y_train.mean()
y_std  = y_train.std() + 1e-8

y_train = (y_train - y_mean) / y_std
y_test  = (y_test  - y_mean) / y_std

print("X_train mean:", X_train.mean())
print("X_train std:", X_train.std())


train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------------------------
# Result Validation models
# --------------------------------------------
model_1 = LSTMModel(input_dim=X_train.shape[2], hidden_dim=HIDDEN_DIM,num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
model_2 = GRUModel(input_dim=X_train.shape[2], hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
model_3 = ODE_RNN(X_tensor.shape[2], HIDDEN_DIM).to(DEVICE)

for model, name in zip([model_1, model_2, model_3], ["LSTM_2d_HYG.pt","GRU_2d_HYG.pt", "ODERNN_2d_HYG.pt"]):
    y_preds = eval_pred(model, test_loader, DEVICE, name)[0]
    y_true = eval_pred(model, test_loader, DEVICE, name)[1]
    residuals = eval_pred(model, test_loader, DEVICE, name)[2]    
    mse = mean_squared_error(y_true, y_preds)
    r2 = r2_score(y_true, y_preds)

    print(evaluate_model(y_true, y_preds))

    plt.hist(residuals, bins=50)
    plt.title("Residuals distribution")
    plt.show()


    plot_acf(residuals, lags=50)
    plt.title("Residuals ACF")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_preds,alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title("Prediction vs Reality")
    plt.text(0.05, 0.95,f"R² = {r2:.4f}\nMSE = {mse:.4f}", transform=plt.gca().transAxes,
            verticalalignment='top')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------
# Volatility forcasting
# ----------------------------------------------

best_model = model_1
name = "LSTM_2d_HYG.pt"
y_preds = eval_pred(best_model, test_loader, DEVICE, name)[0]
y_true = eval_pred(best_model, test_loader, DEVICE, name)[1]
residuals = eval_pred(best_model, test_loader, DEVICE, name)[2]

y_true_denorm = y_true * y_std.item() + y_mean.item()
y_pred_denorm = y_preds * y_std.item() + y_mean.item()

pd.DataFrame(y_true_denorm).to_pickle("Deep_learning_gold_stock/data/raw/numerical/y_tru_denorm.pkl")

var_true = np.exp(y_true_denorm)
var_pred = np.exp(y_pred_denorm)
vol_pred = np.sqrt(var_pred)


plt.figure(figsize=(12,6))
plt.plot(np.sqrt(np.exp(y_true_denorm)), label="Realized Vol t+2")
plt.plot(vol_pred, label="Forecast Vol t+2")
plt.legend()
plt.title("2-step Ahead Volatility Forecast")
plt.show()

def qlike_loss(var_true, var_pred, eps=1e-12):
    var_pred = np.clip(var_pred, eps, None)
    ratio = var_true / var_pred
    return np.mean(ratio - np.log(ratio) - 1)

qlike_before = qlike_loss(var_true, var_pred)
print("QLIKE before calibration:", qlike_before)


X = sm.add_constant(var_pred)
model_mz = sm.OLS(var_true, X).fit()
print(model_mz.summary())

alpha = model_mz.params[0]
beta = model_mz.params[1]

var_pred_calibrated = alpha + beta * var_pred
var_pred_calibrated = np.clip(var_pred_calibrated, 1e-12, None)

qlike_after = qlike_loss(var_true, var_pred_calibrated)

print("QLIKE after calibration:", qlike_after)

qlike = qlike_loss(var_true, var_pred_calibrated)
rmse = np.sqrt(mean_squared_error(var_true, var_pred_calibrated))
r2 = np.sqrt(r2_score(var_true, var_pred_calibrated))
corr = np.corrcoef(var_true, var_pred_calibrated)[0,1]


print("\n=== Forecast Metrics ===")
print(f"QLIKE: {qlike:.4f}, RMSE: {rmse:.6f}, R2_score: {r2},Correlation: {corr:.4f}")

plt.figure(figsize=(12,5))
plt.plot(np.sqrt(var_true), label="Realized Vol t+2 after calibration", alpha=0.7)
plt.plot(np.sqrt(var_pred_calibrated), label="Forecast Calibrated Vol t+2", alpha=0.7)
plt.title("Forecast vs Realized Volatility")
plt.legend()
plt.show()

# Volatility Timing Strategy Intraday 
vol_calibrated = np.sqrt(var_pred_calibrated)
returns_base = log_ret.values  

returns_base = log_ret.values
HORIZON = 2  # t+2 (10 min ahead)
strategy_returns_base = returns_base[HORIZON:len(vol_calibrated)+HORIZON]
vol_calibrated = vol_calibrated[:len(strategy_returns_base)]


periods_per_day = 78  # 5-min bars intraday

def performance_stats(returns, periods_per_day=78):
    mean_ann = np.mean(returns) * periods_per_day
    vol_ann = np.std(returns) * np.sqrt(periods_per_day)
    sharpe = mean_ann / vol_ann
    neg_returns = returns[returns < 0]
    downside_std = np.std(neg_returns) * np.sqrt(periods_per_day) if len(neg_returns) > 0 else 1e-12
    sortino = mean_ann / downside_std
    cumulative = np.cumsum(returns)
    min_dd = np.min(cumulative - np.maximum.accumulate(cumulative))
    calmar = mean_ann / abs(min_dd) if min_dd != 0 else np.nan
    return mean_ann, vol_ann, sharpe, sortino, calmar

quantiles = np.quantile(vol_calibrated, [0.2, 0.4, 0.6, 0.8])
weights_vt_hybrid = np.zeros_like(vol_calibrated)


weights_vt_hybrid[vol_calibrated <= quantiles[0]] = 1.5
weights_vt_hybrid[(vol_calibrated > quantiles[0]) & (vol_calibrated <= quantiles[1])] = 1.3
weights_vt_hybrid[(vol_calibrated > quantiles[1]) & (vol_calibrated <= quantiles[2])] = 1.0
weights_vt_hybrid[(vol_calibrated > quantiles[2]) & (vol_calibrated <= quantiles[3])] = 0.7
weights_vt_hybrid[vol_calibrated > quantiles[3]] = 0.6

alpha = 0.6  
weights_vt_smooth_hybrid = np.zeros_like(weights_vt_hybrid)
weights_vt_smooth_hybrid[0] = weights_vt_hybrid[0]

for t in range(1, len(weights_vt_hybrid)):
    weights_vt_smooth_hybrid[t] = alpha * weights_vt_hybrid[t] + (1 - alpha) * weights_vt_smooth_hybrid[t-1]

strategy_vt_returns_hybrid = weights_vt_smooth_hybrid * strategy_returns_base
buyhold_returns = strategy_returns_base

# --- Performance Metrics ---
mean_s, vol_s, sharpe_s, sortino_s, calmar_s = performance_stats(strategy_vt_returns_hybrid, periods_per_day)
mean_b, vol_b, sharpe_b, sortino_b, calmar_b = performance_stats(buyhold_returns, periods_per_day)

print("=== Performance Comparison Intraday (Hybrid 5-Quantile + Smooth) ===")
print(f"Strategy Sharpe: {sharpe_s:.4f}, Sortino: {sortino_s:.4f}, Calmar: {calmar_s:.4f}")
print(f"Buy & Hold Sharpe: {sharpe_b:.4f}, Sortino: {sortino_b:.4f}, Calmar: {calmar_b:.4f}")
print(f"Strategy Vol: {vol_s:.4f}, Buy & Hold Vol: {vol_b:.4f}")

# --- Visualisation cumulative ---
cum_strategy = np.cumsum(strategy_vt_returns_hybrid)
cum_bh = np.cumsum(buyhold_returns)

plt.figure(figsize=(14,6))
plt.plot(cum_strategy, label="Hybrid 5-Quantile Smooth", color="blue")
plt.plot(cum_bh, label="Buy & Hold", color="orange")
plt.plot(vol_calibrated/np.max(vol_calibrated)*np.max(cum_strategy),
         label="Normalized Forecast Vol", color="green", linestyle="--")
plt.xlabel("Time steps (5 min)")
plt.ylabel("Cumulative Log Returns / Normalized Vol")
plt.title("Intraday Strategy Hybrid 5-Quantile + Smooth vs Buy & Hold")
plt.legend()
plt.grid(True)
plt.show()
