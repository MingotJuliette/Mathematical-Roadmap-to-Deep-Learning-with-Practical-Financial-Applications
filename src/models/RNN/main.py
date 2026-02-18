import yfinance as yf
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import seaborn as sns

data = yf.download("^IXIC",period="60d",interval="5m")   
len(data)

# ---------------------------------
# 1. Data creation
# ---------------------------------

data = data[['Open','High','Low','Close','Volume']]
data.dropna(inplace=True)

data["log_ret"] = np.log(data["Close"] / data["Close"].shift(1))
data["ret"] = data["Close"].pct_change()

print("\n=== EDA: Time delta stats (minutes) ===")
print(data['delta_t'].describe())

data["future_var_2"] = data["log_ret"].shift(-2)**2
eps = 1e-12
data["log_future_var_2"] = np.log(data["future_var_2"] + eps)

data["rv_daily"]   = data["log_ret"]**2
data["rv_weekly"]  = data["log_ret"].rolling(5).apply(lambda x: np.sum(x**2), raw=True)
data["rv_monthly"] = data["log_ret"].rolling(21).apply(lambda x: np.sum(x**2), raw=True)


data["vol_5"]  = data["log_ret"].rolling(5).std()
data["vol_21"] = data["log_ret"].rolling(21).std()
data["vol_63"] = data["log_ret"].rolling(63).std()


data["hl_range"] = (data["High"] - data["Low"]) / data["Close"]
data["co_range"] = (data["Close"] - data["Open"]) / data["Open"]

vix = yf.download("^VIX",period="60d",interval="5m")["Close"]

data["vix"] = vix
data["log_vix"] = np.log(data["vix"])
data["vix_ret"] = np.log(data["vix"] / data["vix"].shift(1))

data["neg_ret"] = (data["log_ret"] < 0).astype(int)
data["neg_ret_sq"] = data["neg_ret"] * data["log_ret"]**2

data["log_rv_weekly"] = np.log(data["rv_weekly"] + eps)
data["log_rv_monthly"] = np.log(data["rv_monthly"] + eps)
data["log_vol_21"] = np.log(data["vol_21"] + eps)

data["vix_spread"] = data["vix"] - np.sqrt(data["rv_monthly"])

data["vol_change_5"] = data["rv_weekly"].pct_change(5)
data["rv_vol"] = data["rv_weekly"].rolling(21).std()
data["parkinson_vol"] = (1/(4*np.log(2))) * (np.log(data["High"]/data["Low"]))**2

data.dropna(inplace=True)

corr = np.abs(data.corr()["log_future_var_2"]).sort_values(ascending=False)
print(corr)

features = ["log_future_var_2", "log_rv_weekly", "log_rv_monthly", "vol_5", "vol_21", "hl_range", "log_vix"]

corr_matrix = data[features].select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True, fmt=".2f",cmap="coolwarm",center=0,linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# ---------------------------------
# 2. Sequences creation
# ---------------------------------

def create_sequences(data, seq_len=50, features=features):
    sequences = []
    delta_ts = []
    for i in range(len(data) - seq_len):
        seq = data[features].iloc[i:i+seq_len].values
        dt = data['delta_t'].iloc[i+1:i+seq_len+1].values  
        sequences.append(seq)
        delta_ts.append(dt)
    return np.array(sequences), np.array(delta_ts)

seq_len = 100
X_seq, delta_t_seq = create_sequences(data, seq_len=seq_len, features=features)

delta_t_seq = np.nan_to_num(delta_t_seq, nan=5.0)
delta_t_seq = np.maximum(delta_t_seq, 1e-3)

print("X_seq shape:", X_seq.shape)
print("delta_t_seq shape:", delta_t_seq.shape)

# 3. torch.tensor conversion
X_tensor = torch.tensor(X_seq, dtype=torch.float32)          
delta_t_tensor = torch.tensor(delta_t_seq, dtype=torch.float32)  

delta_t_tensor = delta_t_tensor.T           
X_tensor.shape
# 4. Downloagind
torch.save(X_tensor, "Deep_learning_gold_stock/data/raw/numerical/X_tensor.pt")
torch.save(delta_t_tensor, "Deep_learning_gold_stock/data/raw/numerical/delta_t_tensor.pt")

print("Tenseurs sauvegardés.")
data["ret"].to_pickle("Deep_learning_gold_stock/data/raw/numerical/ret.pkl")


# ---------------------------------------
# FUNCTIONS 
# ---------------------------------------

def evaluate_model(y_true, y_pred):
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    medae = np.median(np.abs(y_true - y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Pearson correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Directional accuracy
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    # Baseline (predict zero)
    baseline_pred = np.zeros_like(y_true)
    baseline_mse = mean_squared_error(y_true, baseline_pred)
    
    metrics = pd.DataFrame({
        "Metric": [
            "MSE",
            "RMSE",
            "MAE",
            "Median AE",
            "MAPE",
            "R2",
            "Correlation",
            "Directional Accuracy",
            "Baseline MSE"
        ],
        "Value": [
            mse,
            rmse,
            mae,
            medae,
            mape,
            r2,
            corr,
            directional_accuracy,
            baseline_mse
        ]
    })
    
    return metrics


def train_validation(loader_name, EPOCHS, model, DEVICE, optimizer, criterion, test_dataset,train_dataset,train_loader,test_loader,GRAD_CLIP, patience, best_val_loss):
    for epoch in range(EPOCHS):

        print(f"Epoche : {epoch}")
        # ---- TRAIN ----
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        train_loss = total_loss / len(train_dataset)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(test_dataset)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), loader_name)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break


def eval_pred(model, test_loader, DEVICE, loader_name):
    model.load_state_dict(torch.load(loader_name))
    model.eval()

    y_preds = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            y_pred = model(X_batch)

            if y_pred.dim() == 0:
                y_pred = y_pred.unsqueeze(0)
            elif y_pred.dim() > 1:
                y_pred = y_pred.view(-1)

            # Stocker
            y_preds.append(y_pred.cpu())
            y_true.append(y_batch.cpu())

    y_preds_torch = torch.cat(y_preds)
    y_true_torch = torch.cat(y_true)

    y_preds = y_preds_torch.cpu().numpy()
    y_true = y_true_torch.cpu().numpy()

    residuals = (y_true - y_preds)

    return y_preds, y_true, residuals
