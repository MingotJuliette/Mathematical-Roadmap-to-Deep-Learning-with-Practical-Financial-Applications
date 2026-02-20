import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from arch import arch_model

# -------------------------------------------
# Import and prepar data
# -------------------------------------------
log_ret = pd.read_pickle("data/raw/numerical/log_ret.pkl")
log_future_var_2 = pd.read_pickle("data/raw/numerical/log_future_var_2.pkl")

eps = 1e-12
scale = 1000

# log returns
log_ret = log_ret.dropna()*scale # Scaling because value close to zero

target = log_future_var_2.dropna()

# Align indices
df = pd.concat([log_ret, target], axis=1).dropna()
df.columns = ["log_ret", "target"]

# Creat train-test
split = int(0.8 * len(df))
train = df.iloc[:split]
test = df.iloc[split:]

# Fit GARCH(1,1)
model = arch_model(train["log_ret"], vol="Garch", p=1, q=1, dist="normal")
res = model.fit(disp="off")

# ------------------------------------
# rolling to-step hadead forecast
# ------------------------------------
forecasts = []
history = train["log_ret"].copy()

for t in range(len(test)):
    model = arch_model(history, vol="Garch", p=1, q=1, dist="normal")
    res = model.fit(disp="off")
    
    fcast = res.forecast(horizon=2)

    var_2_scaled = fcast.variance.iloc[-1]["h.2"]

    var_2 = var_2_scaled / (scale**2)

    forecasts.append(np.log(var_2 + eps))
        
    # expand window
    history = pd.concat([history, test["log_ret"].iloc[t:t+1]])


# ------------------------------------------
# prediction and evaluation 
# ------------------------------------------
garch_pred = np.array(forecasts)
y_true = test["target"].values

mse = mean_squared_error(y_true, garch_pred)
r2 = r2_score(y_true, garch_pred)

print(f"GARCH(1,1) baseline")
print(f"MSE: {mse:.6f}")
print(f"R²: {r2:.4f}")
