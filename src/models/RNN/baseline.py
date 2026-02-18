from sklearn.metrics import mean_squared_error, r2_score
from arch import arch_model
import torch


returns = torch.load("Deep_learning_gold_stock/data/raw/numerical/X_tensor.pt")[:,-1,0].cpu().numpy()

# --- Baseline GARCH(1,1) ---
garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
garch_res = garch_model.fit(disp='off')

# --- Volatility prediction ---
vol_pred_garch = garch_res.conditional_volatility  


# Metrics
mse = mean_squared_error(returns, vol_pred_garch)
r2 = r2_score(returns, vol_pred_garch)

print(f"GARCH(1,1) baseline - MSE: {mse:.6f}, R^2: {r2:.4f}")
