import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from Deep_learning_gold_stock.src.models.RNN.main import train_validation

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
X_tensor = torch.load("Deep_learning_gold_stock/data/raw/numerical/X_tensor.pt").float()
volat_idx = 0

y_list = []
for i in range(len(X_tensor) - HORIZON):
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

# calcul des stats sur train uniquement
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

# -----------------------------------
# LSTM Model
# -----------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # out: all hidden states, (h_n, c_n): last hidden & cell states
        out, (h_n, c_n) = self.lstm(x)
        final_hidden = h_n[-1]  # last layer's hidden state at last timestep
        out = self.fc(final_hidden)
        return out.squeeze()
    
# ---------------------------------
# Modelisation
# ---------------------------------
model = LSTMModel(input_dim=X_train.shape[2], hidden_dim=HIDDEN_DIM,
                  num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)

criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

best_val_loss = float('inf')
patience = 15
counter = 0

# train_validation("LSTM_2d_HYG.pt", EPOCHS, model, DEVICE, optimizer, criterion, test_dataset,train_dataset,train_loader,test_loader,GRAD_CLIP, patience,best_val_loss )
