import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint
#from Deep_learning_gold_stock.src.models.RNN.main import train_validation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
HIDDEN_DIM = 128
HORIZON = 2
GRAD_CLIP = 1.0


# load last BTC data

X_tensor = torch.load("Deep_learning_gold_stock/data/raw/numerical/X_tensor.pt").float()
volat_idx = 0

y_list = []

for i in range(len(X_tensor) - HORIZON):
    # Prédiction HORIZON steps ahead
    y_target = X_tensor[i + HORIZON, -1, volat_idx]  
    y_list.append(y_target)

y_tensor = torch.stack(y_list)  # [num_sequences-HORIZON]

X_tensor = X_tensor[:len(y_tensor),:,1:]  

print("X_tensor:", X_tensor.shape)
print("y_tensor:", y_tensor.shape)

seq_len = X_tensor.shape[1]

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

# ODE function
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, hidden_dim)
        )

    def forward(self, t, h):
        return self.net(h)

class ODE_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)

        self.odefunc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, hidden_dim)
        )

        self.decoder = nn.Linear(hidden_dim, 1)

        self.integration_time = torch.tensor([0.0, 1.0])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            if t > 0:
                h = odeint(lambda t, h: self.odefunc(h),h,self.integration_time.to(x.device),method="rk4")[1]
            h = self.rnn_cell(x[:, t], h)
        residual = self.decoder(h).squeeze() 
        last_return = x[:, -1, 0]

        return residual + last_return


model = ODE_RNN(X_tensor.shape[2], HIDDEN_DIM).to(DEVICE)
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)


best_val_loss = float('inf')
patience = 15
counter = 0

# train_validation("ODERNN_2d_HYG.pt", EPOCHS, model, DEVICE, optimizer, criterion, test_dataset,train_dataset,train_loader,test_loader,GRAD_CLIP, patience, best_val_loss)
