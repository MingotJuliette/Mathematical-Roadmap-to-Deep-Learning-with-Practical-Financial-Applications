import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint

# ---------------------------------------------
# ODE function
# ---------------------------------------------
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
