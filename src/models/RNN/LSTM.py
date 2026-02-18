import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
