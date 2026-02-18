import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Using last hidden state for stacked GRU
        out, h_n = self.gru(x)
        final_hidden = h_n[-1]  # last layer's hidden state
        out = self.fc(final_hidden)
        return out.squeeze()
