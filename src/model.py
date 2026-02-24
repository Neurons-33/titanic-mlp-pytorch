import torch
import torch.nn as nn
import torch.nn.functional as F

# - model -----------------------------------
class Titanic(nn.Module):
    def __init__(self, input_dim, hidden=16, depth=2, dropout=0.3):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, hidden)

        if depth == 2:
            self.fc2 = nn.Linear(hidden, hidden // 2)
            self.fc3 = nn.Linear(hidden // 2, 1)
        else:
            self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        if self.depth == 2:
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            logits = self.fc3(x)
        else:
            logits = self.fc2(x)
        return logits
