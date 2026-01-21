import torch
from torch import nn
import torch.nn.functional as F

class QNetwork(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.fc1 = nn.Linear(d*d, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, d*d*3)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Returns batchsize Q-tables of dimension (bs, d, d, 3).
        Last dimension is 0 = remove, 1 = add, 2 = reverse
        """
        batch_pass = len(s.shape) == 4
        start_dim = 0
        if batch_pass: # batch pass
            bs = s.shape[0]
            start_dim = 1
        s = s.flatten(start_dim=start_dim)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)
        if batch_pass:
            return s.view(bs, self.d, self.d, 3)
        else:
            return s.view(self.d, self.d, 3)
        

