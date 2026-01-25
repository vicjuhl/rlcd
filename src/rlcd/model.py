import torch
from torch import nn
import torch.nn.functional as F

from rlcd.config import conf

reward_scale = conf["reward_scale"]

class QNetwork(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.fc1 = nn.Linear(d*d, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, d*d*3)
        self.fc3_term = nn.Linear(64, d*d*3)

    def forward(self, s: torch.Tensor, term: torch.Tensor | None=None) -> torch.Tensor:
        """Returns batchsize Q-tables of dimension (bs, d, d, 3).
        Last dimension is 0 = remove, 1 = add, 2 = reverse
        """
        batch_pass = len(s.shape) == 3
        if batch_pass: # batch pass which always has terminal==True
            bs = s.shape[0]
            x = s.view(bs, -1)
        else:
            x = s.flatten()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if batch_pass:
            x_out = torch.zeros((bs, self.d*self.d*3), device=x.device)
            x_out[~term, :] = self.fc3     (x[~term, :])
            x_out[ term, :] = self.fc3_term(x[ term, :])
            return x_out.view(bs, self.d, self.d, 3)
        else:
            x = self.fc3(x)
            return x.view(self.d, self.d, 3)
        

