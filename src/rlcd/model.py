import torch
from torch import nn
import torch.nn.functional as F

from rlcd.config import conf

reward_scale = conf["reward_scale"]

class QNetwork(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.fc1 = nn.Linear(d*d, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128 + 1, d*d*3)

    def forward(self, s: torch.Tensor, term: torch.Tensor | None=None) -> torch.Tensor:
        """Returns batchsize Q-tables of dimension (bs, d, d, 3).
        Last dimension is 0 = remove, 1 = add, 2 = reverse
        """
        batch_pass = len(s.shape) == 3
        if batch_pass: # batch pass which always has terminal not None
            assert term is not None
            bs = s.shape[0]
            x = s.view(bs, -1)
        else:
            x = s.flatten()
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if batch_pass:
            x = torch.cat([x, term.float().unsqueeze(1)], dim=1)
            x = self.fc3(x)
            return x.view(bs, self.d, self.d, 3)
        else:
            x = torch.cat([x, torch.tensor([0])])
            x = self.fc3(x)
            return x.view(self.d, self.d, 3)
        

