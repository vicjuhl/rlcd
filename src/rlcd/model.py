import torch
import torch.nn.functional as F

class QNetwork(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # Layer defs here

    def forward(self, x) -> torch.Tensor:
        """Returns a Q-table of dimension (N, N, 3).
        N = |V|, last dimension is 0 = remove, 1 = add, 2 = reverse
        """
        # apply layers here, such as s = F.relu(self.conv1(x))
        return torch.rand((self.d, self.d, 3))

