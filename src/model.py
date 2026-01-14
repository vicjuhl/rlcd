import torch
import torch.nn.functional as F

class QNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Layer defs here

    def forward(self, x):
        """Returns a Q-table of dimension (N, N, 3).
        N = |V|, last dimension is 0 = remove, 1 = add, 2 = reverse
        """
        # apply layers here, such as s = F.relu(self.conv1(x))
        return x

