import torch
from torch import nn
import torch.nn.functional as F

from rlcd.config import conf

reward_scale = conf["reward_scale"]
device = conf["device"]

class QNetwork(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.fc1 = nn.Linear(d*d, 128, device=device)
        self.fc2 = nn.Linear(128, 127, device=device)
        self.fc3 = nn.Linear(127 + 1, d*d*3, device=device)

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
            x = torch.cat([x, torch.tensor([0], device=device)])
            x = self.fc3(x)
            return x.view(self.d, self.d, 3)
        

class QGNN(nn.Module): # Incomplete implementation TODO
    def __init__(self, node_in_dim, node_hidden_dim, msg_hidden_dim, attn_dim):
        """
        node_in_dim   : input feature size of nodes
        node_hidden_dim: latent embedding size
        msg_hidden_dim : hidden size of message MLP
        attn_dim       : hidden size for attention MLPs
        """
        super().__init__()
        # Node embedding
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, node_hidden_dim)
        )
        
        # Shared message MLP, flag included
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_hidden_dim + 1, msg_hidden_dim),
            nn.ReLU(),
            nn.Linear(msg_hidden_dim, node_hidden_dim)
        )
        
        # Attention MLPs for parents and children
        self.attn_parents = nn.Sequential(
            nn.Linear(node_hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        
        self.attn_children = nn.Sequential(
            nn.Linear(node_hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
    
    def message(self, h_self, h_other, as_child):
        """
        h_self: [N, H] latent embedding of target node
        h_other: [N, H] latent embedding of source node (parent or child)
        as_child: scalar 0/1 indicating role of h_self
        """
        flag = as_child * torch.ones(h_self.size(0), 1, device=device)
        x = torch.cat([h_self, h_other, flag], dim=-1)
        return self.msg_mlp(x)
    
    def aggregate(self, messages, attn_mlp):
        """
        messages: [num_msgs, H]
        attn_mlp: attention network
        """
        attn_scores = attn_mlp(messages)  # [num_msgs, 1]
        attn_weights = F.softmax(attn_scores, dim=0)  # normalize over messages
        pooled = torch.sum(attn_weights * messages, dim=0)  # weighted sum
        return pooled
    
    def forward(self, x_nodes, parents_list, children_list):
        """
        x_nodes: [num_nodes, node_in_dim] node features
        parents_list: list of lists of parent indices per node
        children_list: list of lists of child indices per node
        """
        N = x_nodes.size(0)
        h = self.node_mlp(x_nodes)  # [N, H]
        h_tilde = []
        
        for i in range(N):
            # Messages from parents
            parent_msgs = []
            for p in parents_list[i]:
                parent_msgs.append(self.message(h[i].unsqueeze(0), h[p].unsqueeze(0), as_child=1))
            if parent_msgs:
                parent_msgs = torch.cat(parent_msgs, dim=0)
                msg_up = self.aggregate(parent_msgs, self.attn_parents)
            else:
                msg_up = torch.zeros_like(h[i], device=device)
            
            # Messages to children
            child_msgs = []
            for c in children_list[i]:
                child_msgs.append(self.message(h[i].unsqueeze(0), h[c].unsqueeze(0), as_child=0))
            if child_msgs:
                child_msgs = torch.cat(child_msgs, dim=0)
                msg_down = self.aggregate(child_msgs, self.attn_children)
            else:
                msg_down = torch.zeros_like(h[i], device=device)
            
            # Concatenate latent + aggregated messages
            h_tilde.append(torch.cat([h[i], msg_up, msg_down], dim=-1))
        
        h_tilde = torch.stack(h_tilde, dim=0)
        return h_tilde
