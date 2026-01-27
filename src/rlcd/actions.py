import torch
from typing import Tuple
from itertools import count

from rlcd.config import conf
from rlcd.model import QNetwork

tau_prime = conf["tau_prime"]

def makes_cycles(s: torch.Tensor, new_edge: Tuple[int, int]) -> bool:
    """Asses whether the addition of new_edge creates cycle(s)."""
    new_parent, new_child = new_edge
    N = s.shape[0]
    affected = torch.zeros(N)
    affected[new_child] = 1 # Follow paths starting at new child
    while affected.max() > 0:
        affected = affected @ s # update which nodes are affected
        if affected[new_parent] > 0: # If any path led to parent
            return True
    return False

def filter_illegal_actions_bruteforce(s: torch.Tensor) -> torch.Tensor:
    """Create legals mask tensor where True = legal edge addition, False = illegal edge addition."""
    d, _ = s.shape
    no_existing_edges = ~s.bool()
    no_self_loops = ~torch.eye(d, dtype=torch.bool)
    no_len2_loops = ~s.T.bool()
    legals = no_self_loops & no_len2_loops & no_existing_edges # all actions legal, except short loops
    unchecked = legals.clone()
    for i in range(d):
        for j in range(d):
            if not unchecked[i, j]:
                continue
            legals[i, j] = not makes_cycles(s, (i, j))
            unchecked[i, j] = False
    return legals

def transitive_closure(s: torch.Tensor) -> torch.Tensor:
    """Compute reachability matrix for a DAG (boolean).
    
    Args:
        s: torch.Tensor of shape (batch_size, d, d)
    
    Returns:
        torch.Tensor of shape (batch_size, d, d) with reachability matrices
    """
    _, d, _ = s.shape
    reachable = s.int()
    for _ in range(d):
        reachable = reachable | torch.bmm(reachable, reachable)
    return reachable.bool()

def filter_illegal_actions(s: torch.Tensor) -> torch.Tensor:
    """
    Create legality mask for actions

    Args:
        s: torch.Tensor of shape (batch_size, d, d) adjacency matrices
    
    Returns:
        torch.Tensor of shape (batch_size, d, d, 3) legality masks
        True  = legal action
        False = illegal (would create a cycle, add existing or remove non-existing edge)
    """
    _, d, _ = s.shape
    s_bool = s.bool()
    
    # Forbid self-loops and addition of existing edges
    no_self_loops = ~torch.eye(d, dtype=torch.bool, device=s.device)  # (d, d)
    no_existing_edges = ~s_bool  # (batch_size, d, d)
    
    # Forbid edges that would create cycles (i, j) when i is reachable from j
    reachable = transitive_closure(s)  # (batch_size, d, d)
    no_new_cycles = ~reachable.transpose(-2, -1)  # (batch_size, d, d)

    # All edges are legal for removal (action type 0)
    removal_legal = s_bool

    # Combine constraints for edge addition (action type 1)
    addition_legal = no_existing_edges & no_self_loops & no_new_cycles  # (batch_size, d, d)
    
    # For reversal (action type 2): edge must exist and reversal shouldn't create cycles
    # Reversing (i, j) removes (i, j) and adds (j, i). Cycle forms if j can reach i.
    reversal_legal = s_bool & ~reachable.transpose(-2, -1)  # (batch_size, d, d)
    
    # Stack the three action types: (batch_size, d, d, 3)
    mask = torch.stack([removal_legal, addition_legal, reversal_legal], dim=-1)
    
    return mask

def alter_edge(
        s_old: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
    """Alter the graph by performing an edge removal (0), addition (1) or reversal (2).
    
    Action is a torch.Tensor of shape (3,) with dtype int: [i, j, action_type]
    
    In addition and reversal cases, checks for new cycles added.

    Returns
        new state: torch.Tensor (which may not have been altered in case of cycles)
        success: bool (whether the graph was indeed updated; no cycles found)
    """
    i, j, a = action[0].item(), action[1].item(), action[2].item()

    s_new = s_old.clone()
    match a:
        case 0: # remove
            s_new[i, j] = 0 # remove edge i, j
            return s_new, True
        case 1: # add
            if (s_old[i, j] - 1).abs().item() < 1e-6: # edge already exists
                return s_old, False
            else:
                s_new[i, j] = 1 # add edge i, j
        case 2: # reverse
            s_new[i, j] = 1 # add edge i, j
            s_new[j, i] = 0 # remove edge j, i
        case _:
            raise ValueError(f"Unexpected action: {a}")
    
    if a in [1, 2] and not makes_cycles(s_new, (i, j)): # no cycles
        return s_new, True
    else: # cycles
        return s_old, False
    
def sample_action(q_shape: torch.Size, pi_flat: torch.Tensor) -> torch.Tensor:
    idx = torch.multinomial(pi_flat, num_samples=1)
    i, j, a = torch.unravel_index(idx, q_shape)
    return torch.stack([i, j, a]).squeeze().to(dtype=torch.int64, device=pi_flat.device)

def expectation_of_q(q_table: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Finds expectation of Q tables, one for each batch element."""
    bs = q_table.shape[0]
    tau = tau_from_q(q_table)

    # Flatten per batch
    q_flat = q_table.clone().view(bs, -1) # (bs, d*d)
    mask_flat = legal_mask.view(bs, -1) # (bs, d*d)

    # 'Remove' masked values
    q_flat[~mask_flat] = float(-1e6) # (bs, d*d)

    pi = torch.softmax(q_flat / tau, dim=1)
    return (pi * q_flat).sum(dim=1) # (bs,)

def tau_from_q(q_table: torch.Tensor) -> float:
    # The most likely action will be e^tau_prime times more probable than the least likely action
    tau = ((q_table.max() - q_table.min()) / tau_prime).item()
    tau = tau if tau > 1e-6 else 1.0
    return tau

def perform_legal_action(
    s: torch.Tensor,
    q_network: QNetwork
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample action based on state.
    
    Returns
        new state: torch.Tensor (d, d) adjacency matrix
        action: torch.Tensor of shape (3,) with dtype int: [i, j, action_type]
                where action_type is 0: remove, 1: add, 2: flip
    """
    d = s.shape[0]
    s_bool = s.bool()
    q_table = q_network.forward(s) # (d, d, 3)
    tau = tau_from_q(q_table)

    # Filter removals and reversals (all existing edges)
    # Some reversals may create cycles, checked below
    mask_removal = s_bool
    mask_flip    = s_bool
    # Filter additions. Only low-hanging fruits: loops of len 1 and 2
    # Some additions may create cycles, checked below
    no_existing_edges = ~s_bool
    no_self_loops = ~torch.eye(d, dtype=torch.bool)
    no_len2_loops = ~(s_bool.T)
    mask_add      = no_self_loops * no_len2_loops * no_existing_edges

    q_table[:, :, 0] = torch.where(mask_removal, q_table[:, :, 0], torch.full_like(q_table[:, :, 0], -1e6))
    q_table[:, :, 2] = torch.where(mask_flip,    q_table[:, :, 2], torch.full_like(q_table[:, :, 2], -1e6))
    q_table[:, :, 1] = torch.where(mask_add,     q_table[:, :, 1], torch.full_like(q_table[:, :, 1], -1e6))

    q_flat = q_table.flatten()
    pi_flat = torch.softmax(q_flat / tau, dim=0)

    # Sample until legal action is obtained
    success = False
    warning_printed = False
    for i in count():
        a = sample_action(q_table.shape, pi_flat)
        s_new, success = alter_edge(s, a)
        if success:
            break
        if i > 100 and not warning_printed:
            print("Failing to find action...")
            print(s.int())
            print(q_table)
            warning_printed = True

    return s_new, a