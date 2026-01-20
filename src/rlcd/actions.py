import torch
from typing import Tuple, Literal

from rlcd.config import conf

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
    """Compute reachability matrix for a DAG (boolean)."""
    d = s.shape[0]
    reachable = s.int().clone()
    for _ in range(d):
        reachable |= reachable @ reachable
    return reachable.bool()

def filter_illegal_actions(s: torch.Tensor) -> torch.Tensor:
    """
    Create legality mask for adding edges.
    
    True  = legal edge addition
    False = illegal (would create a cycle or self-loop)
    """
    d = s.shape[0]
    # forbid self-loops
    no_self_loops = ~torch.eye(d, dtype=torch.bool)
    no_existing_edges = ~s.bool()
    
    # forbid edges that would create cycles (i, j) when i is reachable from j
    reachable = transitive_closure(s)
    no_new_cycles = ~reachable.T

    return no_existing_edges & no_self_loops & no_new_cycles

def alter_edge(
        s_old: torch.Tensor,
        action: Tuple[Tuple[int, int], Literal[0, 1, 2]]
    ) -> Tuple[torch.Tensor, bool]:
    """Alter the graph by performing an edge removal (0), addition (1) or reversal (2).
    
    In addition and reversal cases, checks for new cycles added.

    Returns
        new state: torch.Tensor (which may not have been altered in case of cycles)
        success: bool (whether the graph was indeed updated; no cycles found)
    """
    coord, a = action
    s_new = s_old.clone()
    match a:
        case 0: # remove
            s_new[coord] = 0 # remove edge i, j
            return s_new, True
        case 1: # add
            s_new[coord] = 1 # add edge i, j
        case 2: # reverse
            s_new[coord] = 1 # add edge i, j
            coord_rev = (coord[1], coord[0])
            s_new[coord_rev] = 0 # remove edge j, i
        case _:
            raise ValueError(f"Unexpected action: {a}")
    
    if a in [1, 2] and not makes_cycles(s_new, coord): # no cycles
        return s_new, True
    else: # cycles
        return s_old, False
    
def sample_action(q_table: torch.Tensor) -> Tuple[Tuple[int, int], Literal[0, 1, 2]]:
    tau = conf["tau"]
    q_flat = q_table.flatten()
    pi_flat = torch.softmax(q_flat / tau, dim=0)
    idx = torch.multinomial(pi_flat, num_samples=1).item()
    i, j, a = torch.unravel_index(idx, q_table.shape)
    return (i.item(), j.item()), a.item()

def perform_legal_action(s: torch.Tensor) -> Tuple[Tuple[int, int],  Literal[0, 1, 2]]:
    """Sample action based on state.
    
    Returns
    (x_1, x_2), a
    where x_1, x_2 are adj mat coordinates and a the action; 0: remove, 1: add, 2 flip
    """
    # q_table = q_target(s)
    d = s.shape[0]
    # Filter the low-hanging fruits: loops of len 1 and 2
    len_1_loops_filter = ~torch.ones((d, d), dtype=torch.bool)
    len_2_loops_filter = ~s.T
    q_table = torch.rand((d, d, 3)) * len_1_loops_filter * len_2_loops_filter
    # Sample until legal action is obtained
    success = False
    while not success:
        action = sample_action(q_table)
        s_new, success = alter_edge(s, action)

    return s_new, action