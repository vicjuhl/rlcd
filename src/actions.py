import torch
from typing import Tuple, Literal

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