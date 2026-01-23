import torch

def shd(t1: torch.Tensor, t2: torch.Tensor) -> int:
    """Structural Hamming Distance.
    
    missing edge counts 1,
    excess edge counts 1,
    reversed edge counts 2.
    """
    return (t1.int() != t2.int()).sum().item()