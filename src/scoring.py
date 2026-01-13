import torch
from typing import Tuple

from config import conf

def est_mle_b(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    XW = X @ W
    return torch.abs(X - XW).mean(dim=0).clamp_min(1e-8)

def est_mle_W(
    X: torch.Tensor,
    s: torch.Tensor,
    lr=0.05,
    tol=1e-6,
    max_steps=5000,
    verbose=True
) -> torch.Tensor:
    """
    Estimate weight matrix W for Laplace-noise linear model with convergence stopping.
    
    X: (N, d) data
    s: (d, d) mask (bool or 0/1 tensor), True where a parent edge is allowed
    lr: learning rate
    tol: stop when max element-wise change in W < tol
    max_steps: maximum iterations
    verbose: print convergence info
    """
    d = X.shape[1]
    W = torch.zeros((d, d), requires_grad=True)
    optimizer = torch.optim.Adam([W], lr=lr)

    prev_W = W.clone().detach()

    for step in range(max_steps):
        optimizer.zero_grad()

        X_pred = X @ (W * s)
        resids = (X - X_pred).abs()
        loss = resids.sum()  # scalar

        loss.backward()
        optimizer.step()

        # check convergence
        max_change = (W - prev_W).abs().max().item()
        if max_change < tol:
            if verbose:
                print(f"Converged at step {step}, max change {max_change:.2e}")
            break
        prev_W = W.clone().detach()

    return (W * s).detach()

def est_mles(s: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n, d = X.shape
    tol = 1e-6

    b_old = torch.ones(d)
    W_old = torch.zeros_like(s)
    while True:
        b = est_mle_b(X, W_old)
        W = est_mle_W(X, s)
        if (
            (b - b_old).abs().max().item() < tol and
            (W - W_old).abs().max().item() < tol
        ):
            break
        b_old = b.clone()
        W_old = W.clone()

    return b, W

def logllhood(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> float:
    n, d = X.shape
    X_pred = X @ W
    resids = X - X_pred
    return (
        -n * torch.log(2 * b) - 1 / b * resids.abs().sum(dim=0)
    ).sum().item()

def score(s: torch.Tensor, X: torch.Tensor, l_0: float) -> float:
    beta = conf["beta"]
    degree = s.sum()
    b, W = est_mles(s, X)
    l_s = logllhood(X, W, b)

    return l_s - l_0 - beta * degree
