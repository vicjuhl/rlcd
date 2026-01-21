import torch
from typing import Tuple

from rlcd.config import conf

class Scorer:
    def __init__(self, X: torch.Tensor):
        _, d = X.shape
        self.beta = conf["beta"]
        self.X = X

        self.l0 = 0.
        self.l0 = self.score(torch.zeros((d, d)), verbose=True) # likelihood of no-edge graph (which has no penalty)

    def score(self, s: torch.Tensor, verbose=False) -> float:
        degree = s.sum()
        W = self.est_mle_W(s)
        b = self.est_mle_b(W)
        l = self.logllhood(b)

        if verbose:
            print(f"l0: {self.l0:.4}")
            print(f"l: {l:.4}")
            print(f"degree: {degree}")
            print(f"penalty: {self.beta * degree}")
            print(f"Z = l - l0 - beta * degree: {l - self.l0 - self.beta * degree}")
            print()

        return l - self.l0 - self.beta * degree

    def est_mle_W(
        self,
        s: torch.Tensor,
        lr=0.05,
        tol=1e-3,
        max_steps=1000,
        verbose=False
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
        _, d = self.X.shape
        W = torch.zeros((d, d), requires_grad=True)
        optimizer = torch.optim.Adam([W], lr=lr)

        prev_W = W.clone().detach()

        for step in range(max_steps):
            optimizer.zero_grad()

            X_pred = self.X @ (W * s)
            resids = (self.X - X_pred).abs()
            loss = resids.sum()  # scalar

            loss.backward()
            optimizer.step()

            # check convergence
            max_change = (W - prev_W).abs().max().item()
            if max_change < tol:
                break
            prev_W = W.clone().detach()

        if max_change > tol:
            print("Warning: weights optimization did not converge\n")
            print(f"Max change {max_change:.2e}")
        elif verbose:
            print(f"Converged in {step} steps.")
            print(f"Max change {max_change:.2e}")

        return (W * s).detach() # masking with DAG s as a safety measure

    def est_mle_b(self, W: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.X - self.X @ W).mean(dim=0).clamp_min(1e-8)

    def logllhood(self, b: torch.Tensor) -> float:
        n, d = self.X.shape
        return - n * (d + torch.log(2 * b).sum()).item()


