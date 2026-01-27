import torch
from typing import Tuple

from rlcd.config import conf

reward_scale = conf["reward_scale"]
device = conf["device"]

class Scorer:
    def __init__(self, X: torch.Tensor):
        _, d = X.shape
        self.beta = conf["beta"]
        self.X = X

        # Set l0 to the raw likelihood of the empty graph (baseline for comparison)
        W_empty = self.est_mle_W_lbfgs(torch.zeros((d, d), device=device))
        b_empty = self.est_mle_b(W_empty)
        self.l0 = self.logllhood(b_empty)  # raw likelihood, not scaled

    def score(self, s: torch.Tensor, verbose=False) -> torch.Tensor:
        degree = s.sum()
        W = self.est_mle_W_lbfgs(s)
        b = self.est_mle_b(W)
        l = self.logllhood(b)

        if verbose:
            print(f"l0: {self.l0:.4}")
            print(f"l: {l:.4}")
            print(f"degree: {degree}")
            print(f"penalty: {self.beta * degree}")
            print(f"Z = l - l0 - beta * degree: {l - self.l0 - self.beta * degree}")
            print()

        return (l - self.l0) * reward_scale - self.beta * degree
        
    def est_mle_W_adam(
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
        W = torch.zeros((d, d), requires_grad=True, device=device)
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
    
    def est_mle_W_lbfgs(
        self,
        s: torch.Tensor,
        lr=1.,
        tol=1e-6,
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
        W = torch.zeros((d, d), requires_grad=True, device=device)
        optimizer = torch.optim.LBFGS(
            [W],
            lr=lr,
            max_iter=20,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            X_pred = self.X @ (W * s)
            loss = (self.X - X_pred).abs().sum()
            loss.backward()
            return loss

        prev_W = W.clone().detach()

        for step in range(max_steps):
            optimizer.zero_grad()

            X_pred = self.X @ (W * s)
            resids = (self.X - X_pred).abs()
            loss = resids.sum()  # scalar

            loss.backward()
            optimizer.step(closure=closure)

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
    
    def est_mle_W_irls(
        self,
        s: torch.Tensor,
        tol=1e-6,
        max_steps=50,
        eps=1e-6,
        verbose=False,
    ) -> torch.Tensor:
        """
        Correct IRLS MLE for Laplace LiNGAM.
        Each column solved independently to convergence.
        """
        X = self.X
        _, d = X.shape
        W = torch.zeros((d, d), device=device)

        for j in range(d):
            parents = torch.where(s[:, j] != 0)[0]
            if parents.numel() == 0:
                continue

            Xp = X[:, parents]
            w = torch.zeros(len(parents), device=device)

            prev_obj = float("inf")

            for step in range(max_steps):
                r = X[:, j] - Xp @ w
                obj = r.abs().sum().item()

                # true IRLS weights
                weights = 1.0 / (r.abs() + eps)
                sqrt_w = weights.sqrt()

                WX = Xp * sqrt_w[:, None]
                Wy = X[:, j] * sqrt_w

                new_w = torch.linalg.lstsq(WX, Wy).solution
                delta = (new_w - w).abs().max().item()

                # monotone up to numerical noise
                if obj > prev_obj + 1e-8:
                    break

                w = new_w
                prev_obj = obj

                if delta < tol:
                    break

            W[parents, j] = w

            if verbose:
                print(f"Node {j}: converged in {step} steps")

        return (W * s).detach()



    def est_mle_b(self, W: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.X - self.X @ W).mean(dim=0).clamp_min(1e-8)

    def logllhood(self, b: torch.Tensor) -> float:
        n, d = self.X.shape
        return - n * (d + torch.log(2 * b).sum()).item()


