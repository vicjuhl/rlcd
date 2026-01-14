import numpy as np
import pandas as pd
import torch
from typing import Tuple, Literal

from config import conf
from scoring import score
from actions import perform_legal_action

def run_episode(X: torch.Tensor, horizon: int) -> dict[str, torch.Tensor | int]:
    print(f"\nRunning episode with T={horizon}")
    d = X.shape[1]
    s = torch.zeros((d, d))
    l0 = score(s, X, 0)
    r = score(s, X, l0)
    assert abs(r) < 0.0001

    for t in range(horizon):
        if t % (horizon // 10) == 0:
            print(f"t = {t}")
        s, a = perform_legal_action(s)
        r = r - score(s, X, l0)
        # Store SARS TODO
    return {"state": s, "score": score(s, X, l0)}

def search(df: pd.DataFrame) -> torch.Tensor:
    d = len(df.columns)
    X = torch.tensor(df.values)
    best = {"state": torch.zeros((d, d)), "score": 0}
    for T in conf["epoch_T_schedule"]:
        epsd_best = run_episode(X, T)
        if epsd_best["score"] > best["score"]:
            best = epsd_best.copy()
    print("\nBest state:")
    print(best["state"])
    print(f"with score {best["score"]}")
    return best["state"]