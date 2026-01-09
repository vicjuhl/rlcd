import numpy as np
import pandas as pd
import torch
from config import conf

def score(s: np.ndarray) -> float:
    return 42

def run_episode(n_vars: int, horizon: int) -> dict[str, np.ndarray | int]:
    print(f"\nRunning episode with T={horizon}")
    s = np.zeros((n_vars, n_vars))
    for t in range(horizon):
        if t % (horizon // 10) == 0:
            print(f"t = {t}")
    s[1,1] = 1
    return {"state": s, "score": score(s)}

def search(df: pd.DataFrame) -> np.ndarray:
    n_vars = len(df.columns)
    best = {"state": np.zeros((n_vars, n_vars)), "score": 0}
    for T in conf["epoch_T_schedule"]:
        best_epsd = run_episode(n_vars, T)
        if best_epsd["score"] > best["score"]:
            best = best_epsd.copy()
    print("\nBest state:")
    print(best["state"])
    print(f"with score {best["score"]}")
    return best["state"]