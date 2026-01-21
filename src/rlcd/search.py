import pandas as pd
import torch
from typing import Tuple, Literal
from copy import deepcopy

from rlcd.config import conf
from rlcd.actions import perform_legal_action
from rlcd.model import QNetwork
from rlcd.scoring import Scorer

def run_episode(
    X: torch.Tensor,
    horizon: int,
    q_online: QNetwork,
    q_target: QNetwork,
    scorer: Scorer
) -> dict[str, torch.Tensor | int]:
    print(f"\nRunning episode with T={horizon}")
    _, d = X.shape

    s = torch.zeros((d, d))
    r = scorer.score(s)
    assert abs(r) < 0.0001 # remove TODO

    for t in range(horizon):
        if t % (horizon // 10) == 0:
            print(f"t = {t}")
        s, a = perform_legal_action(s, q_target)
        r = r - scorer.score(s)
        # Store SARS TODO
    return {"state": s, "score": scorer.score(s)}

def search(df: pd.DataFrame) -> torch.Tensor:
    d = len(df.columns)
    X = torch.tensor(df.values)
    scorer = Scorer(X)
    # Neural network
    q_online = QNetwork(d)
    q_target = deepcopy(q_online)
    # Maintain best seen graph
    best = {"state": torch.zeros((d, d)), "score": 0}
    # Run episodes according to schedule
    for T in conf["epoch_T_schedule"]:
        epsd_best = run_episode(X, T, q_online, q_target, scorer)
        if epsd_best["score"] > best["score"]:
            best = epsd_best.copy()
    print("\nBest state:")
    print(best["state"])
    print(f"with score {best["score"]}")
    return best["state"]