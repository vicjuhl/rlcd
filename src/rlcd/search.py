import pandas as pd
import torch
from typing import Tuple, Literal
from copy import deepcopy

from rlcd.config import conf
from rlcd.actions import perform_legal_action, filter_illegal_actions, expectation_of_q
from rlcd.model import QNetwork
from rlcd.scoring import Scorer
from rlcd.replay import Transition, ReplayBuffer

def run_episode(
    X: torch.Tensor,
    horizon: int,
    q_online: QNetwork,
    q_target: QNetwork,
    scorer: Scorer,
    memory: ReplayBuffer,
    optim: torch.optim.AdamW,
    criterion: torch.nn.Module
) -> dict[str, torch.Tensor | int]:
    print(f"\nRunning episode with T={horizon}")
    _, d = X.shape
    bs = conf["batch_size"]
    gamma = conf["gamma"]
    xi = conf["xi"]
    # Initial state: no edges
    s = torch.zeros((d, d))
    r = scorer.score(s) # =0

    for t in range(horizon):
        if t % (horizon // 10) == 0:
            print(f"t = {t}")
        s_next, a = perform_legal_action(s, q_target)
        r = r - scorer.score(s_next)
        memory.push(s, a, r, s_next)
        s = s_next
        # Learn
        if len(memory) >= bs:
            batch = memory.sample(bs)
            s_batch = torch.cat(batch.s)
            a_batch = torch.cat(batch.a)
            r_batch = torch.cat(batch.r)
            s_next_batch = torch.cat(batch.s_next)

            # Q values
            qval = q_online.forward(s_batch)
            qval_selected = qval.gather(1, a_batch) # Q value of realized action

            with torch.no_grad():
                qval_target_next = q_target(s_next_batch)
                legal_mask = filter_illegal_actions(qval_target_next)
                s_next_q_expectation = expectation_of_q(qval_target_next, legal_mask)

            td_target = r_batch + s_next_q_expectation * gamma

            loss = criterion(qval_selected, td_target)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Update target weight with Polyak average
            for t, o in zip(q_target.parameters(), q_online.parameters()):
                t.copy_(xi * t + (1 - xi) * o)


    return {"state": s, "score": scorer.score(s)}

def search(df: pd.DataFrame) -> torch.Tensor:
    d = len(df.columns)
    X = torch.tensor(df.values)
    scorer = Scorer(X)
    # Neural network
    q_online = QNetwork(d)
    q_target = deepcopy(q_online)
    q_online.train()
    # Replay and optimimization
    memory = ReplayBuffer(1000)
    optim = torch.optim.AdamW(q_online.parameters(), conf["Q_lr"], amsgrad=True)
    criterion = torch.nn.SmoothL1Loss()
    # Maintain best seen graph
    best = {"state": torch.zeros((d, d)), "score": 0}
    # Run episodes according to schedule
    for T in conf["epoch_T_schedule"]:
        epsd_best = run_episode(X, T, q_online, q_target, scorer, memory, optim, criterion)
        if epsd_best["score"] > best["score"]:
            best = epsd_best.copy()
    print("\nBest state:")
    print(best["state"])
    print(f"with score {best["score"]}")
    return best["state"]