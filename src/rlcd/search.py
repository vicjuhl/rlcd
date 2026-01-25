import pandas as pd
import torch
from copy import deepcopy

from rlcd.config import conf
from rlcd.actions import perform_legal_action, filter_illegal_actions, expectation_of_q
from rlcd.model import QNetwork
from rlcd.scoring import Scorer
from rlcd.replay import Transition, ReplayBuffer
from rlcd.utils import shd
from rlcd.plotting import plot_episode_metrics

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
    _, d = X.shape
    bs = conf["batch_size"]
    gamma = conf["gamma"]
    xi = conf["xi"]
    # Initial state: no edges
    s = torch.zeros((d, d))
    s_best = s.clone()
    latest_score = scorer.score(s).reshape(1,) # =0
    best_score = latest_score.clone()

    for t in range(horizon):
        s_next, a = perform_legal_action(s, q_target)
        r = (scorer.score(s_next) - latest_score)

        memory.push(s, a, r, s_next)
        s = s_next
        latest_score += r

        if latest_score > best_score:
            s_best[:, :] = s
            best_score = latest_score.clone()
        
        # Learn
        if len(memory) >= bs:
            batch = memory.sample(bs)
            s_batch = torch.stack(batch.s)
            a_batch = torch.stack(batch.a)
            r_batch = torch.cat(batch.r)
            s_next_batch = torch.stack(batch.s_next)

            # Q values
            qval = q_online.forward(s_batch)  # (bs, d, d, 3)
            # For batched actions, we need to gather based on coordinates [i, j] and action type
            # qval is (bs, d, d, 3), a_batch is (bs, 3)
            bs_size = qval.shape[0]
            i_coords = a_batch[:, 0].long()
            j_coords = a_batch[:, 1].long()
            action_types = a_batch[:, 2].long()
            qval_selected = qval[torch.arange(bs_size), i_coords, j_coords, action_types]  # (bs,)

            with torch.no_grad():
                qval_target_next = q_target(s_next_batch)  # (bs, d, d, 3)
                legal_mask = filter_illegal_actions(s_next_batch)  # (bs, d, d, 3)
                s_next_q_expectation = expectation_of_q(qval_target_next, legal_mask)

            td_target = r_batch + s_next_q_expectation * gamma

            loss = criterion(qval_selected, td_target)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Update target weight with Polyak average
            with torch.no_grad():
                for t, o in zip(q_target.parameters(), q_online.parameters()):
                    t.copy_(xi * t + (1 - xi) * o)

    assert best_score == scorer.score(s_best), f"{best_score}, {scorer.score(s_best)}"
    return {"state": s_best, "score": best_score}

def search(df: pd.DataFrame, dag_gt: torch.Tensor | None=None) -> torch.Tensor:
    d = len(df.columns)
    X = torch.tensor(df.values)
    scorer = Scorer(X)
    print(f"l0: {scorer.l0}")
    # Neural network
    q_online = QNetwork(d)
    q_target = deepcopy(q_online)
    q_online.train()
    # Replay and optimimization
    memory = ReplayBuffer(100000)
    optim = torch.optim.AdamW(q_online.parameters(), conf["Q_lr"], amsgrad=True)
    criterion = torch.nn.SmoothL1Loss()
    # Maintain best seen graph
    best = {"state": torch.zeros((d, d)), "score": 0}
    
    # Run episodes according to schedule
    epsd_best_scores = []
    epsd_best_shd = []
    for epsd_num, T in enumerate(conf["epoch_T_schedule"]):
        print(f"\nRunning episode {epsd_num} with T={T}")
        epsd_best = run_episode(X, T, q_online, q_target, scorer, memory, optim, criterion)
        print(f"Episode finalized with score {epsd_best["score"].item()}")
        if dag_gt is not None:
            shd_epsd = shd(epsd_best["state"], dag_gt)
            print(f"SHD: {shd_epsd}")
        if epsd_best["score"] > best["score"]:
            best = epsd_best.copy()
        
        epsd_best_shd.append(shd_epsd)
        epsd_best_scores.append(epsd_best["score"])
    print("\nTrue DAG:")
    if dag_gt is not None:
        print(dag_gt)
        dag_gt_score = scorer.score(dag_gt)
        print(f"with score {dag_gt_score} and degree {int(dag_gt.sum())}")
    else:
        print("... is absent")
    print("\nBest state:")
    print(best["state"])
    print(f"with score {best["score"]} and degree {int(dag_gt.sum())}")
    print(f"and SHD: {shd(best["state"], dag_gt)}")

    plot_episode_metrics(
        {"Best episode score": {
            "unit": "graph score",
            "results": epsd_best_scores
        }, "SHD of best scoring graph": {
            "unit": "SHD",
            "results": epsd_best_shd
        }}
        , dag_gt_score=dag_gt_score
    )

    return best["state"]