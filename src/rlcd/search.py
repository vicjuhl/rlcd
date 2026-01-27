import pandas as pd
import torch
from copy import deepcopy

from rlcd.config import conf
from rlcd.actions import perform_legal_action, filter_illegal_actions, expectation_of_q
from rlcd.model import QNetwork
from rlcd.scoring import Scorer
from rlcd.replay import Transition, ReplayBuffer
from rlcd.utils import shd
from rlcd.plotting import plot_episode_metrics, plot_experiment_scores

step_penalty = conf["step_penalty"]
reward_scale = conf["reward_scale"]
k_experiments = conf["k_experiments"]

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
    score = (scorer.score(s)).reshape(1,) # =0
    best_score = score.clone()

    for t in range(horizon):
        s_next, a = perform_legal_action(s, q_target)
        next_score = scorer.score(s_next)
        r = ((next_score - score) - torch.tensor(step_penalty)).reshape(1,)
        terminal = torch.tensor([(t == horizon - 1)], dtype=bool).reshape(1,)

        memory.push(s, a, r, s_next, terminal)
        s = s_next
        score = next_score.clone()

        if score > best_score:
            s_best[:, :] = s
            best_score = score.clone()
        
        # Learn
        if len(memory) >= bs:
            batch = memory.sample(bs)
            s_batch = torch.stack(batch.s)
            a_batch = torch.stack(batch.a)
            r_batch = torch.cat(batch.r)
            s_next_batch = torch.stack(batch.s_next)
            term_batch = torch.cat(batch.terminal)

            # Q values
            qval = q_online.forward(s_batch, term_batch)  # (bs, d, d, 3)
            # For batched actions, we need to gather based on coordinates [i, j] and action type
            # qval is (bs, d, d, 3), a_batch is (bs, 3)
            bs_size = qval.shape[0]
            i_coords = a_batch[:, 0].long()
            j_coords = a_batch[:, 1].long()
            action_types = a_batch[:, 2].long()
            qval_selected = qval[torch.arange(bs_size), i_coords, j_coords, action_types]  # (bs,)

            with torch.no_grad():
                qval_target_next = q_target.forward(s_next_batch, term_batch)  # (bs, d, d, 3)
                legal_mask = filter_illegal_actions(s_next_batch)  # (bs, d, d, 3)
                s_next_q_expectation = expectation_of_q(qval_target_next, legal_mask)

            td_target = r_batch + (~term_batch).float() * (gamma * s_next_q_expectation)

            # for k in range(1):
            #     print(f"target:\t {td_target[k].item():.4} pred:\t{qval_selected[k].item():.4},\t, r: {r_batch[k]:.4}")

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

def search(exp_num: int, X: torch.Tensor, scorer: Scorer, dag_gt: torch.Tensor | None=None) -> tuple[torch.Tensor, list, list]:
    _, d = X.shape
    # Neural network
    q_online = QNetwork(d)
    q_target = deepcopy(q_online)
    q_online.train()
    # Replay and optimimization
    memory = ReplayBuffer(10000)
    optim = torch.optim.AdamW(q_online.parameters(), conf["Q_lr"], amsgrad=True)
    criterion = torch.nn.SmoothL1Loss()
    # Maintain best seen graph
    best = {"state": torch.zeros((d, d)), "score": torch.zeros((1,))}
    
    # Run episodes according to schedule
    epsd_best_scores = []
    epsd_best_shd = []
    for epsd_num, T in enumerate([conf["T"]] * conf["num_episodes"]):
        print(f"\nRunning episode {epsd_num} with T={T}")
        epsd_best = run_episode(X, T, q_online, q_target, scorer, memory, optim, criterion)
        print(f"Episode best score {epsd_best["score"].item()} with degree {int(epsd_best["score"].sum().item())}")
        if dag_gt is not None:
            shd_epsd = shd(epsd_best["state"], dag_gt)
            print(f"SHD: {shd_epsd}")
        if epsd_best["score"] > best["score"]:
            best = epsd_best.copy()
        
        epsd_best_shd.append(shd_epsd)
        epsd_best_scores.append(epsd_best["score"].item())
    print("\nTrue DAG:")
    if dag_gt is not None:
        print(dag_gt)
        dag_gt_score = scorer.score(dag_gt)
        print(f"with score {dag_gt_score.item()} and degree {int(dag_gt.sum())}")
    else:
        print("... is absent")
    print("\nBest state:")
    print(best["state"])
    print(f"with score {best["score"].item()} and degree {int(dag_gt.sum())}")
    print(f"and SHD: {shd(best["state"], dag_gt)}")

    plot_episode_metrics(
        {
            "Best episode score": {
                "unit": "graph score",
                "results": epsd_best_scores
            },
            "SHD of best scoring graph": {
                "unit": "SHD",
                "results": epsd_best_shd
            }
        }
        , exp_num
        , dag_gt_score=dag_gt_score
    )

    return best["state"], epsd_best_scores, epsd_best_shd

def run_experiements(df: pd.DataFrame, dag_gt: torch.Tensor | None=None) -> torch.Tensor:
    X = torch.tensor(df.values)
    scorer = Scorer(X)
    print(f"\nBaseline score: {scorer.l0 * reward_scale}\n")

    states = []
    scores = []
    shds = []
    for i in range(k_experiments):
        state, exp_scores, exp_shds = search(i, X, scorer, dag_gt)
        states.append(state)
        scores.append(exp_scores)
        shds.append(exp_shds)

    print()    
    print("=" * 40)
    print("Global results")
    print("=" * 40)
    print("\nTrue DAG:")
    print(dag_gt.int())
    print(f"with score {scorer.score(dag_gt)}")


    print("\nBest proposal DAGs")
    for s in states:
        print()
        print(s.int())
        print(f"with score {scorer.score(s)}")
    
    plot_experiment_scores(scores, scorer.score(dag_gt))