import matplotlib.pyplot as plt
import pathlib as pl
import numpy as np
from datetime import datetime
import torch
import json
from typing import Literal

from rlcd.config import conf

output_dir = pl.Path(__file__).parent.parent.parent / 'results'
output_dir.mkdir(parents=True, exist_ok=True)
time_out_dir = output_dir / str(datetime.now())
time_out_dir.mkdir(parents=True, exist_ok=True)

def plot_episode_metrics(
    results_dict: dict[str, dict],
    exp_num: int,
    uniform: bool,
    dag_gt_score: float | None = None
):
    time_unif_out_dir = time_out_dir / f"uniform={uniform}"
    time_unif_out_dir.mkdir(parents=True, exist_ok=True)

    _, ax = plt.subplots(figsize=(9,6))
    axes = [ax]
    colors = plt.cm.tab10.colors

    for i, (metric_name, metric_info) in enumerate(results_dict.items()):
        results = metric_info["results"]
        unit = metric_info["unit"]

        window = conf["num_episodes"] // 20
        res_np = np.array(results)
        roll_avg_res = np.array([res_np[max(0, i-window) : i+1].mean() for i in range(len(results))])

        best_yet = -1e6
        res_best_yet = []

        best_yet = float("inf") if unit == "SHD" else -float("inf")
        cmp = min if unit in ["SHD"] else max # TODO: flawed logic for SHD (finds smallest yet SHD, not SHD of largest yet scoring graph)
        for r in results:
            best_yet = cmp(best_yet, r)
            res_best_yet.append(best_yet)

        if i == 0:
            current_ax = ax
        else:
            current_ax = ax.twinx()
            # offset subsequent axes to the right
            current_ax.spines['right'].set_position(('axes', 1 + 0.1 * (i-1)))
            axes.append(current_ax)

        # plot metric
        color = colors[i % len(colors)]
        current_ax.plot(roll_avg_res, label=f"{metric_name} rolling avg, ({unit})", color=color)
        current_ax.plot(res_best_yet, label=f"{metric_name} best yest, ({unit})", color=color, linestyle='dotted')
        current_ax.set_ylabel(f"{metric_name} ({unit})", color=color)
        current_ax.tick_params(axis='y', colors=color)

        current_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # dag_gt_score only for "Best episode score"
        if dag_gt_score is not None and unit == "graph score":
            current_ax.axhline(y=dag_gt_score, color='r', linestyle='--', label='Score of true DAG')

    # create combined legend
    lines, labels = [], []
    for ax_ in axes:
        line, label = ax_.get_legend_handles_labels()   
        lines += line
        labels += label
    ax.legend(lines, labels, loc='center left', bbox_to_anchor=(1.1, 0.5))

    ax.set_xlabel('Episode')
    plt.subplots_adjust(right=0.9)
    plt.savefig(
        time_unif_out_dir / f'expnum={exp_num}.png',
        bbox_inches="tight"
    )
    plt.close()

def plot_experiment_scores(
    scores: list[list[float]],  # list of score series
    metric_name: Literal["shd", "score"],
    uniform: bool,
    dag_gt_score: float | None = None
):
    time_unif_out_dir = time_out_dir / f"uniform={uniform}"
    time_unif_out_dir.mkdir(parents=True, exist_ok=True)

    title = (
        "Best score of episode" if metric_name == "score" else
        "SHD of the best scoring graph per episode" if metric_name == "shd" else ""
    )

    window = conf["num_episodes"] // 20
    scores_np = [np.array(s) for s in scores]
    colors = plt.cm.tab10.colors

    _, ax = plt.subplots(figsize=(9,6))

    roll_avgs = []
    best_yets = []
    cmp = min if metric_name == "shd" else max
    for i, scr in enumerate(scores_np):
        lbl = f"Series {i+1}"
        
        roll_avg = np.array([scr[max(0, j-window):j+1].mean() for j in range(len(scr))])
        roll_avgs.append(roll_avg)
        ax.plot(roll_avg,  label=f"{lbl}", color=colors[i % len(colors)], lw=.5)
        
        if metric_name == "score":
            best_yet = np.zeros_like(scr)
            best_yet[0] = scr[0]
            for j in range(1, len(scr)):
                best_yet[j] = cmp(best_yet[j-1], scr[j])
            best_yets.append(best_yet)
            ax.plot(best_yet, color=colors[i % len(colors)], lw=.5, linestyle="dashed")

    # median across series at each episode
    all_rolling_arr = np.array(roll_avgs)  # shape (n_series, n_episodes)
    rolling_avg_median = np.median(all_rolling_arr, axis=0)
    ax.plot(rolling_avg_median, label="Median of rolling averages", color='black')
    
    if metric_name == "score":
        all_best_arr = np.array(best_yets)
        best_yet_median = np.median(all_best_arr, axis=0)
        ax.plot(best_yet_median, label="Median of best yet", color='brown')

    if dag_gt_score is not None and metric_name == "score":
        ax.axhline(y=dag_gt_score, color='r', linestyle='--', label='DAG GT Score')

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Best episode score")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.subplots_adjust(right=0.8)
    plt.savefig(time_unif_out_dir / f"global_results_{metric_name}.png", bbox_inches="tight")
    plt.close()

def plot_adj_matrix(adj: torch.Tensor, mat_num: int, uniform: bool):
    """
    Plots a binary adjacency matrix.
    
    Parameters:
        adj (2D array): square adjacency matrix with 0s and 1s
    """
    time_unif_out_dir = time_out_dir / f"uniform={uniform}"
    time_unif_out_dir.mkdir(parents=True, exist_ok=True)

    plt.imshow(adj, cmap='Blues', interpolation='none')
    plt.colorbar(label='Value')
    plt.title(f"Adjacency matrix of {'true DAG' if mat_num == -1 else f'run {mat_num}'}")
    plt.xticks(range(adj.shape[0]))
    plt.yticks(range(adj.shape[0]))
    file_name = "true_DAG" if mat_num == -1 else f"matrix_{mat_num}"
    plt.savefig(time_unif_out_dir / f"{file_name}.png", bbox_inches="tight")
    plt.close()

def dump_info(results: dict, uniform: bool) -> None:
    # Configuration
    with open(time_out_dir / "conf.json", "w") as f:
        json.dump(conf, f, indent=2)

    time_unif_out_dir = time_out_dir / f"uniform={uniform}"
    time_unif_out_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists
    for k, v in results.items():
        if not isinstance(v, list):
            results[k] = v.tolist()
    with open(time_unif_out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)