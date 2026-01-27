import matplotlib.pyplot as plt
import pathlib as pl
import numpy as np
from datetime import datetime
import torch

from rlcd.config import conf

output_dir = pl.Path(__file__).parent.parent.parent / 'results'
output_dir.mkdir(parents=True, exist_ok=True)
time_out_dir = output_dir / str(datetime.now())
time_out_dir.mkdir(parents=True, exist_ok=True)

def plot_episode_metrics(
    results_dict: dict[str, dict],
    exp_num: int,
    dag_gt_score: float | None = None
):
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
        cmp = min if unit in ["SHD"] else max
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
        time_out_dir / f'expnum={exp_num}_episode_results_{"_".join([k + "=" + str(v) for k, v in conf.items()])}_window={window}.png',
        bbox_inches="tight"
    )
    plt.close()

def plot_experiment_scores(
    scores: list[list[float]],  # list of score series
    dag_gt_score: float | None = None
):
    if conf is None or "num_episodes" not in conf:
        raise ValueError("conf must be provided with key 'num_episodes'")

    window = conf["num_episodes"] // 20
    scores_np = [np.array(s) for s in scores]
    colors = plt.cm.tab10.colors


    _, ax = plt.subplots(figsize=(9,6))

    roll_avgs = []
    for i, s in enumerate(scores_np):
        lbl = f"Series {i+1}"
        roll_avg = np.array([s[max(0, j-window):j+1].mean() for j in range(len(s))])
        roll_avgs.append(roll_avg)
        ax.plot(roll_avg, label=f"{lbl} rolling avg", color=colors[i % len(colors)], lw=.5)

    # median across series at each episode
    all_scores_array = np.array(roll_avgs)  # shape (n_series, n_episodes)
    median_across_series = np.median(all_scores_array, axis=0)
    ax.plot(median_across_series, label="Median across series", color='black')

    if dag_gt_score is not None:
        ax.axhline(y=dag_gt_score, color='r', linestyle='--', label='DAG GT Score')

    ax.set_xlabel("Episode")
    ax.set_ylabel("Best episode score")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.subplots_adjust(right=0.8)
    plt.savefig(time_out_dir / f"global_results.png", bbox_inches="tight")
    plt.close()

def plot_adj_matrix(adj: torch.Tensor, w: torch.Tensor):
    """
    Plots a binary adjacency matrix.
    
    Parameters:
        adj (2D array): square adjacency matrix with 0s and 1s
    """
    plt.imshow(adj, cmap='Blues', interpolation='none')
    plt.colorbar(label='Value')
    plt.xticks(range(adj.shape[0]))
    plt.yticks(range(adj.shape[0]))
    plt.savefig(time_out_dir / f"true_DAG.png", bbox_inches="tight")
    plt.close()