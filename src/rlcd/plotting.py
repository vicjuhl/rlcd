import matplotlib.pyplot as plt
import pathlib as pl
import numpy as np

from rlcd.config import conf

output_dir = pl.Path(__file__).parent.parent.parent / 'results' / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)

def plot_episode_metrics(
    results_dict: dict[str, dict],
    dag_gt_score: float | None = None
):
    _, ax = plt.subplots()
    axes = [ax]
    colors = plt.cm.tab10.colors

    for i, (metric_name, metric_info) in enumerate(results_dict.items()):
        results = metric_info["results"]
        res_np = np.array([results[0]] * 10 + results)
        roll_avg_res = np.array([res_np[i : i+10].mean() for i in range(len(results))])

        unit = metric_info.get("unit", "")

        if i == 0:
            current_ax = ax
        else:
            current_ax = ax.twinx()
            # offset subsequent axes to the right
            current_ax.spines['right'].set_position(('axes', 1 + 0.1 * (i-1)))
            axes.append(current_ax)

        # plot metric
        color = colors[i % len(colors)]
        current_ax.plot(roll_avg_res, label=f"{metric_name} ({unit})", color=color)
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
    ax.legend(lines, labels, loc='upper left')

    ax.set_xlabel('Episode')
    plt.savefig(output_dir / f'episode_results_{"_".join([k + "=" + str(v) for k, v in conf.items()])}_.png')