import matplotlib.pyplot as plt
import pathlib as pl

output_dir = pl.Path(__file__).parent.parent.parent / 'results' / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)

def plot_episode_metrics(
    results_dict: dict[str, list[float]],
    l0: float,
    dag_gt_score: float | None=None
):
    _, ax = plt.subplots()
    for metric_name, results in results_dict.items():
        ax.plot(results, label=metric_name)
    ax.axhline(y=l0, color='b', linestyle='--', label='Score of edge-free graph')
    if dag_gt_score is not None:
        ax.axhline(y=dag_gt_score, color='r', linestyle='--', label='Score of true DAG')
    ax.legend(loc='upper left')
    ax.set_title('')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig('results/figures/episode_results.png')