import torch
import pandas as pd
import networkx as nx

from rlcd.config import conf
from rlcd.actions import alter_edge

device = conf["device"]

def gen_dag() -> torch.Tensor:
    """Generate topologically ordered DAG."""
    # Generate DAG
    d = conf["d"]
    indeg = conf["indegree"]
    dag = torch.zeros((d,d), device=device)
    for _ in range(indeg):
        success = False
        while not success:
            i, j = torch.randint(0, d, (2,)).tolist()
            action = torch.tensor([i, j, 1], dtype=torch.int64, device=device)  # action type 1 = add
            dag, success = alter_edge(dag, action)
    
    # Topological order
    G = nx.DiGraph(dag.cpu().numpy())
    order = torch.tensor(list(nx.topological_sort(G)), device=device)
    dag = dag[order][:, order]
    return dag

def gen_funcs(dag, noise_scale):
    """Assign weights to edges; assign uniform scales."""
    # Sample weights W
    d = conf["d"]
    w_excess = torch.empty_like(dag).uniform_(-2, 2)
    while (w_excess.abs() < 1).any():
        w_excess[w_excess.abs() < 1] *= 2
    w = dag * w_excess

    # Sample bias
    c = torch.empty((d,)).uniform_(-1, 1)
    c[:] = 0

    # Sample scale b
    dist = torch.distributions.Gamma(2, 2)
    b = dist.sample((d,)).to(device) / 10 * noise_scale

    return w, c, b

def gen_data(dag: torch.Tensor) -> pd.DataFrame:
    d = conf["d"]
    n = conf["n"]
    noise_scale = conf["noise_scale"]
    
    w, c, b = gen_funcs(dag, noise_scale)

    X = torch.zeros((n, d), device=device)
    for i in range(d):
        mu = X @ w[:, i] + c[i]
        e = torch.distributions.Laplace(0.0, b[i]).sample((n,)).to(device)
        X[:, i] = mu + e

    df = pd.DataFrame(X.cpu().numpy(), columns=[f"x{i+1}" for i in range(d)])
    
    print(dag.int())
    print(w)
    print(c)
    print(b)
    print(df)

    return df

def gen_data_from_dag(dag: torch.Tensor, n: int, noise_scale: float):
    d = dag.shape[0]

    # 1. Topological order
    G = nx.DiGraph(dag.cpu().numpy())
    order = torch.tensor(list(nx.topological_sort(G)), device=device)

    # 2. Reorder DAG
    dag_ord = dag[order][:, order]

    # 3. Generate data in topological order
    w, c, b = gen_funcs(dag_ord, noise_scale)
    X = torch.zeros((n, d), device=device)
    for i in range(d):
        mu = X @ w[:, i] + c[i]
        e = torch.distributions.Laplace(0.0, b[i]).sample((n,))
        X[:, i] = mu + e

    # 4. Invert permutation
    inv_order = torch.argsort(order)

    # 5. Restore original column order
    X = X[:, inv_order]

    return X.cpu().numpy()
