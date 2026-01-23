import torch
import pandas as pd
import networkx as nx

from rlcd.config import conf
from rlcd.actions import alter_edge

def gen_dag() -> torch.Tensor:
    """Generate topologically ordered DAG."""
    # Generate DAG
    N = conf["N"]
    indeg = conf["indegree"]
    dag = torch.zeros((N,N))
    for _ in range(indeg):
        success = False
        while not success:
            i, j = torch.randint(0, N, (2,)).tolist()
            action = torch.tensor([i, j, 1], dtype=torch.int64)  # action type 1 = add
            dag, success = alter_edge(dag, action)
    
    # Topological order
    G = nx.DiGraph(dag.numpy())
    order = torch.tensor(list(nx.topological_sort(G)))
    dag = dag[order][:, order]
    return dag

def gen_funcs(dag, noise_scale):
    """Assign uniform [-1, 1] weights to edges; assign uniform [.25, 4] scales."""
    # Sample weights W
    N = conf["N"]
    w_excess = torch.empty_like(dag).uniform_(-1, 1) ** 3
    w = dag * w_excess

    # Sample bias
    c = torch.empty((N,)).uniform_(-1, 1)

    # Sample scale b
    dist = torch.distributions.Gamma(2, 2)
    b = dist.sample((N,)) / 10 * noise_scale

    return w, c, b

def gen_data(dag: torch.Tensor) -> pd.DataFrame:
    N = conf["N"]
    n = conf["n"]
    noise_scale = conf["noise_scale"]
    
    w, c, b = gen_funcs(dag, noise_scale)

    X = torch.zeros((n, N))
    for i in range(N):
        mu = X @ w[:, i] + c[i]
        e = torch.distributions.Laplace(0.0, b[i]).sample((n,))
        X[:, i] = mu + e

    df = pd.DataFrame(X.numpy(), columns=[f"x{i+1}" for i in range(N)])
    
    print(dag.int())
    print(w)
    print(c)
    print(b)
    print(df)

    return df

def gen_data_from_dag(dag: torch.Tensor, n: int, noise_scale: float):
    d = dag.shape[0]

    # 1. Topological order
    G = nx.DiGraph(dag.numpy())
    order = torch.tensor(list(nx.topological_sort(G)))

    # 2. Reorder DAG
    dag_ord = dag[order][:, order]

    # 3. Generate data in topological order
    w, c, b = gen_funcs(dag_ord, noise_scale)
    X = torch.zeros((n, d))
    for i in range(d):
        mu = X @ w[:, i] + c[i]
        e = torch.distributions.Laplace(0.0, b[i]).sample((n,))
        X[:, i] = mu + e

    # 4. Invert permutation
    inv_order = torch.argsort(order)

    # 5. Restore original column order
    X = X[:, inv_order]

    return X.numpy()
