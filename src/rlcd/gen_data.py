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
            dag, success = alter_edge(dag, ((i, j), 1))
    
    # Topological order
    G = nx.DiGraph(dag.numpy())
    order = torch.tensor(list(nx.topological_sort(G)))
    dag = dag[order][:, order]
    return dag

def gen_funcs(dag):
    """Assign uniform [-1, 1] weights to edges; assign uniform [.25, 4] scales."""
    # Sample weights W
    N = conf["N"]
    more_w = torch.empty_like(dag).uniform_(-1, 1)
    w = dag * more_w

    # Sample bias
    c = torch.empty((N,)).uniform_(-1, 1)

    # Sample scale b
    dist = torch.distributions.Gamma(2, 2)
    b = dist.sample((N,))

    return w, c, b

def gen_data():
    N = conf["N"]
    n = conf.get("n_samples", 1000)
    
    dag = gen_dag()
    w, c, b = gen_funcs(dag)

    X = torch.zeros((n, N))
    for i in range(N):
        mu = X @ w[:, i] + c[i]
        e = torch.distributions.Laplace(0.0, b[i]).sample((n,))
        X[:, i] = mu + e / 10

    df = pd.DataFrame(X.numpy(), columns=[f"x{i+1}" for i in range(N)])
    
    # print(dag)
    # print(w)
    # print(c)
    # print(b)
    # print(df)

    return df
