"""
Benchmark the three est_mle_W implementations in the Scorer class.
"""
import torch
import time
from rlcd.scoring import Scorer
from rlcd.gen_data import gen_dag, gen_data

def benchmark_mle_w_implementations(n_samples=1000, n_vars=10, n_runs=10):
    """
    Benchmark each implementation multiple times.
    
    Args:
        n_samples: Number of samples in the data
        n_vars: Number of variables (dimensions)
        n_runs: Number of times to run each implementation
    """
    dag = gen_dag()
    df = gen_data(dag)
    X = torch.tensor(df.values)
    
    scorer = Scorer(X)
    
    s = gen_dag()
    
    print(f"Data shape: {X.shape}")
    print(f"DAG structure sparsity: {(s.sum() / (n_vars * n_vars)):.2%}")
    print(f"Running {n_runs} iterations of each implementation...\n")
    
    # Benchmark est_mle_W_adam
    print("=" * 50)
    print("est_mle_W_adam")
    print("=" * 50)
    times_adam = []
    for i in range(n_runs):
        start = time.perf_counter()
        W_adam = scorer.est_mle_W_adam(s)
        end = time.perf_counter()
        elapsed = end - start
        times_adam.append(elapsed)
        print(f"Run {i+1:2d}: {elapsed:.4f}s")
    
    avg_adam = sum(times_adam) / len(times_adam)
    print(f"Average: {avg_adam:.4f}s")
    print(f"Std dev: {(sum((t - avg_adam)**2 for t in times_adam) / len(times_adam))**0.5:.4f}s")
    print()
    
    # Benchmark est_mle_W_lbfgs
    print("=" * 50)
    print("est_mle_W_lbfgs")
    print("=" * 50)
    times_lbfgs = []
    for i in range(n_runs):
        start = time.perf_counter()
        W_lbfgs = scorer.est_mle_W_lbfgs(s)
        end = time.perf_counter()
        elapsed = end - start
        times_lbfgs.append(elapsed)
        print(f"Run {i+1:2d}: {elapsed:.4f}s")
    
    avg_lbfgs = sum(times_lbfgs) / len(times_lbfgs)
    print(f"Average: {avg_lbfgs:.4f}s")
    print(f"Std dev: {(sum((t - avg_lbfgs)**2 for t in times_lbfgs) / len(times_lbfgs))**0.5:.4f}s")
    print()
    
    # Benchmark est_mle_W_irls
    print("=" * 50)
    print("est_mle_W_irls")
    print("=" * 50)
    times_irls = []
    for i in range(n_runs):
        start = time.perf_counter()
        W_irls = scorer.est_mle_W_irls(s)
        end = time.perf_counter()
        elapsed = end - start
        times_irls.append(elapsed)
        print(f"Run {i+1:2d}: {elapsed:.4f}s")
    
    avg_irls = sum(times_irls) / len(times_irls)
    print(f"Average: {avg_irls:.4f}s")
    print(f"Std dev: {(sum((t - avg_irls)**2 for t in times_irls) / len(times_irls))**0.5:.4f}s")
    print()
    
    # Summary comparison
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    results = [
        ("est_mle_W_adam", avg_adam),
        ("est_mle_W_lbfgs", avg_lbfgs),
        ("est_mle_W_irls", avg_irls),
    ]
    results_sorted = sorted(results, key=lambda x: x[1])
    
    fastest = results_sorted[0][1]
    for name, avg_time in results_sorted:
        speedup = avg_time / fastest if fastest > 0 else 1.0
        print(f"{name:20s}: {avg_time:.4f}s ({speedup:.2f}x)")


if __name__ == "__main__":
    benchmark_mle_w_implementations(n_samples=1000, n_vars=10, n_runs=10)
