import math

n = 1000

conf = {
    "epoch_T_schedule": [100, 200],
    "N": 10,
    "n": n,
    "indegree": 16,
    "beta": math.log(n) * 50,
    "tau": 1,
    "Q_lr": 1e-3,
    "gamma": 0.98,
    "xi": 0.995,
    "batch_size": 32
}