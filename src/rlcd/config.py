import math

n = 100

conf = {
    "epoch_T_schedule": [10, 100],
    "N": 10,
    "n": n,
    "indegree": 16,
    "beta": 1/2 * math.log(n),
    "tau": 1
}