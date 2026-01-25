import math

n = 1000
d = 5
reward_scale = 1e-3

conf = {
    # "epoch_T_schedule": [10 + 2*i for i in range(20)],
    "epoch_T_schedule": [20] * 500,
    "d": d,
    "n": n,
    "indegree": 5,
    "noise_scale": 1,
    "beta": .010 * d * n * reward_scale,
    "tau": 2, # Not used currently
    "Q_lr": 1e-3,
    "gamma": 0.9,
    "xi": 0.95,
    "batch_size": 32,
    "reward_scale": reward_scale,
    "step_penalty": .001 * d * n * reward_scale
}

for k, v in conf.items():
    print(k, v)