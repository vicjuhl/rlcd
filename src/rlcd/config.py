import math

n = 10000
reward_scale = 1e-3

conf = {
    # "epoch_T_schedule": [10 + 2*i for i in range(20)],
    "epoch_T_schedule": [10] * 100,
    "N": 4,
    "n": n,
    "indegree": 4,
    "noise_scale": 1,
    "beta": math.log(n) * 200,
    "tau": 2, # Not used currently
    "Q_lr": 1e-3,
    "gamma": 0.5,
    "xi": 0.99,
    "batch_size": 32,
    "reward_scale": reward_scale,
    "step_penalty": 100 * reward_scale
}