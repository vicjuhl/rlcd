import math

n = 1000
d = 4
reward_scale = 1e-3

conf = {
    "k_experiments": 3,
    "T": 10,
    "num_episodes": 15,
    "d": d,
    "n": n,
    "indegree": 3,
    "noise_scale": 1,
    "beta": .02 * d * n * reward_scale,
    "tau_prime": 5,
    "Q_lr": 1e-0,
    "gamma": 0.98,
    "xi": 0.98,
    "batch_size": 32,
    "reward_scale": reward_scale,
    "step_penalty": .002 * d * n * reward_scale
}

for k, v in conf.items():
    print(k, v)