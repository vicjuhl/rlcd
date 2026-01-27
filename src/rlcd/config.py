import torch

n = 1000
d = 5
reward_scale = 1e-3
T = d * 2

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device="cpu" # runs faster on my machine

conf = {
    "device": device,
    "k_experiments": 5,
    "T": T,
    "num_episodes": 150,
    "d": d,
    "n": n,
    "indegree": 5,
    "noise_scale": 1,
    "beta": .02 * d * n * reward_scale,
    "tau_prime": 4,
    "Q_lr": 1e-2,
    "gamma": 0.98,
    "xi": 0.98,
    "batch_size": 32,
    "reward_scale": reward_scale,
    "step_penalty": .02 * d * n * reward_scale / T
}

for k, v in conf.items():
    print(k, v)