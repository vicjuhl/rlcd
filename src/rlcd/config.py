import torch

n = 1000
d = 4
reward_scale = 1e-3

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
    "T": 3,
    "num_episodes": 5,
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