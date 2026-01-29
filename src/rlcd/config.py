import torch

n = 1000
d = 6
reward_scale = 1e-3
T = 25

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
    "num_episodes": 80,
    "d": d,
    "n": n,
    "indegree": int(d**1.15),
    "noise_scale": .1,
    "beta": .0225 * d * n * reward_scale, # scales poorly with n: becomes overly edge-cauious for large n
    "tau_prime": 5,
    "Q_lr": 1e-2,
    "gamma": 0.98,
    "xi": 0.98,
    "batch_size": 32,
    "reward_scale": reward_scale,
    "step_penalty": .02 * d * n * reward_scale / T
}

for k, v in conf.items():
    print(k, v)