import numpy as np
import pandas as pd

def gen_data():
    N = 10
    n = 1000
    observations = np.random.uniform(-1, 1, size=(n, N))
    return pd.DataFrame(observations, columns=[f"x{i+1}" for i in range(N)])
