import numpy as np
import pandas as pd

def search(df: pd.DataFrame) -> np.ndarray:
    n_vars = len(df.columns)
    return np.zeros((n_vars, n_vars))