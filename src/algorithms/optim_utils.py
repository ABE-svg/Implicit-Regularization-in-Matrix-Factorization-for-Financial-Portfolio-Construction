import numpy as np

def trace_psd(X):
    return float(np.trace(X))

def eigenvalues_desc(X):
    return np.sort(np.linalg.eigvalsh(X))[::-1]

def singular_values_desc(U):
    return np.sort(np.linalg.svd(U, compute_uv=False))[::-1]
