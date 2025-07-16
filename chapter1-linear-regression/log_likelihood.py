import numpy as np

def log_likelihood(X, y, theta, sigma=1.0):
    m = len(y)
    residuals = y - X @ theta
    return -m * np.log(np.sqrt(2 * np.pi) * sigma) - (1 / (2 * sigma**2)) * np.sum(residuals**2)