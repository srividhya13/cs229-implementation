import numpy as np
from load_student_data import load_student_data

def batch_gradient_descent(X, y, alpha=0.0001, iterations=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []

    for _ in range(iterations):
        error = X @ theta - y
        theta -= (alpha / m) * (X.T @ error)
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_history.append(cost)

    return theta, cost_history

if __name__ == "__main__":
    X, y = load_student_data()
    theta, costs = batch_gradient_descent(X, y)
    print("Theta (Batch GD):", theta.ravel())
