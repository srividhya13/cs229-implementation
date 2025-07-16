import numpy as np
from load_student_data import load_student_data

def stochastic_gradient_descent(X, y, alpha=0.01, epochs=50):
    m, n = X.shape
    theta = np.zeros((n, 1))

    for _ in range(epochs):
        for i in range(m):
            xi = X[i].reshape(1, -1)
            yi = y[i]
            error = xi @ theta - yi
            theta -= alpha * xi.T @ error

    return theta

if __name__ == "__main__":
    X, y = load_student_data()
    theta = stochastic_gradient_descent(X, y)
    print("Theta (SGD):", theta.ravel())