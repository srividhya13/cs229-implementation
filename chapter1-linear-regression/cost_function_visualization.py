import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cost_surface(X, y):
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i, t0 in enumerate(theta0_vals):
        for j, t1 in enumerate(theta1_vals):
            theta = np.array([[t0], [t1]])
            error = X @ theta - y
            J_vals[i, j] = (1 / (2 * len(y))) * np.sum(error**2)

    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals.T, cmap='viridis')
    plt.title("Cost Function Surface")
    plt.show()