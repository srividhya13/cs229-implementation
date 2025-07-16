import numpy as np
import matplotlib.pyplot as plt

def feature_scaling(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def plot_scaling_effect(original_X, scaled_X):
    plt.subplot(1, 2, 1)
    plt.scatter(original_X[:, 0], original_X[:, 1])
    plt.title("Original Features")

    plt.subplot(1, 2, 2)
    plt.scatter(scaled_X[:, 0], scaled_X[:, 1])
    plt.title("Scaled Features")

    plt.tight_layout()
    plt.show()