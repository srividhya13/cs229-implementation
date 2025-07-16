import numpy as np
from normal_equation import normal_equation
from log_likelihood import log_likelihood
from load_student_data import load_student_data

def compare_mle_least_squares(X, y):
    theta = normal_equation(X, y)
    ll = log_likelihood(X, y, theta)
    print("MLE = Least Squares")
    print("Log-Likelihood:", ll)
    return theta

if __name__ == "__main__":
    X, y = load_student_data()
    theta = compare_mle_least_squares(X, y)
    print("Theta (MLE ≈ LSQ):", theta.ravel())