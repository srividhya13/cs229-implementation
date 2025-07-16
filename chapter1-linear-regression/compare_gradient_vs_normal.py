import numpy as np

def compare_theta(theta_gd, theta_ne):
    diff = np.abs(theta_gd - theta_ne)
    print("Gradient Descent θ:", theta_gd.ravel())
    print("Normal Equation θ:", theta_ne.ravel())
    print("Absolute Difference:", diff.ravel())