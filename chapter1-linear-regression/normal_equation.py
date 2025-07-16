import numpy as np
from load_student_data import load_student_data

def normal_equation(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

if __name__ == "__main__":
    X, y = load_student_data()
    theta = normal_equation(X, y)
    print("Theta (Normal Equation):", theta.ravel())