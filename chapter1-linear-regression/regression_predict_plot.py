import matplotlib.pyplot as plt
from load_student_data import load_student_data
from normal_equation import normal_equation

def plot_predictions(X, y, theta):
    y_pred = X @ theta
    plt.scatter(y, y_pred)
    plt.xlabel("Actual Math Scores")
    plt.ylabel("Predicted Math Scores")
    plt.title("Actual vs Predicted (Linear Regression)")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    X, y = load_student_data()
    theta = normal_equation(X, y)
    plot_predictions(X, y, theta)