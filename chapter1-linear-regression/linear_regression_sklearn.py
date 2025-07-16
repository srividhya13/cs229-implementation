from sklearn.linear_model import LinearRegression
from load_student_data import load_student_data

def sklearn_regression(X, y):
    model = LinearRegression()
    model.fit(X[:, 1:], y)  # Remove bias column
    return model.coef_, model.intercept_

if __name__ == "__main__":
    X, y = load_student_data()
    coef, intercept = sklearn_regression(X, y)
    print("sklearn coef:", coef)
    print("sklearn intercept:", intercept)