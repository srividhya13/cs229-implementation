def dot(a, b):
    return sum(x * y for x, y in zip(a, b))
def predict(X, theta):
    return [dot(x, theta) for x in X]
def compute_cost(X, y, theta):
    m = len(y)
    return sum((dot(x, theta) - yi[0]) ** 2 for x, yi in zip(X, y)) / (2 * m)
def gradient_descent(X, y, alpha=0.01, num_iters=1000):
    m, n = len(X), len(X[0])
    theta = [0.0] * n
    for _ in range(num_iters):
        for j in range(n):
            grad = sum((dot(X[i], theta) - y[i][0]) * X[i][j] for i in range(m)) / m
            theta[j] -= alpha * grad
    return theta
if __name__ == "__main__":
    X = [[1, 1], [1, 2], [1, 3], [1, 4]]
    y = [[3], [5], [7], [9]]
    theta = gradient_descent(X, y, alpha=0.1, num_iters=1000)
    print("weights:", theta)
    print("Predictions:", predict(X, theta))