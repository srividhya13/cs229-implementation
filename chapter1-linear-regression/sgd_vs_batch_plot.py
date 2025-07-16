import matplotlib.pyplot as plt

def plot_costs(batch_costs, sgd_costs):
    plt.plot(batch_costs, label="Batch GD")
    plt.plot(sgd_costs, label="SGD", linestyle='--')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("SGD vs Batch GD")
    plt.legend()
    plt.show()