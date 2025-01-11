import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cost(theta_0, theta_1, X, y):
    m = len(X)
    h = theta_0 + theta_1 * X
    return (1 / (2 * m)) * np.sum((h - y) ** 2)


def compute_batch_gradients(theta_0, theta_1, X, y):
    m = len(X)
    h = theta_0 + theta_1 * X
    d_theta_0 = (1 / m) * np.sum(h - y)
    d_theta_1 = (1 / m) * np.sum((h - y) * X)
    return d_theta_0, d_theta_1


def compute_stochastic_gradients(theta_0, theta_1, X, y):
    m = len(X)
    i = np.random.randint(0, m)
    h = theta_0 + theta_1 * X[i]
    d_theta_0 = h - y[i]
    d_theta_1 = (h - y[i]) * X[i]
    return d_theta_0, d_theta_1


def compute_mini_batch_gradients(theta_0, theta_1, X, y, batch_size):
    m = len(X)
    indices = np.random.choice(m, batch_size, replace=False)
    X_batch, y_batch = X[indices], y[indices]
    h = theta_0 + theta_1 * X_batch
    d_theta_0 = (1 / batch_size) * np.sum(h - y_batch)
    d_theta_1 = (1 / batch_size) * np.sum((h - y_batch) * X_batch)
    return d_theta_0, d_theta_1


def gradient_descent(X, y, alpha, iterations, method='batch', batch_size=1):
    theta_0, theta_1 = 0, 0
    cost_history = []

    for _ in range(iterations):
        if method == 'batch':
            d_theta_0, d_theta_1 = compute_batch_gradients(theta_0, theta_1, X, y)
        elif method == 'stochastic':
            d_theta_0, d_theta_1 = compute_stochastic_gradients(theta_0, theta_1, X, y)
        else:  # mini-batch
            d_theta_0, d_theta_1 = compute_mini_batch_gradients(theta_0, theta_1, X, y, batch_size)

        theta_0 -= alpha * d_theta_0
        theta_1 -= alpha * d_theta_1
        cost = compute_cost(theta_0, theta_1, X, y)
        cost_history.append(cost)

    return theta_0, theta_1, cost_history


def predict(X, theta_0, theta_1):
    return theta_0 + theta_1 * X


def normalize(X):
    return (X - np.mean(X)) / np.std(X)


def plot_data(alpha, iterations, method='batch', batch_size=1):
    X = normalize(pd.read_csv('linearX.csv').values)
    Y = normalize(pd.read_csv('linearY.csv').values)

    theta_0, theta_1, cost_history = gradient_descent(
        X, Y, alpha, iterations, method, batch_size
    )

    # Plot cost history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iterations + 1), cost_history, color='purple', linewidth=2)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.title(f'Cost vs. Iterations (α={alpha}, method={method})', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f'fig_cost_{method}_{alpha}.png')
    plt.close()

    # Plot regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color='blue', label='Data points', alpha=0.6, edgecolors='w', s=100)
    plt.plot(X, predict(X, theta_0, theta_1), color='red', linewidth=2, label='Regression line')
    plt.xlabel('X', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(f'Linear Regression (α={alpha}, method={method})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f'fig_regression_{method}_{alpha}.png')
    plt.close()

    # Print results
    print(f"Method: {method.upper()}")
    print(f"Final parameters: theta_0={theta_0.item():.6f}, theta_1={theta_1.item():.6f}")
    print(f"Final cost: {cost_history[-1].item():.6f}\n")


# Run all experiments
plot_data(alpha=0.5, iterations=50, method='batch')
plot_data(alpha=0.005, iterations=50, method='batch')
plot_data(alpha=5, iterations=50, method='batch')
plot_data(alpha=0.5, iterations=50, method='stochastic')
plot_data(alpha=0.5, iterations=50, method='mini-batch', batch_size=10)