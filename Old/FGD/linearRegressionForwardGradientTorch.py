
import numpy as np
from numpy import random
import yaml
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp

with open("/home/muskelgurke/PycharmProjects/MA/Old/Old_config.yaml", "r") as file:
    config = yaml.safe_load(file)

learningRate = config["learningRate"]
iterations = config["iterations"]
patience = config["patience"]
precision = config["precision"]
value_a = config["a"]
value_b = config["b"]
data_x = config["x"]
data_y = config["y"]

X_np = np.array(data_x)
ones = np.ones_like(X_np)
X_np = np.stack([X_np, ones], axis=1)
Y_np = np.array(data_y)

def linearFkt(a,b,x):
    return a * x + b

def calculationMSE(theta,X,y):
    Y_pred = X @ theta
    return np.mean((Y_pred - y) ** 2)

# Numerische Richtungsableitung (Forward-Gradient)
def directional_derivative(f, theta, v, X, Y, eps=1e-8):
    """
    Approximiert f'(theta)·v
    """
    return (f(theta + eps*v, X, Y) - f(theta, X, Y)) / eps

def startTrain(plot=False):
    print('\nstarting training linearRegressionnp - Forward Gradient...')
    theta = np.array([value_a, value_b], dtype=float)
    loss_list = []
    counter = 0
    best_loss = np.inf
    save_fg = np.zeros((2, iterations), dtype=float)
    rng = np.random.default_rng(42)

    for i in range(iterations):
        v = rng.normal(size=theta.shape)

        directional_grad = directional_derivative(calculationMSE, theta, v, X_np, Y_np)
        g_theta = directional_grad * v

        # Update speichern
        save_fg[:, i] = g_theta

        # Parameter-Update
        theta = theta - learningRate * g_theta

        # Loss berechnen
        loss = calculationMSE(theta, X_np, Y_np)
        loss_list.append(loss)
        #print(f'theta = {theta}')

        #print(f'Iteration: {i} \n Loss: {loss_list[i]:.4f} \n a: {theta.item(0):.6f}\n b: {theta.item(1):.6f}')
        # print(f'Your New Funktion: y = {a.item():.6f} * X + {b.item():.6f}\n')

        # Early Stopping: Converged to?
        if np.abs(loss-best_loss) < float(precision):
            counter += 1
            if counter >= patience:
                print(f"\n\tStopping early at Iteration {i + 1}-Change < {precision}\n")
                break
        else:
            counter = 0
            best_loss = loss

    #print(f'save_fg: {save_fg}')
    a_learned, b_learned = theta
    # Plotting the loss
    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(loss_list, 'r')
        plt.grid(True, color='y')
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss over iterations")

        # Plotting the final result
        plt.subplot(1, 2, 2)
        plt.scatter(X_np, Y_np, color='blue', label='Target Points')

        x_sorted = np.sort(X_np)
        y_sorted = linearFkt(a_learned,b_learned,x_sorted)

        plt.plot(x_sorted, y_sorted, color='red', label='Learned Line')
        plt.grid(True, color='y')
        plt.legend()
        plt.title(f'Final: y = {a_learned:.2f}x + {b_learned:.2f}')

        plt.tight_layout()
        plt.show()

    return a_learned, b_learned

def getDecimalDigit(value,position):
    as_str = f'{value:.{position + 2}f}'
    return int(as_str.split(".")[1][position - 1])

def standardize_vectors(vectors):
    """Standardize a matrix of vectors (shape: [dimension, count])"""
    means = np.mean(vectors, axis=1, keepdims=True)
    stds = np.std(vectors, axis=1, keepdims=True)
    return (vectors - means) / stds