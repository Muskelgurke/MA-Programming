
import numpy as np
from numpy import random
import yaml
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
import torch
from torch.autograd import forward_ad

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

def calculationMSE(theta, X, y):
    Y_pred = X @ theta
    return torch.mean((Y_pred - y) ** 2)

# Numerische Richtungsableitung (Forward-Gradient)

def startTrain(plot=False):
    print('\nstarting training linearRegressionnp - Forward AD...')

    # Initialize parameters as regular tensors (no requires_grad needed for forward AD)
    theta_t = torch.tensor([value_a, value_b], dtype=torch.float32)
    X_t = torch.tensor(X_np, dtype=torch.float32)
    Y_t = torch.tensor(Y_np, dtype=torch.float32)

    loss_list = []
    counter = 0
    best_loss = np.inf
    save_fg = np.zeros((2, iterations), dtype=float)
    rng = np.random.default_rng(1)

    for i in range(iterations):
        # Generate random direction vector
        v = rng.normal(size=theta_t.shape)
        v_t = torch.tensor(v, dtype=torch.float32)

        # Create dual tensor with forward AD
        with forward_ad.dual_level():
            dual_theta = forward_ad.make_dual(theta_t, v_t)

            # Compute function and directional derivative
            dual_loss = calculationMSE(dual_theta, X_t, Y_t)
            f_val = forward_ad.unpack_dual(dual_loss).primal
            directional_derivative = forward_ad.unpack_dual(dual_loss).tangent

        # Compute gradient estimate
        g_theta = directional_derivative * v_t
        save_fg[:, i] = g_theta.detach().numpy()

        # Parameter update
        theta_t = theta_t - learningRate * g_theta

        # Loss calculation
        loss = calculationMSE(theta_t, X_t, Y_t).item()
        loss_list.append(loss)

        # Early stopping logic
        if np.abs(loss - best_loss) < float(precision):
            counter += 1
            if counter >= patience:
                print(f"\n\tStopping early at Iteration {i + 1}-Change < {precision}\n")
                break
        else:
            counter = 0
            best_loss = loss

    a_learned, b_learned = theta_t.detach().numpy()

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