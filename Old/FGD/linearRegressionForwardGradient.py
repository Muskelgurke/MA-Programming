import jax
import jax.numpy as jnp
from jax import random
import yaml
import matplotlib.pyplot as plt

with open("../Old_config.yaml", "r") as file:
    config = yaml.safe_load(file)

learningRate = config["learningRate"]
iterations = config["iterations"]
patience = config["patience"]
precision = config["precision"]
value_a = config["a"]
value_b = config["b"]
data_x = config["x"]
data_y = config["y"]

X_jax = jnp.array(data_x)
ones = jnp.ones_like(X_jax)
X_jax = jnp.stack([X_jax,ones], axis=1)
Y_jax = jnp.array(data_y)

def linearFkt(a,b,x):
    return a * x + b

def calculationMSE(theta,X,y):
    Y_pred = X @ theta
    return jnp.mean((Y_pred - y) ** 2)

def startTrain(plot=False):
    print('\nstarting training linearRegressionJax - Forward Gradient...')
    theta = jnp.array([value_a, value_b], dtype=float)
    loss_list = []
    counter = 0
    best_loss = jnp.inf
    save_fg = jnp.zeros((2, iterations), dtype=float)
    key = random.PRNGKey(42)

    for i in range(iterations):
        key, subkey = random.split(key)

        v = random.normal(key, shape=theta.shape,dtype=float)

        f_val, directional_derivative = jax.jvp(lambda th: calculationMSE(th,X_jax,Y_jax),(theta,),(v,))
        loss = f_val
        loss_list.append(loss.item())

        g_theta = directional_derivative * v
        save_fg = save_fg.at[:,i].set(g_theta)
        #print(f'g_theta = {g_theta}')

        theta = theta - learningRate * g_theta
        #print(f'theta = {theta}')

        #print(f'Iteration: {i} \n Loss: {loss_list[i]:.4f} \n a: {theta.item(0):.6f}\n b: {theta.item(1):.6f}')
        # print(f'Your New Funktion: y = {a.item():.6f} * X + {b.item():.6f}\n')

        # Early Stopping: Converged to?
        if jnp.abs(loss-best_loss) < float(precision):
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
        plt.scatter(X_jax, Y_jax, color='blue', label='Target Points')

        x_sorted = jnp.sort(X_jax)
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
    means = jnp.mean(vectors, axis=1, keepdims=True)
    stds = jnp.std(vectors, axis=1, keepdims=True)
    return (vectors - means) / stds